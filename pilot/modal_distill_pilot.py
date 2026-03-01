"""
Run 21: Distillation Pilot — SFT + Trace Generation
=====================================================
LoRA-SFT Qwen2.5-1.5B on R1-Distill traces vs v2 traces,
then generate 64 rollouts per problem from each student.

Tests: Is explorability a property of traces or model state?

Pipeline:
  1. Upload SFT data to Modal volume
  2. LoRA SFT Qwen2.5-1.5B → Student-R1, Student-v2
  3. Merge LoRA weights
  4. Generate 64 rollouts × 60 problems via vLLM
  5. Save traces + summaries

Usage:
  modal run pilot/modal_distill_pilot.py                # full run
  modal run pilot/modal_distill_pilot.py --smoke         # 2 problems, K=4
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import time
from typing import Optional

import modal

app = modal.App("spark-distill-pilot")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0,<5.0.0",
        "datasets",
        "peft>=0.13.0",
        "trl>=0.14.0",
        "accelerate",
        "bitsandbytes",
        "numpy",
        "huggingface_hub",
    )
)

image_vllm = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.4")
    .pip_install(
        "transformers>=4.45.0,<5.0.0",
        "datasets",
        "numpy",
        "huggingface_hub",
    )
)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("spark-pilot-results", create_if_missing=True)

# ── Config ─────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-1.5B"

STUDENTS = [
    {"name": "student-r1", "data_file": "r1-distill.jsonl", "label": "Student-R1 (SFT on R1-Distill traces)"},
    {"name": "student-v2", "data_file": "nemotron-v2.jsonl", "label": "Student-v2 (SFT on Nemotron-v2 traces)"},
]

PASS_K_VALUES = [1, 4, 8, 16, 32, 64]

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)

FULL_CFG = {
    "sft": {
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 4,
        "grad_accum": 4,
        "max_length": 2048,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "warmup_ratio": 0.1,
    },
    "gen": {
        "K": 64,
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
    },
}

SMOKE_CFG = {
    "sft": {
        **FULL_CFG["sft"],
        "num_epochs": 1,
        "max_length": 1024,
    },
    "gen": {
        "K": 4,
        "max_tokens": 2048,
        "temperature": 0.6,
        "top_p": 0.95,
    },
}


# ── Helpers ────────────────────────────────────────────────────


def extract_boxed(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


def normalize_answer(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    s = ans.strip().replace(" ", "").replace(",", "").lower()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return s


def check_answer(pred: Optional[str], gt: str) -> bool:
    p, g = normalize_answer(pred), normalize_answer(gt)
    if not p or not g:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, OverflowError):
        return False


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def count_think_tokens(text: str) -> int:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return len(m.group(1).split()) if m else 0


# ── Phase 1: SFT ──────────────────────────────────────────────


def run_sft(student_cfg: dict, cfg: dict, sft_data_dir: str, out_dir: str):
    """LoRA SFT on a single student."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    name = student_cfg["name"]
    data_path = os.path.join(sft_data_dir, student_cfg["data_file"])
    adapter_dir = os.path.join(out_dir, f"{name}-adapter")
    merged_dir = os.path.join(out_dir, f"{name}-merged")

    sft_cfg = cfg["sft"]

    print(f"\n{'='*60}")
    print(f"SFT: {name} — {student_cfg['label']}")
    print(f"  Data: {data_path}")
    print(f"  Config: lr={sft_cfg['learning_rate']}, epochs={sft_cfg['num_epochs']}, "
          f"r={sft_cfg['lora_r']}, max_len={sft_cfg['max_length']}")
    print(f"{'='*60}")

    t0 = time.time()

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.chat_template = CHATML_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # LoRA config
    peft_config = LoraConfig(
        r=sft_cfg["lora_r"],
        lora_alpha=sft_cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=sft_cfg["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Load dataset
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"  Dataset: {len(dataset)} examples")

    # Training config
    training_args = SFTConfig(
        output_dir=adapter_dir,
        learning_rate=sft_cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=sft_cfg["warmup_ratio"],
        per_device_train_batch_size=sft_cfg["batch_size"],
        gradient_accumulation_steps=sft_cfg["grad_accum"],
        num_train_epochs=sft_cfg["num_epochs"],
        max_length=sft_cfg["max_length"],
        packing=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=25,
        save_strategy="no",  # only save at end
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print(f"  Starting training...")
    train_result = trainer.train()
    train_time = time.time() - t0

    # Log training metrics
    metrics = train_result.metrics
    print(f"\n  Training complete in {train_time:.0f}s")
    print(f"  Loss: {metrics.get('train_loss', '?'):.4f}")
    print(f"  Steps: {metrics.get('train_steps', '?')}")

    # Save adapter
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"  Adapter saved to {adapter_dir}")

    # Merge LoRA into base model
    print(f"  Merging LoRA weights...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"  Merged model saved to {merged_dir}")

    # Clean up
    del trainer, model, merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "name": name,
        "train_time_s": round(train_time, 1),
        "train_loss": metrics.get("train_loss"),
        "n_examples": len(dataset),
        "merged_dir": merged_dir,
    }


# ── Phase 2: Trace Generation ─────────────────────────────────


def generate_student_traces(
    student_name: str, merged_dir: str,
    problems: list[dict], cfg: dict,
) -> dict:
    """Generate K rollouts per problem from a merged student model using vLLM."""
    import torch
    from vllm import LLM, SamplingParams

    gen_cfg = cfg["gen"]
    K = gen_cfg["K"]

    print(f"\n{'='*60}")
    print(f"GENERATING: {student_name}")
    print(f"  {len(problems)} problems × K={K} = {len(problems) * K} rollouts")
    print(f"  T={gen_cfg['temperature']}, top_p={gen_cfg['top_p']}, max_tokens={gen_cfg['max_tokens']}")
    print(f"{'='*60}")

    t0 = time.time()
    llm = LLM(
        model=merged_dir,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        max_model_len=gen_cfg["max_tokens"] + 512,  # prompt + generation room
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # Apply the same ChatML template for prompt construction
    tokenizer.chat_template = CHATML_TEMPLATE
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Build prompts
    prompts = []
    for p in problems:
        msgs = [{
            "role": "user",
            "content": f"{p['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        }]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        ))

    params = SamplingParams(
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        max_tokens=gen_cfg["max_tokens"],
        n=K,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, params)
    gen_time = time.time() - t0
    total = len(problems) * K
    print(f"  Generated {total} rollouts in {gen_time:.1f}s ({total / gen_time:.1f}/s)")

    # Process
    problem_results = []
    for prob_idx, (out_obj, problem) in enumerate(zip(outputs, problems)):
        rollouts = []
        for roll_idx, completion in enumerate(out_obj.outputs):
            text = completion.text
            answer = extract_boxed(text)
            correct = check_answer(answer, problem["answer"])
            rollouts.append({
                "idx": roll_idx,
                "response": text,
                "final_answer": answer,
                "is_correct": correct,
                "n_tokens": len(completion.token_ids),
                "n_think_words": count_think_tokens(text),
            })

        n_correct = sum(r["is_correct"] for r in rollouts)
        unique_ans = set(normalize_answer(r["final_answer"]) for r in rollouts if r["final_answer"])

        pak = {}
        for k in PASS_K_VALUES:
            if k <= K:
                pak[str(k)] = round(pass_at_k(K, n_correct, k), 6)

        problem_results.append({
            "problem_id": problem["problem_id"],
            "n_correct": n_correct,
            "n_unique_answers": len(unique_ans),
            "pass_at_k": pak,
            "rollouts": rollouts,
        })

        if prob_idx % 10 == 0:
            print(f"  [{prob_idx}/{len(problems)}] correct={n_correct}/{K} unique_ans={len(unique_ans)}")

    # Free GPU
    import torch.distributed as dist
    del llm, tokenizer
    if dist.is_initialized():
        dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "load_time": load_time,
        "gen_time": gen_time,
        "problem_results": problem_results,
    }


# ── Load problems ─────────────────────────────────────────────


def load_problems_from_volume(smoke: bool = False) -> list[dict]:
    """Load the same 60 problems from Run 14."""
    problems_path = "/results/gen_traces/full/problems.json"
    with open(problems_path) as f:
        problems = json.load(f)
    if smoke:
        # Take 1 easy + 1 hard for minimal test
        easy = [p for p in problems if p["tier"] == "easy"][:1]
        hard = [p for p in problems if p["tier"] == "hard"][:1]
        return easy + hard
    return problems


# ── Summary ────────────────────────────────────────────────────


def compute_summary(student_name: str, label: str, problem_results: list[dict],
                    problems: list[dict], K: int, load_time: float, gen_time: float) -> dict:
    n = len(problem_results)
    agg_pass = {}
    for k in PASS_K_VALUES:
        sk = str(k)
        if k <= K:
            vals = [r["pass_at_k"].get(sk, 0) for r in problem_results]
            agg_pass[sk] = round(sum(vals) / len(vals), 4)

    summary = {
        "model": student_name,
        "label": label,
        "n_problems": n,
        "K": K,
        "pass_at_k": agg_pass,
        "mean_correct": round(sum(r["n_correct"] for r in problem_results) / n, 2),
        "mean_unique_answers": round(sum(r["n_unique_answers"] for r in problem_results) / n, 2),
        "timing": {"load_s": round(load_time, 1), "gen_s": round(gen_time, 1)},
    }

    # Per-tier
    tiers = {}
    for tier in ["easy", "medium", "hard"]:
        tier_ids = {p["problem_id"] for p in problems if p["tier"] == tier}
        tier_res = [r for r in problem_results if r["problem_id"] in tier_ids]
        if not tier_res:
            continue
        nt = len(tier_res)
        tier_pass = {}
        for k in PASS_K_VALUES:
            sk = str(k)
            if k <= K:
                vals = [r["pass_at_k"].get(sk, 0) for r in tier_res]
                tier_pass[sk] = round(sum(vals) / len(vals), 4)
        tiers[tier] = {
            "n": nt,
            "pass_at_k": tier_pass,
            "mean_correct": round(sum(r["n_correct"] for r in tier_res) / nt, 2),
        }
    summary["per_tier"] = tiers
    return summary


# ── Main pipeline ──────────────────────────────────────────────


def _run_pipeline(cfg: dict, smoke: bool = False):
    """Full pipeline: SFT both students, then generate traces from each."""
    os.environ["HF_HOME"] = "/hf-cache"

    tag = "smoke" if smoke else "full"
    out_dir = f"/results/distill_pilot/{tag}"
    sft_dir = os.path.join(out_dir, "sft")
    traces_dir = os.path.join(out_dir, "traces")
    os.makedirs(sft_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    sft_data_dir = "/results/distill_pilot/sft_data"
    run_start = time.time()

    K = cfg["gen"]["K"]
    problems = load_problems_from_volume(smoke=smoke)

    print(f"\n{'='*60}")
    print(f"RUN 21: DISTILLATION PILOT [{tag}]")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Students: {[s['name'] for s in STUDENTS]}")
    print(f"  Problems: {len(problems)}, K={K}")
    print(f"  SFT config: {cfg['sft']}")
    print(f"{'='*60}")

    # ── Phase 1: SFT ──
    sft_results = []
    for student in STUDENTS:
        result = run_sft(student, cfg, sft_data_dir, sft_dir)
        sft_results.append(result)
        results_vol.commit()

    # ── Phase 2: Generation (needs vLLM, separate from SFT) ──
    # We've merged the LoRA weights, so we can load with vLLM
    gen_results = []
    model_summaries = []

    for student, sft_result in zip(STUDENTS, sft_results):
        name = student["name"]
        merged_dir = sft_result["merged_dir"]
        model_dir = os.path.join(traces_dir, name)
        os.makedirs(model_dir, exist_ok=True)

        gen = generate_student_traces(name, merged_dir, problems, cfg)

        # Save traces
        traces_path = os.path.join(model_dir, "traces.json")
        with open(traces_path, "w") as f:
            json.dump({
                "model": {"name": name, "label": student["label"], "base": BASE_MODEL},
                "timestamp": int(time.time()),
                "K": K,
                "sft_config": cfg["sft"],
                "gen_config": cfg["gen"],
                "sft_metrics": {
                    "train_loss": sft_result["train_loss"],
                    "train_time_s": sft_result["train_time_s"],
                    "n_examples": sft_result["n_examples"],
                },
                "problems": gen["problem_results"],
            }, f)
        size_mb = os.path.getsize(traces_path) / (1024 * 1024)
        print(f"  Saved: {traces_path} ({size_mb:.1f}MB)")

        # Summary
        summary = compute_summary(
            name, student["label"], gen["problem_results"],
            problems, K, gen["load_time"], gen["gen_time"],
        )
        with open(os.path.join(model_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        model_summaries.append(summary)

        pk = summary["pass_at_k"]
        print(f"\n  {name}: pass@1={pk.get('1','?')} pass@{K}={pk.get(str(K),'?')} "
              f"unique_ans={summary['mean_unique_answers']}")
        for tier, td in summary.get("per_tier", {}).items():
            tpk = td["pass_at_k"]
            print(f"    {tier}: pass@1={tpk.get('1','?')} mean_correct={td['mean_correct']}/{K}")

        gen_results.append(gen)
        results_vol.commit()

    # ── Save problems ──
    with open(os.path.join(traces_dir, "problems.json"), "w") as f:
        json.dump(problems, f, indent=2)

    # ── Global summary ──
    total_time = time.time() - run_start
    global_summary = {
        "tag": tag,
        "timestamp": int(time.time()),
        "base_model": BASE_MODEL,
        "total_time_s": round(total_time, 1),
        "n_students": len(STUDENTS),
        "n_problems": len(problems),
        "K": K,
        "sft_config": cfg["sft"],
        "gen_config": cfg["gen"],
        "sft_results": [{k: v for k, v in r.items() if k != "merged_dir"} for r in sft_results],
        "model_summaries": model_summaries,
    }
    with open(os.path.join(out_dir, "global_summary.json"), "w") as f:
        json.dump(global_summary, f, indent=2)
    results_vol.commit()

    # ── Comparison table ──
    print(f"\n{'='*70}")
    print("STUDENT COMPARISON")
    print(f"{'='*70}")
    header_ks = [k for k in PASS_K_VALUES if k <= K]
    hdr = f"{'Student':<16}" + "".join(f" {'p@'+str(k):>7}" for k in header_ks) + f" {'uniq':>6} {'loss':>7}"
    print(hdr)
    print("-" * len(hdr))
    for s, sft in zip(model_summaries, sft_results):
        pk = s["pass_at_k"]
        row = f"{s['model']:<16}"
        for k in header_ks:
            row += f" {pk.get(str(k), 0):>7.3f}"
        row += f" {s['mean_unique_answers']:>6.1f}"
        row += f" {sft['train_loss']:>7.4f}" if sft.get("train_loss") else "    ???"
        print(row)

    print(f"\nTotal: {total_time/60:.1f}min | Output: {out_dir}")
    return global_summary


# ── Upload SFT data ────────────────────────────────────────────


@app.function(
    image=image,
    volumes={"/results": results_vol},
    timeout=60,
)
def upload_sft_data(data_r1: bytes, data_v2: bytes):
    """Upload SFT data files to the Modal volume."""
    out_dir = "/results/distill_pilot/sft_data"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "r1-distill.jsonl"), "wb") as f:
        f.write(data_r1)
    with open(os.path.join(out_dir, "nemotron-v2.jsonl"), "wb") as f:
        f.write(data_v2)
    results_vol.commit()
    print(f"Uploaded SFT data: r1={len(data_r1)} bytes, v2={len(data_v2)} bytes")
    return True


# ── Modal entrypoints ──────────────────────────────────────────


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,  # 6 hours max
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
)
def run_sft_phase(smoke: bool = False):
    """Phase 1: SFT only (produces merged models)."""
    os.environ["HF_HOME"] = "/hf-cache"
    cfg = SMOKE_CFG if smoke else FULL_CFG
    tag = "smoke" if smoke else "full"
    sft_dir = f"/results/distill_pilot/{tag}/sft"
    sft_data_dir = "/results/distill_pilot/sft_data"
    os.makedirs(sft_dir, exist_ok=True)

    sft_results = []
    for student in STUDENTS:
        result = run_sft(student, cfg, sft_data_dir, sft_dir)
        sft_results.append(result)
        results_vol.commit()

    return sft_results


@app.function(
    image=image_vllm,
    gpu="A10G",
    timeout=60 * 60 * 4,  # 4 hours for generation
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
)
def run_gen_phase(smoke: bool = False):
    """Phase 2: Generate traces from merged models."""
    os.environ["HF_HOME"] = "/hf-cache"
    cfg = SMOKE_CFG if smoke else FULL_CFG
    tag = "smoke" if smoke else "full"

    sft_dir = f"/results/distill_pilot/{tag}/sft"
    traces_dir = f"/results/distill_pilot/{tag}/traces"
    os.makedirs(traces_dir, exist_ok=True)

    K = cfg["gen"]["K"]
    problems = load_problems_from_volume(smoke=smoke)

    model_summaries = []
    for student in STUDENTS:
        name = student["name"]
        merged_dir = os.path.join(sft_dir, f"{name}-merged")
        model_dir = os.path.join(traces_dir, name)
        os.makedirs(model_dir, exist_ok=True)

        gen = generate_student_traces(name, merged_dir, problems, cfg)

        # Save traces
        traces_path = os.path.join(model_dir, "traces.json")
        with open(traces_path, "w") as f:
            json.dump({
                "model": {"name": name, "label": student["label"], "base": BASE_MODEL},
                "timestamp": int(time.time()),
                "K": K,
                "problems": gen["problem_results"],
            }, f)

        # Summary
        summary = compute_summary(
            name, student["label"], gen["problem_results"],
            problems, K, gen["load_time"], gen["gen_time"],
        )
        with open(os.path.join(model_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        model_summaries.append(summary)

        pk = summary["pass_at_k"]
        print(f"\n  {name}: pass@1={pk.get('1','?')} pass@{K}={pk.get(str(K),'?')}")

        results_vol.commit()

    # Save problems
    with open(os.path.join(traces_dir, "problems.json"), "w") as f:
        json.dump(problems, f, indent=2)
    results_vol.commit()

    return model_summaries


@app.local_entrypoint()
def main(smoke: bool = False):
    from pathlib import Path

    # Step 1: Upload SFT data
    print("Uploading SFT data to Modal volume...")
    data_dir = Path("data/sft_data")
    data_r1 = (data_dir / "r1-distill.jsonl").read_bytes()
    data_v2 = (data_dir / "nemotron-v2.jsonl").read_bytes()
    upload_sft_data.remote(data_r1, data_v2)
    print("  Upload complete.")

    # Step 2: SFT phase
    if smoke:
        print("\nRunning SMOKE SFT (1 epoch, short sequences)...")
    else:
        print("\nRunning FULL SFT (3 epochs, 2048 seq length)...")
    sft_results = run_sft_phase.remote(smoke=smoke)
    print("\nSFT Results:")
    for r in sft_results:
        print(f"  {r['name']}: loss={r.get('train_loss', '?')}, time={r['train_time_s']}s, n={r['n_examples']}")

    # Step 3: Generation phase
    if smoke:
        print("\nRunning SMOKE generation (K=4, 2 problems)...")
    else:
        print("\nRunning FULL generation (K=64, 60 problems)...")
    summaries = run_gen_phase.remote(smoke=smoke)
    print("\nGeneration Results:")
    for s in summaries:
        pk = s["pass_at_k"]
        print(f"  {s['model']}: pass@1={pk.get('1','?')} pass@64={pk.get('64','?')} "
              f"unique_ans={s['mean_unique_answers']}")

    print("\nDone! Download results with:")
    tag = "smoke" if smoke else "full"
    print(f"  modal volume get spark-pilot-results distill_pilot/{tag}/ data/modal_runs/distill_pilot/{tag}/")
