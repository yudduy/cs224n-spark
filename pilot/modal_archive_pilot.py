"""
Archive-Guided Strategy Discovery Pilot
========================================
2 models × 10 problems × 3 iterations × 16 rollouts = 960 traces.
Uses strategy archives from baseline LLM judge to prompt models for
fundamentally different approaches.

Output (Modal volume spark-pilot-results → /archive_pilot/):
  config.json
  {model_name}/traces.json

Usage:
  modal run pilot/modal_archive_pilot.py               # full pilot
  modal run pilot/modal_archive_pilot.py --smoke        # 2 problems, 1 iter, n=4
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from typing import Optional

import modal

app = modal.App("spark-archive-pilot")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.4")
    .pip_install(
        "transformers>=4.45.0,<5.0.0",
        "huggingface_hub",
    )
)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("spark-pilot-results", create_if_missing=True)

# ── Models (R1-Distill vs Nemotron v2 — maximum contrast) ─────

MODELS = [
    {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "name": "r1-distill",
        "revision": None,
        "label": "R1-Distill (base)",
    },
    {
        "hf_id": "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
        "name": "nemotron-v2",
        "revision": None,
        "label": "Nemotron v2 (ProRL 3K steps)",
    },
]

# ── Configuration ──────────────────────────────────────────────

FULL_CFG = {
    "n_iterations": 3,
    "K": 16,  # rollouts per iteration
    "max_tokens": 4096,
    "temperature": 0.6,  # DeepSeek recommended (controlled diversity)
    "top_p": 0.95,
}

SMOKE_CFG = {
    **FULL_CFG,
    "n_iterations": 1,
    "K": 4,
}

# ── Helpers (from modal_gen_traces.py) ─────────────────────────


def extract_boxed(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


def normalize_answer(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    return ans.strip().replace(" ", "").replace(",", "").lower()


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


def count_think_tokens(text: str) -> int:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return len(m.group(1).split()) if m else 0


# ── Prompt building ────────────────────────────────────────────


def build_archive_prompt(tokenizer, problem_text: str, archive: dict, iteration: int) -> str:
    """Build archive-guided prompt. All content in user message (no system prompt)."""
    strategies = archive.get("strategies", [])

    # Build strategy section
    strat_lines = []
    for s in strategies:
        strat_lines.append(f"### Strategy {s['id']}: {s['name']}")
        strat_lines.append(s["description"])
        strat_lines.append("")

    strat_block = "\n".join(strat_lines).strip()

    # Core prompt
    parts = [
        "Previous solvers have found these approaches for the problem below:",
        "",
        strat_block,
        "",
        "Your task: solve this problem using a FUNDAMENTALLY DIFFERENT mathematical approach than any listed above.",
    ]

    # Iteration escalation
    if iteration > 1:
        parts.append("")
        parts.append(
            "Note: additional attempts have been made. "
            "Try harder to find a genuinely novel approach."
        )

    parts.extend([
        "",
        problem_text,
        "Please reason step by step, and put your final answer within \\boxed{}.",
    ])

    content = "\n".join(parts)
    msgs = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def build_baseline_prompt(tokenizer, problem_text: str) -> str:
    """Standard prompt without archive (for comparison baseline within pilot)."""
    msgs = [{
        "role": "user",
        "content": f"{problem_text}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    }]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


# ── Core generation ────────────────────────────────────────────


def run_archive_pilot(models: list[dict], cfg: dict, archive_path: str, tag: str):
    """Generate archive-guided traces for all models."""
    import torch
    from vllm import LLM, SamplingParams

    os.environ["HF_HOME"] = "/hf-cache"

    out_dir = f"/results/archive_pilot/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    run_start = time.time()
    ts = int(run_start)

    K = cfg["K"]
    n_iters = cfg["n_iterations"]

    # Load archives
    with open(archive_path) as f:
        archives = json.load(f)

    pilot_pids = sorted(int(k) for k in archives.keys())
    n_problems = len(pilot_pids)
    total_rollouts = len(models) * n_problems * n_iters * K

    print(f"\n{'='*60}")
    print(f"ARCHIVE-GUIDED PILOT [{tag}]")
    print(f"  {len(models)} models × {n_problems} problems × {n_iters} iters × K={K}")
    print(f"  Total: {total_rollouts} rollouts")
    print(f"{'='*60}")

    # Save config
    config = {
        "cfg": cfg,
        "models": [m["name"] for m in models],
        "pilot_pids": pilot_pids,
        "timestamp": ts,
        "tag": tag,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    results_vol.commit()

    for mi, model_cfg in enumerate(models):
        name = model_cfg["name"]
        model_dir = os.path.join(out_dir, name)
        os.makedirs(model_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"MODEL: {name} — {model_cfg['label']}")
        print(f"{'='*60}")

        # Load model
        t0 = time.time()
        llm_kwargs = dict(
            model=model_cfg["hf_id"],
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            dtype="bfloat16",
            max_model_len=5120,  # slightly larger for archive prompt
            trust_remote_code=True,
        )
        if model_cfg["revision"]:
            llm_kwargs["revision"] = model_cfg["revision"]

        llm = LLM(**llm_kwargs)
        tokenizer = llm.get_tokenizer()
        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s")

        params = SamplingParams(
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
            n=K,
        )

        # Generate per problem per iteration
        problem_results = []
        gen_start = time.time()

        for pi, pid in enumerate(pilot_pids):
            arch = archives[str(pid)]
            problem_text = arch["problem_text"]
            answer = arch["answer"]

            iterations = []
            for it in range(1, n_iters + 1):
                prompt = build_archive_prompt(tokenizer, problem_text, arch, it)

                outputs = llm.generate([prompt], params)
                out_obj = outputs[0]

                rollouts = []
                for roll_idx, completion in enumerate(out_obj.outputs):
                    text = completion.text
                    pred = extract_boxed(text)
                    correct = check_answer(pred, answer)
                    rollouts.append({
                        "idx": roll_idx,
                        "response": text,
                        "final_answer": pred,
                        "is_correct": correct,
                        "n_tokens": len(completion.token_ids),
                        "n_think_words": count_think_tokens(text),
                    })

                n_correct = sum(r["is_correct"] for r in rollouts)
                iterations.append({
                    "iter": it,
                    "rollouts": rollouts,
                    "n_correct": n_correct,
                })

            total_correct = sum(it_data["n_correct"] for it_data in iterations)
            total_rollouts_prob = sum(len(it_data["rollouts"]) for it_data in iterations)

            problem_results.append({
                "problem_id": pid,
                "problem_text": problem_text,
                "answer": answer,
                "tier": arch.get("tier", ""),
                "subject": arch.get("subject", ""),
                "iterations": iterations,
                "total_correct": total_correct,
                "total_rollouts": total_rollouts_prob,
                "archive_strategies": arch["strategies"],
            })

            print(f"  P{pid:2d}: {total_correct}/{total_rollouts_prob} correct "
                  f"(iters: {[it['n_correct'] for it in iterations]})")

        gen_time = time.time() - gen_start

        # Save traces
        traces_data = {
            "model": model_cfg,
            "timestamp": ts,
            "cfg": cfg,
            "load_time_s": round(load_time, 1),
            "gen_time_s": round(gen_time, 1),
            "problems": problem_results,
        }
        traces_path = os.path.join(model_dir, "traces.json")
        with open(traces_path, "w") as f:
            json.dump(traces_data, f)
        size_mb = os.path.getsize(traces_path) / (1024 * 1024)
        print(f"\n  Saved {size_mb:.1f}MB to {traces_path}")
        print(f"  Gen time: {gen_time:.1f}s ({gen_time/60:.1f}min)")

        # Free GPU
        import torch.distributed as dist
        del llm, tokenizer
        if dist.is_initialized():
            dist.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"  GPU freed.")

        results_vol.commit()

    total_time = time.time() - run_start
    print(f"\n{'='*60}")
    print(f"DONE. Total time: {total_time/60:.1f}min")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    return {"tag": tag, "total_time_s": round(total_time, 1)}


# ── Modal entrypoints ─────────────────────────────────────────


# Upload local archive file to Modal volume before running
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 2,
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
)
def run_full(archive_json: str):
    # Write archive to temp file
    archive_path = "/tmp/strategy_archives.json"
    with open(archive_path, "w") as f:
        f.write(archive_json)
    return run_archive_pilot(MODELS, FULL_CFG, archive_path, "full")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 1,
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
)
def run_smoke(archive_json: str):
    archive_path = "/tmp/strategy_archives.json"
    with open(archive_path, "w") as f:
        f.write(archive_json)
    # Smoke: only first 2 problems
    with open(archive_path) as f:
        archives = json.load(f)
    pids = sorted(int(k) for k in archives.keys())[:2]
    smoke_archives = {str(pid): archives[str(pid)] for pid in pids}
    smoke_path = "/tmp/strategy_archives_smoke.json"
    with open(smoke_path, "w") as f:
        json.dump(smoke_archives, f)
    return run_archive_pilot(MODELS, SMOKE_CFG, smoke_path, "smoke")


@app.local_entrypoint()
def main(smoke: bool = False):
    # Read local archive and send to Modal
    archive_path = "data/analysis/strategy_archives.json"
    with open(archive_path) as f:
        archive_json = f.read()

    if smoke:
        print("Running SMOKE test (2 models × 2 problems × 1 iter × K=4)...")
        result = run_smoke.remote(archive_json)
    else:
        print("Running FULL pilot (2 models × 10 problems × 3 iters × K=16)...")
        result = run_full.remote(archive_json)
    print("\n" + json.dumps(result, indent=2))
