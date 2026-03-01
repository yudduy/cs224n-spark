"""
Temperature Sweep: Trace Generation (Experiment 3)
====================================================
R1-Distill × 10 multi-strategy problems × 3 temperatures × 64 rollouts.
Tests: does our diversity metric increase monotonically with temperature?

Output (Modal volume spark-pilot-results → /temp_sweep/{tag}/):
  config.json, problems.json
  T_{temp}/traces.json

Usage:
  modal run pilot/modal_temp_sweep.py               # full (~10 min)
  modal run pilot/modal_temp_sweep.py --smoke        # quick test
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

app = modal.App("spark-temp-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.4")
    .pip_install(
        "transformers>=4.45.0,<5.0.0",
        "numpy",
        "huggingface_hub",
    )
)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("spark-pilot-results", create_if_missing=True)

# ── Config ──

MODEL_CFG = {
    "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "name": "r1-distill",
    "revision": None,
    "label": "R1-Distill (base)",
}

# Multi-strategy problem IDs from Run 16 judge
MULTI_STRAT_PIDS = [0, 2, 13, 17, 19, 27, 34, 35, 38, 40]

FULL_CFG = {
    "temperatures": [0.3, 0.6, 1.0],
    "K": 64,
    "max_tokens": 4096,
    "top_p": 1.0,
    "problem_ids": MULTI_STRAT_PIDS,
}

SMOKE_CFG = {
    "temperatures": [0.3, 1.0],
    "K": 4,
    "max_tokens": 4096,
    "top_p": 1.0,
    "problem_ids": MULTI_STRAT_PIDS[:3],
}


# ── Helpers (from modal_gen_traces.py) ──

def make_prompt(tokenizer, problem: str) -> str:
    msgs = [{
        "role": "user",
        "content": f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    }]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


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


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def count_think_tokens(text: str) -> int:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return len(m.group(1).split()) if m else 0


# ── Core run ──

def _run(cfg: dict, tag: str):
    """Generate traces at multiple temperatures for R1-Distill."""
    import numpy as np
    from vllm import LLM, SamplingParams

    os.environ["HF_HOME"] = "/hf-cache"
    out_dir = f"/results/temp_sweep/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    run_start = time.time()
    ts = int(run_start)

    K = cfg["K"]
    temps = cfg["temperatures"]
    pids = cfg["problem_ids"]

    print(f"\n{'='*60}")
    print(f"TEMPERATURE SWEEP [{tag}]")
    print(f"  Model: {MODEL_CFG['label']}")
    print(f"  {len(pids)} problems × {len(temps)} temps × K={K} = {len(pids) * len(temps) * K} rollouts")
    print(f"{'='*60}")

    # Load problems from existing data
    problems_path = "/results/gen_traces/full/problems.json"
    with open(problems_path) as f:
        all_problems = json.load(f)
    selected = [all_problems[pid] for pid in pids]

    # Save config + problems
    with open(os.path.join(out_dir, "problems.json"), "w") as f:
        json.dump(selected, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"cfg": cfg, "model": MODEL_CFG, "timestamp": ts, "tag": tag}, f, indent=2)

    # Load model once
    print(f"\nLoading model: {MODEL_CFG['hf_id']}")
    t0 = time.time()
    llm = LLM(
        model=MODEL_CFG["hf_id"],
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        max_model_len=4608,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Build prompts
    prompts = [make_prompt(tokenizer, p["problem"]) for p in selected]

    # Generate at each temperature
    temp_results = {}
    for temp in temps:
        print(f"\n{'─'*60}")
        print(f"Temperature = {temp}")
        print(f"{'─'*60}")

        params = SamplingParams(
            temperature=temp,
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
            n=K,
        )

        t0 = time.time()
        outputs = llm.generate(prompts, params)
        gen_time = time.time() - t0
        total = len(selected) * K
        print(f"  Generated {total} rollouts in {gen_time:.1f}s ({total/gen_time:.1f}/s)")

        # Process results
        problem_results = []
        for prob_idx, (out_obj, problem) in enumerate(zip(outputs, selected)):
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

            problem_results.append({
                "problem_id": problem["problem_id"],
                "n_correct": n_correct,
                "n_unique_answers": len(unique_ans),
                "pass_at_k": {
                    str(k): round(pass_at_k(K, n_correct, k), 6)
                    for k in [1, 4, 8, 16, 32, 64] if k <= K
                },
                "rollouts": rollouts,
            })

            print(f"  P{problem['problem_id']}: {n_correct}/{K} correct, {len(unique_ans)} unique answers")

        # Save
        temp_key = f"T_{temp}"
        temp_dir = os.path.join(out_dir, temp_key)
        os.makedirs(temp_dir, exist_ok=True)

        with open(os.path.join(temp_dir, "traces.json"), "w") as f:
            json.dump({
                "model": MODEL_CFG,
                "temperature": temp,
                "timestamp": ts,
                "K": K,
                "problems": problem_results,
            }, f)

        # Summary stats
        mean_correct = np.mean([r["n_correct"] for r in problem_results])
        mean_unique = np.mean([r["n_unique_answers"] for r in problem_results])
        mean_p1 = np.mean([r["pass_at_k"]["1"] for r in problem_results])

        temp_results[temp_key] = {
            "temperature": temp,
            "gen_time": round(gen_time, 1),
            "mean_correct": round(float(mean_correct), 2),
            "mean_unique_answers": round(float(mean_unique), 2),
            "mean_pass_at_1": round(float(mean_p1), 4),
        }
        print(f"  Summary: mean_correct={mean_correct:.1f}/{K}, p@1={mean_p1:.3f}")

        results_vol.commit()

    # Cleanup
    import torch
    import torch.distributed as dist
    del llm, tokenizer
    if dist.is_initialized():
        dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    total_time = time.time() - run_start
    global_summary = {
        "tag": tag,
        "timestamp": ts,
        "total_time_s": round(total_time, 1),
        "load_time_s": round(load_time, 1),
        "model": MODEL_CFG["name"],
        "n_problems": len(selected),
        "K": K,
        "temperatures": temps,
        "total_rollouts": len(selected) * len(temps) * K,
        "per_temperature": temp_results,
    }

    with open(os.path.join(out_dir, "global_summary.json"), "w") as f:
        json.dump(global_summary, f, indent=2)
    results_vol.commit()

    print(f"\n{'='*60}")
    print("TEMPERATURE SWEEP COMPLETE")
    print(f"  Total time: {total_time/60:.1f}min")
    print(f"  Total rollouts: {len(selected) * len(temps) * K}")
    for tk, tv in temp_results.items():
        print(f"  {tk}: p@1={tv['mean_pass_at_1']:.3f}  correct={tv['mean_correct']:.1f}/{K}")
    print(f"{'='*60}")

    return global_summary


# ── Modal entrypoints ──

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 2,
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
)
def run_full():
    return _run(FULL_CFG, "full")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
)
def run_smoke():
    return _run(SMOKE_CFG, "smoke")


@app.local_entrypoint()
def main(smoke: bool = False):
    if smoke:
        print("Running SMOKE test...")
        result = run_smoke.remote()
    else:
        print("Running FULL temperature sweep...")
        result = run_full.remote()
    print("\n" + json.dumps(result, indent=2))
