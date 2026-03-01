"""
Phase 1: Trace Generation + Embedding
======================================
4 ProRL-trajectory checkpoints × 60 MATH problems × 64 rollouts.
Saves raw traces, correctness labels, and sentence embeddings.

Output (Modal volume spark-pilot-results → /gen_traces/{tag}/):
  config.json, problems.json, global_summary.json
  {model_name}/traces.json, embeddings.npy, summary.json

Usage:
  modal run pilot/modal_gen_traces.py               # full run (~2h)
  modal run pilot/modal_gen_traces.py --smoke        # quick test (~10min)
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

app = modal.App("spark-gen-traces")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.4")
    .pip_install(
        "transformers>=4.45.0,<5.0.0",
        "datasets",
        "numpy",
        "sentence-transformers",
        "huggingface_hub",
    )
)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("spark-pilot-results", create_if_missing=True)


# ── Models (same Qwen2 1.5B trajectory) ────────────────────────

MODELS = [
    {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "name": "r1-distill",
        "revision": None,
        "label": "R1-Distill (base)",
    },
    {
        "hf_id": "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
        "name": "nemotron-v1",
        "revision": "v1",
        "label": "Nemotron v1 (ProRL 2K steps)",
    },
    {
        "hf_id": "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
        "name": "nemotron-v2",
        "revision": None,
        "label": "Nemotron v2 (ProRL 3K steps)",
    },
    {
        "hf_id": "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
        "name": "nemotron-brorl",
        "revision": "brorl",
        "label": "BroRL (v2 + 419 steps, N=512)",
    },
]


# ── Configuration ───────────────────────────────────────────────

PASS_K_VALUES = [1, 4, 8, 16, 32, 64]

FULL_CFG = {
    "problems_per_tier": 20,
    "tiers": {"easy": [1, 2], "medium": [3], "hard": [4, 5]},
    "seed": 42,
    "K": 64,
    "max_tokens": 4096,
    "temperature": 1.0,
    "top_p": 1.0,
    "embedder_id": "sentence-transformers/all-MiniLM-L6-v2",
    "embed_chunk_chars": 1000,  # ~250 tokens per MiniLM window
}

SMOKE_CFG = {
    **FULL_CFG,
    "problems_per_tier": 2,  # 6 problems total
    "K": 4,
}


# ── Helpers ─────────────────────────────────────────────────────


def load_problems(cfg: dict) -> list[dict]:
    """Load MATH-500 problems stratified by difficulty tier."""
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rng = np.random.RandomState(cfg["seed"])
    problems = []
    pid = 0
    for tier_name, levels in cfg["tiers"].items():
        pool = [r for r in ds if int(r.get("level", 0)) in levels]
        n = min(cfg["problems_per_tier"], len(pool))
        idx = rng.choice(len(pool), size=n, replace=False)
        for i in sorted(idx.tolist()):
            r = pool[i]
            problems.append({
                "problem_id": pid,
                "problem": r["problem"],
                "answer": r.get("answer", ""),
                "level": int(r.get("level", 0)),
                "subject": r.get("type", r.get("subject", "unknown")),
                "tier": tier_name,
            })
            pid += 1
    return problems


def make_prompt(tokenizer, problem: str) -> str:
    msgs = [{
        "role": "user",
        "content": f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    }]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def extract_boxed(text: str) -> Optional[str]:
    """Extract last \\boxed{...} from text."""
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
    """Unbiased pass@k estimator (Chen et al. 2021, Codex)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_at_ks(n: int, c: int, K: int) -> dict:
    """Compute pass@k for all standard k values up to K."""
    result = {}
    for k in PASS_K_VALUES:
        if k <= K:
            result[str(k)] = round(pass_at_k(n, c, k), 6)
    return result


def count_think_tokens(text: str) -> int:
    """Approximate word count inside <think> block."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return len(m.group(1).split()) if m else 0


# ── Core: Generation ────────────────────────────────────────────


def generate_model_traces(model_cfg: dict, problems: list[dict], cfg: dict) -> dict:
    """Load one model, generate K rollouts per problem, return structured results."""
    import torch
    from vllm import LLM, SamplingParams

    name = model_cfg["name"]
    K = cfg["K"]

    print(f"\n{'='*60}")
    print(f"GENERATING: {name} — {model_cfg['label']}")
    print(f"  {len(problems)} problems × K={K} = {len(problems) * K} rollouts")
    print(f"{'='*60}")

    # Load model
    t0 = time.time()
    llm_kwargs = dict(
        model=model_cfg["hf_id"],
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        max_model_len=4608,
        trust_remote_code=True,
    )
    if model_cfg["revision"]:
        llm_kwargs["revision"] = model_cfg["revision"]

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Generate
    prompts = [make_prompt(tokenizer, p["problem"]) for p in problems]
    params = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["max_tokens"],
        n=K,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, params)
    gen_time = time.time() - t0
    total = len(problems) * K
    print(f"  Generated {total} rollouts in {gen_time:.1f}s ({total / gen_time:.1f}/s)")

    # Process
    all_texts = []  # flat list for embedding; index = problem_id * K + rollout_idx
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
            all_texts.append(text)

        n_correct = sum(r["is_correct"] for r in rollouts)
        unique_ans = set(
            normalize_answer(r["final_answer"])
            for r in rollouts if r["final_answer"]
        )

        problem_results.append({
            "problem_id": problem["problem_id"],
            "n_correct": n_correct,
            "n_unique_answers": len(unique_ans),
            "pass_at_k": compute_pass_at_ks(K, n_correct, K),
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
    print(f"  GPU freed.")

    return {
        "load_time": load_time,
        "gen_time": gen_time,
        "problem_results": problem_results,
        "all_texts": all_texts,
    }


# ── Core: Embedding ─────────────────────────────────────────────


def embed_traces(texts: list[str], cfg: dict):
    """Mean-pooled MiniLM embeddings over chunked traces. Returns (N, 384) array."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"  Embedding {len(texts)} traces (chunked, mean-pooled)...")
    t0 = time.time()
    embedder = SentenceTransformer(cfg["embedder_id"])
    chunk_size = cfg["embed_chunk_chars"]

    # Chunk all texts and track boundaries
    chunks = []
    boundaries = []
    for text in texts:
        start = len(chunks)
        if len(text) == 0:
            chunks.append("")
        else:
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i : i + chunk_size])
        boundaries.append((start, len(chunks)))

    # Batch encode all chunks
    chunk_embs = embedder.encode(
        chunks, batch_size=256, show_progress_bar=False, normalize_embeddings=True
    )

    # Mean-pool per text, re-normalize
    dim = chunk_embs.shape[1]
    embeddings = np.zeros((len(texts), dim), dtype=np.float32)
    for i, (s, e) in enumerate(boundaries):
        mean = np.mean(chunk_embs[s:e], axis=0)
        norm = np.linalg.norm(mean)
        embeddings[i] = mean / norm if norm > 1e-8 else mean

    embed_time = time.time() - t0
    print(f"  Done: {len(chunks)} chunks → {len(texts)} vectors (dim={dim}) in {embed_time:.1f}s")

    del embedder
    gc.collect()
    return embeddings


# ── Summary computation ─────────────────────────────────────────


def compute_summary(
    model_cfg: dict, problem_results: list[dict],
    problems: list[dict], cfg: dict,
    load_time: float, gen_time: float,
) -> dict:
    K = cfg["K"]
    n = len(problem_results)

    # Aggregate pass@k
    agg_pass = {}
    for k in PASS_K_VALUES:
        sk = str(k)
        if k <= K:
            vals = [r["pass_at_k"].get(sk, 0) for r in problem_results]
            agg_pass[sk] = round(sum(vals) / len(vals), 4)

    summary = {
        "model": model_cfg["name"],
        "label": model_cfg["label"],
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


# ── Run logic ───────────────────────────────────────────────────


def _run(models: list[dict], cfg: dict, tag: str):
    """Core run: generate traces for all models, embed, save."""
    import numpy as np

    os.environ["HF_HOME"] = "/hf-cache"

    out_dir = f"/results/gen_traces/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    run_start = time.time()
    ts = int(run_start)

    K = cfg["K"]
    n_problems = sum(cfg["problems_per_tier"] for _ in cfg["tiers"])

    print(f"\n{'='*60}")
    print(f"PHASE 1: TRACE GENERATION + EMBEDDING [{tag}]")
    print(f"  {len(models)} models × {n_problems} problems × K={K} = {len(models) * n_problems * K} rollouts")
    print(f"{'='*60}")

    # Load problems
    problems = load_problems(cfg)
    tier_counts = {}
    for p in problems:
        tier_counts[p["tier"]] = tier_counts.get(p["tier"], 0) + 1
    print(f"\n  Problems loaded: {len(problems)} total — {tier_counts}")
    for tier in ["easy", "medium", "hard"]:
        tier_probs = [p for p in problems if p["tier"] == tier]
        subjects = sorted(set(p["subject"] for p in tier_probs))
        print(f"    {tier}: subjects={subjects}")

    # Save problems + config
    with open(os.path.join(out_dir, "problems.json"), "w") as f:
        json.dump(problems, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"cfg": cfg, "models": models, "timestamp": ts, "tag": tag}, f, indent=2)
    results_vol.commit()

    # Generate per model
    model_summaries = []
    for mi, model_cfg in enumerate(models):
        name = model_cfg["name"]
        model_dir = os.path.join(out_dir, name)
        os.makedirs(model_dir, exist_ok=True)

        # Generate
        try:
            gen = generate_model_traces(model_cfg, problems, cfg)
        except Exception as e:
            print(f"\n  ERROR on {name}: {e}")
            model_summaries.append({"model": name, "error": str(e)})
            continue

        # Embed
        embeddings = embed_traces(gen["all_texts"], cfg)
        np.save(os.path.join(model_dir, "embeddings.npy"), embeddings)
        print(f"  Saved embeddings: shape={embeddings.shape}")

        # Save traces (compact JSON — no indent, these are large)
        traces_path = os.path.join(model_dir, "traces.json")
        with open(traces_path, "w") as f:
            json.dump({
                "model": model_cfg,
                "timestamp": ts,
                "K": K,
                "problems": gen["problem_results"],
            }, f)
        size_mb = os.path.getsize(traces_path) / (1024 * 1024)
        print(f"  Saved traces: {size_mb:.1f}MB")

        # Summary
        summary = compute_summary(
            model_cfg, gen["problem_results"], problems, cfg,
            gen["load_time"], gen["gen_time"],
        )
        with open(os.path.join(model_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        model_summaries.append(summary)

        # Print
        pk = summary["pass_at_k"]
        print(f"\n  {name}: pass@1={pk.get('1','?')} pass@{K}={pk.get(str(K),'?')} unique_ans={summary['mean_unique_answers']}")
        for tier, td in summary.get("per_tier", {}).items():
            tpk = td["pass_at_k"]
            print(f"    {tier}: pass@1={tpk.get('1','?')} mean_correct={td['mean_correct']}/{K}")

        # ETA
        elapsed = time.time() - run_start
        done = mi + 1
        if done < len(models):
            eta = (elapsed / done) * (len(models) - done)
            print(f"  [{done}/{len(models)}] {elapsed/60:.1f}min elapsed, ~{eta/60:.1f}min remaining")

        # Commit after each model (crash safety)
        results_vol.commit()

    # Global summary
    total_time = time.time() - run_start
    global_summary = {
        "tag": tag,
        "timestamp": ts,
        "total_time_s": round(total_time, 1),
        "n_models": len(models),
        "n_problems": len(problems),
        "K": K,
        "total_rollouts": len(models) * len(problems) * K,
        "models": model_summaries,
    }
    with open(os.path.join(out_dir, "global_summary.json"), "w") as f:
        json.dump(global_summary, f, indent=2)
    results_vol.commit()

    # Cross-model comparison table
    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*70}")
    header_ks = [k for k in PASS_K_VALUES if k <= K]
    hdr = f"{'Model':<18}" + "".join(f" {'p@'+str(k):>7}" for k in header_ks) + f" {'uniq':>6}"
    print(hdr)
    print("-" * len(hdr))
    for s in model_summaries:
        if "error" in s:
            print(f"{s['model']:<18} ERROR: {s['error']}")
        else:
            pk = s["pass_at_k"]
            row = f"{s['model']:<18}"
            for k in header_ks:
                row += f" {pk.get(str(k), 0):>7.3f}"
            row += f" {s['mean_unique_answers']:>6.1f}"
            print(row)

    print(f"\nTotal: {total_time/60:.1f}min | Output: {out_dir}")
    return global_summary


# ── Modal entrypoints ───────────────────────────────────────────


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 5,
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
)
def run_full():
    return _run(MODELS, FULL_CFG, "full")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 1,
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
)
def run_smoke():
    smoke_models = [MODELS[0], MODELS[2]]  # R1-Distill + Nemotron v2
    return _run(smoke_models, SMOKE_CFG, "smoke")


@app.local_entrypoint()
def main(smoke: bool = False):
    if smoke:
        print("Running SMOKE test (2 models × 6 problems × K=4)...")
        result = run_smoke.remote()
    else:
        print("Running FULL generation (4 models × 60 problems × K=64)...")
        result = run_full.remote()
    print("\n" + json.dumps(result, indent=2))
