#!/usr/bin/env python3
"""
Prepare SFT Training Data from Existing Traces (Run 21)
========================================================
Extracts correct traces from R1-Distill and Nemotron-v2,
formats as chat completions for LoRA SFT.

Equalizes total token count between corpora by subsampling
the larger one (preserving problem diversity).

Output: data/sft_data/{r1-distill,nemotron-v2}.jsonl
Each line: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
  python3 pilot/prepare_sft_data.py
"""

import json
import random
from pathlib import Path

DATA_DIR = Path("data/modal_runs/gen_traces_full")
OUT_DIR = Path("data/sft_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["r1-distill", "nemotron-v2"]
SEED = 42


def load_traces(model_name: str) -> list[dict]:
    """Load correct traces from a model's trace file."""
    with open(DATA_DIR / model_name / "traces.json") as f:
        data = json.load(f)
    return data["problems"]


def load_problems() -> list[dict]:
    with open(DATA_DIR / "problems.json") as f:
        return json.load(f)


def extract_correct_traces(problems_data: list[dict], problems_meta: list[dict]) -> list[dict]:
    """Extract correct traces as chat-format examples."""
    examples = []
    for p_data, p_meta in zip(problems_data, problems_meta):
        pid = p_meta["problem_id"]
        problem_text = p_meta["problem"]

        for r in p_data["rollouts"]:
            if not r["is_correct"]:
                continue
            examples.append({
                "problem_id": pid,
                "n_tokens": r["n_tokens"],
                "messages": [
                    {
                        "role": "user",
                        "content": f"{problem_text}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
                    },
                    {
                        "role": "assistant",
                        "content": r["response"],
                    },
                ],
            })
    return examples


def equalize_by_tokens(examples_a: list[dict], examples_b: list[dict], rng: random.Random) -> tuple:
    """Subsample the larger corpus to match the smaller one's total token count."""
    tokens_a = sum(e["n_tokens"] for e in examples_a)
    tokens_b = sum(e["n_tokens"] for e in examples_b)

    if tokens_a <= tokens_b:
        smaller, larger = examples_a, examples_b
        target_tokens = tokens_a
        name_smaller, name_larger = "r1-distill", "nemotron-v2"
    else:
        smaller, larger = examples_b, examples_a
        target_tokens = tokens_b
        name_smaller, name_larger = "nemotron-v2", "r1-distill"

    print(f"  {name_smaller}: {len(smaller)} traces, {target_tokens:,} tokens (keeping all)")
    print(f"  {name_larger}: {len(larger)} traces, {sum(e['n_tokens'] for e in larger):,} tokens (subsampling to ~{target_tokens:,})")

    # Group by problem to preserve diversity
    by_problem = {}
    for e in larger:
        pid = e["problem_id"]
        by_problem.setdefault(pid, []).append(e)

    # Subsample: round-robin across problems until we hit target
    subsampled = []
    current_tokens = 0
    problem_ids = sorted(by_problem.keys())

    # Shuffle within each problem's traces
    for pid in problem_ids:
        rng.shuffle(by_problem[pid])

    # Round-robin
    idx_per_problem = {pid: 0 for pid in problem_ids}
    while current_tokens < target_tokens:
        added_any = False
        for pid in problem_ids:
            traces = by_problem[pid]
            idx = idx_per_problem[pid]
            if idx >= len(traces):
                continue
            e = traces[idx]
            if current_tokens + e["n_tokens"] > target_tokens * 1.05:
                # Allow 5% overshoot
                idx_per_problem[pid] = idx + 1
                continue
            subsampled.append(e)
            current_tokens += e["n_tokens"]
            idx_per_problem[pid] = idx + 1
            added_any = True
            if current_tokens >= target_tokens:
                break
        if not added_any:
            break

    print(f"  {name_larger} subsampled: {len(subsampled)} traces, {current_tokens:,} tokens")

    if tokens_a <= tokens_b:
        return smaller, subsampled
    else:
        return subsampled, smaller


def save_jsonl(examples: list[dict], path: Path):
    """Save as JSONL with only the messages field."""
    with open(path, "w") as f:
        for e in examples:
            json.dump({"messages": e["messages"]}, f)
            f.write("\n")


def main():
    rng = random.Random(SEED)
    problems_meta = load_problems()

    print(f"Loading traces from {DATA_DIR}...")

    all_examples = {}
    for model in MODELS:
        traces = load_traces(model)
        examples = extract_correct_traces(traces, problems_meta)
        total_tokens = sum(e["n_tokens"] for e in examples)
        n_problems = len(set(e["problem_id"] for e in examples))
        print(f"  {model}: {len(examples)} correct traces across {n_problems} problems, {total_tokens:,} tokens")
        all_examples[model] = examples

    print(f"\nEqualizing token counts...")
    r1_eq, v2_eq = equalize_by_tokens(all_examples["r1-distill"], all_examples["nemotron-v2"], rng)

    # Shuffle before saving
    rng.shuffle(r1_eq)
    rng.shuffle(v2_eq)

    # Verify problem coverage
    r1_pids = set(e["problem_id"] for e in r1_eq)
    v2_pids = set(e["problem_id"] for e in v2_eq)
    print(f"\nProblem coverage — R1: {len(r1_pids)}/60, v2: {len(v2_pids)}/60")
    print(f"Shared problems: {len(r1_pids & v2_pids)}")

    # Save
    save_jsonl(r1_eq, OUT_DIR / "r1-distill.jsonl")
    save_jsonl(v2_eq, OUT_DIR / "nemotron-v2.jsonl")

    # Summary
    r1_tokens = sum(e["n_tokens"] for e in r1_eq)
    v2_tokens = sum(e["n_tokens"] for e in v2_eq)
    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'r1-distill.jsonl'}: {len(r1_eq)} examples, {r1_tokens:,} tokens")
    print(f"  {OUT_DIR / 'nemotron-v2.jsonl'}: {len(v2_eq)} examples, {v2_tokens:,} tokens")
    print(f"  Token ratio: {max(r1_tokens, v2_tokens) / min(r1_tokens, v2_tokens):.3f}x")


if __name__ == "__main__":
    main()
