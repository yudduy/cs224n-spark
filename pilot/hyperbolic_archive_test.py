#!/usr/bin/env python3
"""
Hyperbolic API Archive-Guided Test (Run 20)
============================================
Tests archive-guided prompting with DeepSeek-R1-0528 (671B) via Hyperbolic API
to disentangle 1.5B ICL capacity confound from explorability signal.

Key question: Does R1-671B maintain accuracy under archive prompting?
If yes → 1.5B collapse is scale limitation, not explorability signal.
If no  → archive prompting genuinely disrupts reasoning (even at scale).

Design: 3 problems × K=8 rollouts × {baseline, archive} = 48 API calls
Estimated cost: ~$0.15-0.30

Usage:
  python3 pilot/hyperbolic_archive_test.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────

HYPERBOLIC_API_KEY = os.environ.get("HYPERBOLIC_API_KEY", "")
BASE_URL = "https://api.hyperbolic.xyz/v1"
MODEL = "deepseek-ai/DeepSeek-R1-0528"

# 3 key problems: P42 (hard, R1-1.5B=0%), P46 (hard, R1-1.5B=0%), P2 (easy, sanity)
TEST_PIDS = [42, 46, 2]
K = 4  # rollouts per condition (reduced from 8 — R1-671B is slow)
MAX_TOKENS = 16384  # R1-671B needs room for long CoT
TEMPERATURE = 0.6
TOP_P = 0.95

ARCHIVE_PATH = Path("data/analysis/strategy_archives.json")
OUT_PATH = Path("data/analysis/hyperbolic_archive_test.json")

# ── Helpers ───────────────────────────────────────────────────


def extract_boxed(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


def normalize_answer(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    s = ans.strip().replace(" ", "").replace(",", "").lower()
    # Normalize LaTeX fraction variants
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


def build_archive_prompt(problem_text: str, archive: dict) -> str:
    """Build archive-guided prompt (same as modal_archive_pilot.py)."""
    strategies = archive.get("strategies", [])
    strat_lines = []
    for s in strategies:
        strat_lines.append(f"### Strategy {s['id']}: {s['name']}")
        strat_lines.append(s["description"])
        strat_lines.append("")

    strat_block = "\n".join(strat_lines).strip()

    parts = [
        "Previous solvers have found these approaches for the problem below:",
        "",
        strat_block,
        "",
        "Your task: solve this problem using a FUNDAMENTALLY DIFFERENT mathematical approach than any listed above.",
        "",
        problem_text,
        "Please reason step by step, and put your final answer within \\boxed{}.",
    ]
    return "\n".join(parts)


def build_baseline_prompt(problem_text: str) -> str:
    """Standard prompt without archive."""
    return f"{problem_text}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


# ── Main ──────────────────────────────────────────────────────


def run_test():
    client = OpenAI(api_key=HYPERBOLIC_API_KEY, base_url=BASE_URL)

    with open(ARCHIVE_PATH) as f:
        archives = json.load(f)

    print(f"\n{'='*60}")
    print(f"HYPERBOLIC ARCHIVE TEST (Run 20)")
    print(f"  Model: {MODEL}")
    print(f"  Problems: {TEST_PIDS}")
    print(f"  K={K} per condition (baseline + archive)")
    print(f"  Total API calls: {len(TEST_PIDS) * K * 2}")
    print(f"{'='*60}")

    results = []
    total_input = 0
    total_output = 0
    start_time = time.time()

    for pid in TEST_PIDS:
        arch = archives[str(pid)]
        problem_text = arch["problem_text"]
        answer = arch["answer"]
        n_strats = arch["n_strategies"]

        print(f"\nP{pid} [{arch['tier']}] — {arch['subject']}")
        print(f"  Baseline strategies: {n_strats}")
        print(f"  Answer: {answer}")

        problem_result = {
            "problem_id": pid,
            "tier": arch["tier"],
            "subject": arch["subject"],
            "answer": answer,
            "n_baseline_strategies": n_strats,
            "conditions": {},
        }

        for condition in ["baseline", "archive"]:
            if condition == "baseline":
                prompt = build_baseline_prompt(problem_text)
            else:
                prompt = build_archive_prompt(problem_text, arch)

            rollouts = []
            n_correct = 0

            for i in range(K):
                try:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                    )

                    text = resp.choices[0].message.content
                    usage = resp.usage
                    total_input += usage.prompt_tokens
                    total_output += usage.completion_tokens

                    pred = extract_boxed(text)
                    correct = check_answer(pred, answer)
                    if correct:
                        n_correct += 1

                    rollouts.append({
                        "idx": i,
                        "response": text,
                        "final_answer": pred,
                        "is_correct": correct,
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                    })

                    status = "✓" if correct else "✗"
                    print(f"  {condition}[{i}]: {status} (pred={pred}, {usage.completion_tokens} tok)")

                except Exception as e:
                    print(f"  {condition}[{i}]: ERROR — {e}")
                    rollouts.append({
                        "idx": i,
                        "error": str(e),
                        "is_correct": False,
                    })

                # Rate limiting — be conservative
                time.sleep(0.5)

            acc = n_correct / K if K > 0 else 0
            problem_result["conditions"][condition] = {
                "rollouts": rollouts,
                "n_correct": n_correct,
                "accuracy": acc,
            }
            print(f"  {condition}: {n_correct}/{K} = {acc:.0%}")

        results.append(problem_result)

    # ── Summary ───────────────────────────────────────────────

    elapsed = time.time() - start_time
    cost_input = total_input * 0.40 / 1e6   # R1-0528 input ~$0.40/M
    cost_output = total_output * 1.75 / 1e6  # R1-0528 output ~$1.75/M
    total_cost = cost_input + cost_output

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Problem':>8} {'Tier':>7} {'Baseline':>10} {'Archive':>10} {'Delta':>8}")
    print("-" * 50)

    baseline_total = 0
    archive_total = 0
    for r in results:
        pid = r["problem_id"]
        tier = r["tier"]
        b_acc = r["conditions"]["baseline"]["accuracy"]
        a_acc = r["conditions"]["archive"]["accuracy"]
        delta = a_acc - b_acc
        baseline_total += b_acc
        archive_total += a_acc
        print(f"  P{pid:<5d} {tier:>7} {b_acc:>9.0%} {a_acc:>9.0%} {delta:>+7.0%}")

    n = len(results)
    print("-" * 50)
    print(f"  {'AVG':>7} {'':>7} {baseline_total/n:>9.0%} {archive_total/n:>9.0%} {(archive_total-baseline_total)/n:>+7.0%}")

    print(f"\nAPI usage: {total_input:,} input + {total_output:,} output tokens")
    print(f"Estimated cost: ${total_cost:.4f}")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ── Interpretation ────────────────────────────────────────

    avg_baseline = baseline_total / n
    avg_archive = archive_total / n
    delta = avg_archive - avg_baseline

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    if delta > -0.1:
        print("  R1-671B maintains accuracy under archive prompting.")
        print("  → 1.5B collapse is a SCALE LIMITATION (ICL capacity)")
        print("  → NOT an explorability signal")
        print("  → Archive-guided search is viable at sufficient scale")
    else:
        print("  R1-671B also degrades under archive prompting.")
        print("  → Archive prompting genuinely disrupts reasoning")
        print("  → This is NOT just a scale issue")
        print("  → Need to rethink prompt design")

    # ── Compare with 1.5B results ─────────────────────────────

    print(f"\n  1.5B comparison (from Run 19):")
    run19_data = {
        42: {"r1_base": "0/64",  "r1_arch": "0/48",  "v2_base": "9/64",  "v2_arch": "37/48"},
        46: {"r1_base": "64/64", "r1_arch": "29/48", "v2_base": "64/64", "v2_arch": "48/48"},
        2:  {"r1_base": "59/64", "r1_arch": "6/48",  "v2_base": "64/64", "v2_arch": "48/48"},
    }
    for pid in TEST_PIDS:
        r19 = run19_data.get(pid, {})
        r20 = next((r for r in results if r["problem_id"] == pid), None)
        if r19 and r20:
            r671b_base = r20["conditions"]["baseline"]["accuracy"]
            r671b_arch = r20["conditions"]["archive"]["accuracy"]
            print(f"    P{pid}: R1-1.5B base={r19['r1_base']} arch={r19['r1_arch']} | "
                  f"R1-671B base={r671b_base:.0%} arch={r671b_arch:.0%}")

    # ── Save ──────────────────────────────────────────────────

    output = {
        "model": MODEL,
        "test_pids": TEST_PIDS,
        "K": K,
        "temperature": TEMPERATURE,
        "results": results,
        "summary": {
            "avg_baseline_accuracy": round(avg_baseline, 4),
            "avg_archive_accuracy": round(avg_archive, 4),
            "delta": round(delta, 4),
        },
        "api_usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": round(total_cost, 4),
        },
        "elapsed_s": round(elapsed, 1),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    run_test()
