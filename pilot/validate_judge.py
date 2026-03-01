#!/usr/bin/env python3
"""
Judge Reliability Validation (Experiments 2a–2c)
=================================================
Tests LLM judge (GPT-5-nano) stability via prompt rephrasing,
repeated runs, and soft classification.

Estimated cost: ~$0.15

Usage:
  python3 pilot/validate_judge.py
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from openai import OpenAI

# ── Paths & Config ──

DATA_DIR = Path("data/modal_runs/gen_traces_full")
JUDGE_PATH = Path("data/analysis/llm_judge_pilot.json")
OUT_PATH = Path("data/analysis/judge_validation.json")

MODELS = ["r1-distill", "nemotron-v1", "nemotron-v2", "nemotron-brorl"]
MODEL_LABELS = {
    "r1-distill": "A (R1-Distill, base)",
    "nemotron-v1": "B (v1, 2K RL steps)",
    "nemotron-v2": "C (v2, 3K RL steps)",
    "nemotron-brorl": "D (BroRL, breadth RL)",
}
SAMPLE_PER_MODEL = 8
MODEL_NAME = "gpt-5-nano"

# ── Load data ──

env_path = Path("/Users/duy/Documents/build/eigen/seigen/.env.ecloud")
api_key = None
with open(env_path) as ef:
    for line in ef:
        if line.startswith("OPENAI_API_KEY="):
            api_key = line.strip().split("=", 1)[1]
            break

if not api_key:
    print("ERROR: Could not find OPENAI_API_KEY")
    sys.exit(1)

client = OpenAI(api_key=api_key)

with open(JUDGE_PATH) as f:
    judge_data = json.load(f)

with open(DATA_DIR / "problems.json") as f:
    problems = json.load(f)

all_data = {}
for m in MODELS:
    with open(DATA_DIR / m / "traces.json") as f:
        all_data[m] = json.load(f)

# Select 10 multi-strategy problems
multi_strat = [j for j in judge_data if j.get("llm_response", {}).get("n_strategies", 0) > 1]
SELECTED_PIDS = [j["problem_id"] for j in multi_strat[:10]]
print(f"Selected {len(SELECTED_PIDS)} multi-strategy problems: {SELECTED_PIDS}")


# ── Shared: build trace block for a problem ──

def build_traces_block(pid):
    """Build the trace block for a problem (same across all prompts)."""
    prob = problems[pid]
    traces_block = []
    trace_labels = []
    for m in MODELS:
        label_prefix = MODEL_LABELS[m].split(" ")[0]
        p_data = all_data[m]["problems"][pid]
        correct_rollouts = [r for r in p_data["rollouts"] if r["is_correct"]]
        if not correct_rollouts:
            traces_block.append(f"\n[Model {MODEL_LABELS[m]}]: No correct solutions.\n")
            continue
        n = min(SAMPLE_PER_MODEL, len(correct_rollouts))
        step = max(1, len(correct_rollouts) // n)
        sampled = correct_rollouts[::step][:n]
        for i, r in enumerate(sampled):
            tid = f"{label_prefix}{i+1}"
            trace_labels.append(tid)
            text = r["response"][:3000]
            if len(r["response"]) > 3000:
                text += "\n[...truncated...]"
            traces_block.append(f"\n--- Trace {tid} (Model {MODEL_LABELS[m]}) ---\n{text}\n")
    return prob, "\n".join(traces_block), trace_labels


# ══════════════════════════════════════════════════════════
# EXPERIMENT 2a: Prompt Sensitivity
# ══════════════════════════════════════════════════════════

PROMPT_VARIANTS = {
    "original": """You are analyzing mathematical reasoning traces from 4 AI model checkpoints on the SAME training trajectory.
Your task: identify how many DISTINCT high-level solution strategies are used across all traces.

IMPORTANT:
- A "strategy" is a fundamentally different MATHEMATICAL APPROACH (e.g., algebraic vs geometric, direct computation vs proof by contradiction, substitution vs factoring)
- Differences in verbosity, formatting, step ordering, or phrasing are NOT different strategies
- Two traces that use the same approach but one is more detailed or rambles more = SAME strategy
- Focus on the CORE mathematical method, not surface presentation""",

    "concise": """Classify these mathematical solution traces by their HIGH-LEVEL approach.
Group traces that use the same core mathematical method, regardless of verbosity or presentation differences.
Only separate traces into different strategies if they use fundamentally different mathematical techniques.""",

    "detailed": """You are a mathematics education researcher analyzing how different AI models solve the same problem.
Your goal is to identify genuinely distinct solution STRATEGIES — not surface variations.

Key distinction: A strategy is about WHAT mathematical tools/techniques are used (e.g., coordinate geometry vs. synthetic geometry, direct proof vs. contradiction, algebraic manipulation vs. substitution).
NOT about: how many words are used, how steps are ordered, or how results are formatted.

Two traces using the same technique but different levels of detail = SAME strategy.
Two traces arriving at the same answer via different mathematical routes = DIFFERENT strategies.""",

    "adversarial": """Identify the MINIMUM number of genuinely distinct solution approaches in these traces.
Be CONSERVATIVE — only count strategies as different if they use fundamentally incompatible mathematical methods.
Traces that are verbose vs concise versions of the same approach = SAME strategy.
Traces that use slightly different algebraic rearrangements = SAME strategy.
Only flag a new strategy if the mathematical FRAMEWORK is different (e.g., geometric vs algebraic, recursive vs closed-form).""",
}

JSON_SUFFIX = """

Respond in this exact JSON format:
{{
  "n_strategies": <int>,
  "strategies": [
    {{"id": 1, "name": "<short name>", "description": "<1-2 sentence description of the mathematical approach>"}},
    ...
  ],
  "classifications": {{
    "<trace_id>": <strategy_id>,
    ...
  }},
  "confidence": "<high|medium|low>",
  "notes": "<any observations about strategy diversity>"
}}"""


def run_prompt_sensitivity():
    """Run judge with 4 prompt variants, measure strategy count CV."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2a: PROMPT SENSITIVITY")
    print("=" * 70)

    results = {}
    total_tokens = 0

    for pid in SELECTED_PIDS:
        prob, traces_str, trace_labels = build_traces_block(pid)
        prob_text = prob["problem"]
        answer = prob.get("answer", "unknown")

        counts = {}
        for variant_name, preamble in PROMPT_VARIANTS.items():
            prompt = f"""{preamble}

Problem: {prob_text}

Correct answer: {answer}

{traces_str}
{JSON_SUFFIX}"""

            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                total_tokens += resp.usage.prompt_tokens + resp.usage.completion_tokens
                parsed = json.loads(resp.choices[0].message.content)
                counts[variant_name] = parsed.get("n_strategies", 0)
            except Exception as e:
                print(f"  ERROR P{pid} {variant_name}: {e}")
                counts[variant_name] = None

            time.sleep(1.0)  # rate limit (200K TPM limit)

        valid = [c for c in counts.values() if c is not None and c > 0]
        if len(valid) >= 2:
            cv = np.std(valid) / np.mean(valid)
        else:
            cv = float("nan")

        results[pid] = {"counts": counts, "cv": round(cv, 4) if not np.isnan(cv) else None}
        print(f"  P{pid}: counts={counts}  CV={cv:.3f}" if not np.isnan(cv) else f"  P{pid}: counts={counts}  CV=N/A")

    # Aggregate
    cvs = [r["cv"] for r in results.values() if r["cv"] is not None]
    mean_cv = np.mean(cvs) if cvs else float("nan")
    print(f"\n  Mean CV across problems: {mean_cv:.3f}")
    print(f"  Total tokens: {total_tokens}")

    verdict = "PASS" if mean_cv < 0.10 else ("PARTIAL" if mean_cv < 0.20 else "FAIL")
    print(f"  Verdict: {verdict} — CV {'< 0.10' if mean_cv < 0.10 else ('< 0.20' if mean_cv < 0.20 else '>= 0.20')}")

    return {"per_problem": {str(k): v for k, v in results.items()}, "mean_cv": round(mean_cv, 4) if not np.isnan(mean_cv) else None,
            "total_tokens": total_tokens, "verdict": verdict}


# ══════════════════════════════════════════════════════════
# EXPERIMENT 2b: Self-Consistency (Fleiss' Kappa)
# ══════════════════════════════════════════════════════════

def fleiss_kappa(ratings_matrix):
    """Compute Fleiss' kappa for inter-rater agreement.
    ratings_matrix: (n_subjects, n_categories) — counts of raters assigning each category."""
    N, k = ratings_matrix.shape
    n = ratings_matrix.sum(axis=1)[0]  # number of raters per subject (should be constant)

    # Proportion of assignments to each category
    p_j = ratings_matrix.sum(axis=0) / (N * n)

    # Per-subject agreement
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = np.mean(P_i)

    # Expected agreement
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def run_self_consistency():
    """Run same prompt 3 times with T=0.3, compute Fleiss' kappa on per-trace classifications."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2b: SELF-CONSISTENCY (Fleiss' Kappa)")
    print("=" * 70)

    N_RUNS = 3
    total_tokens = 0

    all_kappas = []

    for pid in SELECTED_PIDS:
        prob, traces_str, trace_labels = build_traces_block(pid)
        prob_text = prob["problem"]
        answer = prob.get("answer", "unknown")

        prompt = f"""{PROMPT_VARIANTS['original']}

Problem: {prob_text}

Correct answer: {answer}

{traces_str}
{JSON_SUFFIX}"""

        run_classifications = []
        for run_idx in range(N_RUNS):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                total_tokens += resp.usage.prompt_tokens + resp.usage.completion_tokens
                parsed = json.loads(resp.choices[0].message.content)
                run_classifications.append(parsed.get("classifications", {}))
            except Exception as e:
                print(f"  ERROR P{pid} run {run_idx}: {e}")
                run_classifications.append({})
            time.sleep(1.0)  # rate limit

        # Compute agreement: for each trace, build a vector of strategy assignments across runs
        if len(run_classifications) < 2:
            continue

        # Get all strategy IDs seen across runs
        all_strat_ids = set()
        for cls in run_classifications:
            all_strat_ids.update(cls.values())
        strat_list = sorted(all_strat_ids)
        strat_to_idx = {s: i for i, s in enumerate(strat_list)}

        if len(strat_list) < 2:
            print(f"  P{pid}: Only 1 strategy seen across all runs — perfect agreement by default")
            all_kappas.append(1.0)
            continue

        # Build ratings matrix: (n_traces, n_categories)
        # Each trace is rated by N_RUNS raters
        common_labels = set(trace_labels)
        for cls in run_classifications:
            common_labels &= set(cls.keys())

        if not common_labels:
            print(f"  P{pid}: No common trace labels across runs")
            continue

        ratings = np.zeros((len(common_labels), len(strat_list)), dtype=int)
        for row_idx, label in enumerate(sorted(common_labels)):
            for cls in run_classifications:
                sid = cls.get(label)
                if sid is not None and sid in strat_to_idx:
                    ratings[row_idx, strat_to_idx[sid]] += 1

        kappa = fleiss_kappa(ratings)
        all_kappas.append(kappa)
        print(f"  P{pid}: kappa={kappa:.3f}  (n_traces={len(common_labels)}, n_strats={len(strat_list)})")

    mean_kappa = np.mean(all_kappas) if all_kappas else float("nan")
    print(f"\n  Mean Fleiss' kappa: {mean_kappa:.3f}")
    print(f"  Total tokens: {total_tokens}")

    verdict = "PASS" if mean_kappa > 0.60 else ("PARTIAL" if mean_kappa > 0.40 else "FAIL")
    print(f"  Verdict: {verdict} — kappa {'> 0.60' if mean_kappa > 0.60 else ('> 0.40' if mean_kappa > 0.40 else '<= 0.40')}")

    return {"per_problem_kappas": [round(k, 4) for k in all_kappas], "mean_kappa": round(mean_kappa, 4) if not np.isnan(mean_kappa) else None,
            "total_tokens": total_tokens, "verdict": verdict}


# ══════════════════════════════════════════════════════════
# EXPERIMENT 2c: Soft Classification Check
# ══════════════════════════════════════════════════════════

SOFT_PROMPT = """You are analyzing mathematical reasoning traces from 4 AI model checkpoints.
Identify distinct solution strategies, then classify each trace.

IMPORTANT: Some traces may use a MIX of strategies or be AMBIGUOUS. For each trace, provide:
- The primary strategy ID
- A confidence score (0.0 to 1.0) indicating how clearly the trace fits that strategy
- If confidence < 0.7, also list the secondary strategy ID

Differences in verbosity or formatting are NOT different strategies.
Focus on the CORE mathematical method."""

SOFT_JSON_SUFFIX = """

Respond in this exact JSON format:
{{
  "n_strategies": <int>,
  "strategies": [
    {{"id": 1, "name": "<short name>", "description": "<1-2 sentence description>"}},
    ...
  ],
  "classifications": {{
    "<trace_id>": {{"primary": <strategy_id>, "confidence": <0.0-1.0>, "secondary": <strategy_id_or_null>}},
    ...
  }},
  "notes": "<observations about ambiguity>"
}}"""


def run_soft_classification():
    """Run judge with soft classification prompt, check ambiguity rate."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2c: SOFT CLASSIFICATION CHECK")
    print("=" * 70)

    total_tokens = 0
    all_confidences = []
    n_ambiguous = 0
    n_total_traces = 0

    per_problem = {}

    for pid in SELECTED_PIDS:
        prob, traces_str, trace_labels = build_traces_block(pid)
        prob_text = prob["problem"]
        answer = prob.get("answer", "unknown")

        prompt = f"""{SOFT_PROMPT}

Problem: {prob_text}

Correct answer: {answer}

{traces_str}
{SOFT_JSON_SUFFIX}"""

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            total_tokens += resp.usage.prompt_tokens + resp.usage.completion_tokens
            parsed = json.loads(resp.choices[0].message.content)

            cls = parsed.get("classifications", {})
            problem_confs = []
            problem_ambig = 0

            for tid, info in cls.items():
                if isinstance(info, dict):
                    conf = info.get("confidence", 1.0)
                    problem_confs.append(conf)
                    all_confidences.append(conf)
                    n_total_traces += 1
                    if conf < 0.7:
                        n_ambiguous += 1
                        problem_ambig += 1
                else:
                    # Fallback: hard classification returned despite soft prompt
                    all_confidences.append(1.0)
                    n_total_traces += 1

            mean_conf = np.mean(problem_confs) if problem_confs else 1.0
            per_problem[pid] = {
                "n_traces": len(cls),
                "mean_confidence": round(mean_conf, 3),
                "n_ambiguous": problem_ambig,
                "n_strategies": parsed.get("n_strategies", 0),
            }
            print(f"  P{pid}: {len(cls)} traces, mean_conf={mean_conf:.2f}, ambiguous={problem_ambig}/{len(cls)}")

        except Exception as e:
            print(f"  ERROR P{pid}: {e}")

        time.sleep(1.0)  # rate limit

    ambig_rate = n_ambiguous / n_total_traces if n_total_traces > 0 else 0
    mean_conf = np.mean(all_confidences) if all_confidences else 1.0

    print(f"\n  Total traces classified: {n_total_traces}")
    print(f"  Ambiguous (conf < 0.7): {n_ambiguous} ({ambig_rate:.1%})")
    print(f"  Mean confidence: {mean_conf:.3f}")
    print(f"  Total tokens: {total_tokens}")

    if ambig_rate > 0.30:
        verdict = "CONCERN"
        print(f"  Verdict: CONCERN — >30% ambiguous, hard classification is lossy")
    elif ambig_rate > 0.15:
        verdict = "PARTIAL"
        print(f"  Verdict: PARTIAL — 15-30% ambiguous, some noise from hard classification")
    else:
        verdict = "PASS"
        print(f"  Verdict: PASS — <15% ambiguous, hard classification is adequate")

    return {
        "n_total_traces": n_total_traces, "n_ambiguous": n_ambiguous,
        "ambiguity_rate": round(ambig_rate, 4), "mean_confidence": round(mean_conf, 4),
        "per_problem": {str(k): v for k, v in per_problem.items()},
        "total_tokens": total_tokens, "verdict": verdict,
    }


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("JUDGE RELIABILITY VALIDATION (Experiments 2a–2c)")
    print("=" * 70)

    all_results = {}

    all_results["2a_prompt_sensitivity"] = run_prompt_sensitivity()
    all_results["2b_self_consistency"] = run_self_consistency()
    all_results["2c_soft_classification"] = run_soft_classification()

    # Overall
    print("\n" + "=" * 70)
    print("JUDGE VALIDATION SUMMARY")
    print("=" * 70)

    tests = [
        ("2a Prompt Sensitivity", all_results["2a_prompt_sensitivity"]["verdict"]),
        ("2b Self-Consistency", all_results["2b_self_consistency"]["verdict"]),
        ("2c Soft Classification", all_results["2c_soft_classification"]["verdict"]),
    ]

    total_tokens = sum(r.get("total_tokens", 0) for r in all_results.values())
    cost = total_tokens * 0.10 / 1e6 + total_tokens * 0.40 / 1e6  # rough estimate
    print(f"  Total tokens: {total_tokens}")
    print(f"  Estimated cost: ${cost:.4f}")

    for name, verdict in tests:
        marker = {"PASS": "+", "PARTIAL": "~", "FAIL": "-", "CONCERN": "!"}[verdict]
        print(f"  [{marker}] {name}: {verdict}")

    pass_count = sum(1 for _, v in tests if v == "PASS")
    print(f"\n  Score: {pass_count}/3 PASS")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUT_PATH}")
