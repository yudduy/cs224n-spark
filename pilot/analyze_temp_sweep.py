#!/usr/bin/env python3
"""
Temperature Sweep Analysis (Experiment 3)
==========================================
1. Run LLM judge on temperature-swept traces (~$0.05)
2. Compute Rao's Q (all 4 metrics) at each temperature
3. Test monotonicity: Q(T=0.3) < Q(T=0.6) < Q(T=1.0)
4. Plot diversity-accuracy Pareto frontier

Usage:
  # First, download traces from Modal:
  modal volume get spark-pilot-results temp_sweep/full data/modal_runs/temp_sweep_full

  # Then run analysis:
  python3 pilot/analyze_temp_sweep.py
"""

import json
import os
import re
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from openai import OpenAI
from scipy import stats as sp_stats

# ── Paths ──

SWEEP_DIR = Path("data/modal_runs/temp_sweep_full/full")
JUDGE_PATH = Path("data/analysis/llm_judge_pilot.json")
PROBLEMS_PATH = Path("data/modal_runs/gen_traces_full/problems.json")
OUT_PATH = Path("data/analysis/temp_sweep_results.json")

MODELS_LABELS = {
    "r1-distill": "A (R1-Distill, base)",
}
SAMPLE_PER_MODEL = 8
LLM_MODEL = "gpt-5-nano"

# ── Load OpenAI ──

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

# ── Load existing data ──

with open(PROBLEMS_PATH) as f:
    all_problems = json.load(f)

# ── Reused metric functions ──

MATH_TAGS = {
    "direct": ["direct", "straightforward", "compute", "evaluate", "step-by-step"],
    "substitution": ["substitut", "plug", "let x =", "replace"],
    "factoring": ["factor", "factoriz"],
    "algebraic_manipulation": ["algebraic", "manipulat", "simplif", "rearrang"],
    "trigonometric": ["trig", "sin", "cos", "tan", "cot", "pythag"],
    "geometric": ["geometr", "visual", "diagram", "graph", "coordinate"],
    "combinatorial": ["combinat", "count", "permut", "choose"],
    "number_theory": ["modular", "divis", "gcd", "prime", "congruenc"],
    "calculus": ["deriv", "integral", "limit", "differenti"],
    "recursive": ["recurs", "induct", "iterati"],
    "case_analysis": ["case", "case analysis", "cases"],
    "formula": ["formula", "closed-form", "well-known", "theorem", "identity"],
    "scaling": ["scale", "proportion", "ratio", "percent"],
    "pattern": ["pattern", "sequence", "recurrence"],
    "optimization": ["optim", "minimize", "maximize", "lagrange", "am-gm", "inequality"],
    "polynomial": ["polynomial", "roots", "coefficients", "vieta"],
    "modular_arithmetic": ["mod ", "modulo", "remainder", "congruent"],
    "parity": ["parity", "even", "odd", "divisib"],
    "conversion": ["convert", "transform", "rewrite", "equivalent form"],
}

REPRESENTATION_TYPES = ["algebraic", "geometric", "numerical", "trigonometric", "combinatorial", "analytic"]
TECHNIQUE_TYPES = ["direct_computation", "substitution", "factoring", "formula_application",
                   "case_analysis", "algebraic_manipulation", "scaling", "pattern_recognition"]


def extract_tags(strategy_desc):
    text = strategy_desc.lower()
    tags = set()
    for tag, keywords in MATH_TAGS.items():
        for kw in keywords:
            if re.search(kw, text):
                tags.add(tag)
                break
    return tags


def jaccard_distance(tags_i, tags_j):
    if not tags_i and not tags_j:
        return 0.0
    union = tags_i | tags_j
    if not union:
        return 0.0
    return 1.0 - len(tags_i & tags_j) / len(union)


def extract_strategy_traits(strategy):
    text = (strategy["name"] + " " + strategy["description"]).lower()
    rep = "algebraic"
    for r in REPRESENTATION_TYPES:
        if r[:4] in text:
            rep = r
            break
    tech = "direct_computation"
    for t in TECHNIQUE_TYPES:
        short = t.split("_")[0][:4]
        if short in text:
            tech = t
            break
    desc_len = len(strategy["description"].split())
    uses_formula = 1 if any(w in text for w in ["formula", "theorem", "identity", "well-known"]) else 0
    uses_transform = 1 if any(w in text for w in ["convert", "transform", "rewrite", "substitute"]) else 0
    return {
        "representation": rep, "technique": tech, "desc_length": desc_len,
        "uses_formula": uses_formula, "uses_transform": uses_transform,
    }


def gower_distance(traits_i, traits_j):
    dists = []
    for cat_trait in ["representation", "technique"]:
        dists.append(0.0 if traits_i[cat_trait] == traits_j[cat_trait] else 1.0)
    max_range = 50.0
    dists.append(min(abs(traits_i["desc_length"] - traits_j["desc_length"]) / max_range, 1.0))
    for bin_trait in ["uses_formula", "uses_transform"]:
        dists.append(0.0 if traits_i[bin_trait] == traits_j[bin_trait] else 1.0)
    return np.mean(dists)


def gini_simpson(freq_dict):
    if not freq_dict:
        return 0.0
    return 1.0 - sum(p**2 for p in freq_dict.values())


# ══════════════════════════════════════════════════════════
# STEP 1: Run LLM judge on temperature-swept traces
# ══════════════════════════════════════════════════════════

JUDGE_PREAMBLE = """You are analyzing mathematical reasoning traces from an AI model at different temperature settings.
Your task: identify how many DISTINCT high-level solution strategies are used across all traces.

IMPORTANT:
- A "strategy" is a fundamentally different MATHEMATICAL APPROACH (e.g., algebraic vs geometric, direct computation vs proof by contradiction, substitution vs factoring)
- Differences in verbosity, formatting, step ordering, or phrasing are NOT different strategies
- Two traces that use the same approach but one is more detailed or rambles more = SAME strategy
- Focus on the CORE mathematical method, not surface presentation"""

JUDGE_JSON = """

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


def run_judge_on_temp_traces():
    """Run LLM judge on each temperature's traces."""
    print("\n" + "=" * 70)
    print("STEP 1: LLM JUDGE ON TEMPERATURE-SWEPT TRACES")
    print("=" * 70)

    # Detect available temperatures
    temp_dirs = sorted([d for d in SWEEP_DIR.iterdir() if d.is_dir() and d.name.startswith("T_")])
    print(f"  Found temperature dirs: {[d.name for d in temp_dirs]}")

    total_tokens = 0
    judge_results = {}

    for temp_dir in temp_dirs:
        temp_name = temp_dir.name
        temp_val = float(temp_name.split("_")[1])

        traces_path = temp_dir / "traces.json"
        if not traces_path.exists():
            print(f"  {temp_name}: traces.json not found, skipping")
            continue

        with open(traces_path) as f:
            temp_data = json.load(f)

        print(f"\n  {temp_name} (T={temp_val}):")
        judge_results[temp_name] = {"temperature": temp_val, "problems": []}

        for p_data in temp_data["problems"]:
            pid = p_data["problem_id"]
            prob = all_problems[pid]

            correct_rollouts = [r for r in p_data["rollouts"] if r["is_correct"]]
            if not correct_rollouts:
                print(f"    P{pid}: no correct rollouts, skipping")
                judge_results[temp_name]["problems"].append({
                    "problem_id": pid, "n_correct": 0, "n_strategies": 0,
                    "judge_response": None,
                })
                continue

            # Build trace block (all from same model, label A)
            n = min(SAMPLE_PER_MODEL, len(correct_rollouts))
            step = max(1, len(correct_rollouts) // n)
            sampled = correct_rollouts[::step][:n]

            traces_block = []
            trace_labels = []
            for i, r in enumerate(sampled):
                tid = f"A{i+1}"
                trace_labels.append(tid)
                text = r["response"][:3000]
                if len(r["response"]) > 3000:
                    text += "\n[...truncated...]"
                traces_block.append(f"\n--- Trace {tid} ---\n{text}\n")

            traces_str = "\n".join(traces_block)

            prompt = f"""{JUDGE_PREAMBLE}

Problem: {prob['problem']}

Correct answer: {prob.get('answer', 'unknown')}

{traces_str}
{JUDGE_JSON}"""

            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                total_tokens += resp.usage.prompt_tokens + resp.usage.completion_tokens
                parsed = json.loads(resp.choices[0].message.content)

                n_strat = parsed.get("n_strategies", 0)
                print(f"    P{pid}: {p_data['n_correct']}/64 correct, {n_strat} strategies")

                judge_results[temp_name]["problems"].append({
                    "problem_id": pid,
                    "n_correct": p_data["n_correct"],
                    "n_strategies": n_strat,
                    "judge_response": parsed,
                    "trace_labels": trace_labels,
                })
            except Exception as e:
                print(f"    P{pid}: ERROR {e}")
                judge_results[temp_name]["problems"].append({
                    "problem_id": pid, "n_correct": p_data["n_correct"],
                    "error": str(e),
                })

            time.sleep(1.0)  # rate limit

    print(f"\n  Total judge tokens: {total_tokens}")
    cost = total_tokens * 0.10 / 1e6 + total_tokens * 0.40 / 1e6
    print(f"  Estimated cost: ${cost:.4f}")

    return judge_results, total_tokens


# ══════════════════════════════════════════════════════════
# STEP 2: Compute Rao's Q at each temperature
# ══════════════════════════════════════════════════════════

def compute_rao_q_for_temp(judge_entry):
    """Compute all 4 Rao's Q variants for a single problem at a single temperature."""
    parsed = judge_entry.get("judge_response")
    if not parsed or parsed.get("n_strategies", 0) <= 1:
        return {"gini_simpson": 0.0, "jaccard": 0.0, "gower": 0.0}

    strategies = {s["id"]: s for s in parsed["strategies"]}
    classifications = parsed["classifications"]

    # All traces are from model A (R1-Distill)
    counts = Counter()
    total = 0
    for label, strat_id in classifications.items():
        counts[strat_id] += 1
        total += 1

    if total == 0:
        return {"gini_simpson": 0.0, "jaccard": 0.0, "gower": 0.0}

    freq = {s: c / total for s, c in counts.items()}

    # Gini-Simpson
    gs = gini_simpson(freq)

    # Jaccard
    strat_tags = {sid: extract_tags(s["name"] + " " + s["description"]) for sid, s in strategies.items()}
    q_jac = 0.0
    for si, pi in freq.items():
        for sj, pj in freq.items():
            if si != sj:
                d = jaccard_distance(strat_tags.get(si, set()), strat_tags.get(sj, set()))
                q_jac += d * pi * pj

    # Gower
    strat_traits = {sid: extract_strategy_traits(s) for sid, s in strategies.items()}
    q_gow = 0.0
    for si, pi in freq.items():
        for sj, pj in freq.items():
            if si != sj:
                d = gower_distance(strat_traits.get(si, {}), strat_traits.get(sj, {}))
                q_gow += d * pi * pj

    return {"gini_simpson": round(gs, 6), "jaccard": round(q_jac, 6), "gower": round(q_gow, 6)}


def analyze_temp_sweep(judge_results):
    """Compute Rao's Q per temperature and test monotonicity."""
    print("\n" + "=" * 70)
    print("STEP 2: RAO'S Q AT EACH TEMPERATURE")
    print("=" * 70)

    temp_metrics = {}  # temp_name -> {metric: mean_value}

    for temp_name, temp_data in sorted(judge_results.items()):
        temp_val = temp_data["temperature"]
        metrics_per_problem = []

        for entry in temp_data["problems"]:
            if "error" in entry or not entry.get("judge_response"):
                continue
            q = compute_rao_q_for_temp(entry)
            q["problem_id"] = entry["problem_id"]
            q["n_correct"] = entry["n_correct"]
            q["n_strategies"] = entry.get("n_strategies", 0)
            q["pass_at_1"] = entry["n_correct"] / 64.0
            metrics_per_problem.append(q)

        if not metrics_per_problem:
            continue

        means = {
            "temperature": temp_val,
            "gini_simpson": round(np.mean([m["gini_simpson"] for m in metrics_per_problem]), 6),
            "jaccard": round(np.mean([m["jaccard"] for m in metrics_per_problem]), 6),
            "gower": round(np.mean([m["gower"] for m in metrics_per_problem]), 6),
            "mean_strategies": round(np.mean([m["n_strategies"] for m in metrics_per_problem]), 2),
            "mean_pass_at_1": round(np.mean([m["pass_at_1"] for m in metrics_per_problem]), 4),
            "mean_correct": round(np.mean([m["n_correct"] for m in metrics_per_problem]), 1),
            "per_problem": metrics_per_problem,
        }
        temp_metrics[temp_name] = means

        print(f"\n  {temp_name} (T={temp_val}):")
        print(f"    Gini-Simpson: {means['gini_simpson']:.4f}")
        print(f"    Jaccard:      {means['jaccard']:.4f}")
        print(f"    Gower:        {means['gower']:.4f}")
        print(f"    Mean strats:  {means['mean_strategies']:.1f}")
        print(f"    Mean p@1:     {means['mean_pass_at_1']:.3f}")
        print(f"    Mean correct: {means['mean_correct']:.0f}/64")

    return temp_metrics


# ══════════════════════════════════════════════════════════
# STEP 3: Monotonicity Test
# ══════════════════════════════════════════════════════════

def test_monotonicity(temp_metrics):
    """Test: Q(T=0.3) < Q(T=0.6) < Q(T=1.0) for each metric."""
    print("\n" + "=" * 70)
    print("STEP 3: MONOTONICITY TEST")
    print("=" * 70)

    temps_sorted = sorted(temp_metrics.values(), key=lambda x: x["temperature"])
    if len(temps_sorted) < 2:
        print("  Not enough temperatures to test monotonicity")
        return {"verdict": "SKIP"}

    results = {}
    for metric in ["gini_simpson", "jaccard", "gower"]:
        vals = [t[metric] for t in temps_sorted]
        temp_labels = [t["temperature"] for t in temps_sorted]

        # Check strict monotonicity
        monotone = all(vals[i] < vals[i+1] for i in range(len(vals)-1))
        # Check weak monotonicity
        weak_mono = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))

        results[metric] = {
            "values": list(zip(temp_labels, vals)),
            "strictly_monotone": monotone,
            "weakly_monotone": weak_mono,
        }

        arrow = " < ".join(f"T={t}:{v:.4f}" for t, v in zip(temp_labels, vals))
        status = "MONOTONE" if monotone else ("WEAK" if weak_mono else "VIOLATED")
        print(f"  {metric:<15}: {arrow}  [{status}]")

    # Also check pass@1 decreases
    p1_vals = [t["mean_pass_at_1"] for t in temps_sorted]
    p1_decreasing = all(p1_vals[i] >= p1_vals[i+1] for i in range(len(p1_vals)-1))
    results["pass_at_1"] = {
        "values": list(zip([t["temperature"] for t in temps_sorted], p1_vals)),
        "decreasing": p1_decreasing,
    }
    p1_arrow = " > ".join(f"T={t['temperature']}:{t['mean_pass_at_1']:.3f}" for t in temps_sorted)
    print(f"  {'pass@1':<15}: {p1_arrow}  [{'DECREASING' if p1_decreasing else 'NOT DECREASING'}]")

    # Verdict
    n_monotone = sum(1 for m in ["gini_simpson", "jaccard", "gower"] if results[m]["strictly_monotone"])
    verdict = "PASS" if n_monotone >= 2 else ("PARTIAL" if n_monotone >= 1 else "FAIL")
    print(f"\n  Verdict: {verdict} — {n_monotone}/3 metrics strictly monotone with temperature")

    results["verdict"] = verdict
    return results


# ══════════════════════════════════════════════════════════
# STEP 4: Pareto Frontier
# ══════════════════════════════════════════════════════════

def compute_pareto(temp_metrics):
    """Diversity-accuracy Pareto frontier across temperatures."""
    print("\n" + "=" * 70)
    print("STEP 4: DIVERSITY-ACCURACY PARETO FRONTIER")
    print("=" * 70)

    points = []
    for temp_name, data in sorted(temp_metrics.items()):
        points.append({
            "temperature": data["temperature"],
            "diversity_gower": data["gower"],
            "diversity_gs": data["gini_simpson"],
            "accuracy": data["mean_pass_at_1"],
        })

    print(f"  {'T':>4}  {'Gower':>8}  {'GS':>8}  {'p@1':>6}")
    print("  " + "-" * 35)
    for p in points:
        print(f"  {p['temperature']:>4.1f}  {p['diversity_gower']:>8.4f}  {p['diversity_gs']:>8.4f}  {p['accuracy']:>6.3f}")

    # Check if there's a clear tradeoff
    if len(points) >= 2:
        div_vals = [p["diversity_gower"] for p in points]
        acc_vals = [p["accuracy"] for p in points]
        rho, pval = sp_stats.spearmanr(div_vals, acc_vals)
        print(f"\n  Diversity-Accuracy correlation: rho={rho:.3f}  p={pval:.4f}")
        if rho < -0.5:
            print("  => Clear tradeoff: higher diversity = lower accuracy (expected)")
        elif abs(rho) < 0.3:
            print("  => No clear tradeoff: diversity and accuracy are independent")
        else:
            print(f"  => Unexpected: rho={rho:.3f}")

    return points


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("TEMPERATURE SWEEP ANALYSIS (Experiment 3)")
    print("=" * 70)

    # Check data exists
    if not SWEEP_DIR.exists():
        print(f"\nERROR: {SWEEP_DIR} not found.")
        print("Run these commands first:")
        print(f"  /Users/duy/Library/Python/3.9/bin/modal run pilot/modal_temp_sweep.py")
        print(f"  /Users/duy/Library/Python/3.9/bin/modal volume get spark-pilot-results temp_sweep/full {SWEEP_DIR}")
        sys.exit(1)

    # Step 1: Judge
    judge_results, total_tokens = run_judge_on_temp_traces()

    # Step 2: Compute Rao's Q
    temp_metrics = analyze_temp_sweep(judge_results)

    # Step 3: Monotonicity
    mono_results = test_monotonicity(temp_metrics)

    # Step 4: Pareto
    pareto_points = compute_pareto(temp_metrics)

    # ── Overall ──
    print("\n" + "=" * 70)
    print("TEMPERATURE SWEEP SUMMARY")
    print("=" * 70)
    print(f"  Judge tokens: {total_tokens}")
    print(f"  Monotonicity verdict: {mono_results.get('verdict', 'N/A')}")

    # Save
    all_results = {
        "judge_results": {k: {kk: vv for kk, vv in v.items() if kk != "problems"}
                          for k, v in judge_results.items()},
        "temp_metrics": {k: {kk: vv for kk, vv in v.items() if kk != "per_problem"}
                         for k, v in temp_metrics.items()},
        "monotonicity": mono_results,
        "pareto": pareto_points,
        "total_judge_tokens": total_tokens,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUT_PATH}")

    # Also save full judge results (with per-problem detail) separately
    judge_detail_path = Path("data/analysis/temp_sweep_judge_detail.json")
    with open(judge_detail_path, "w") as f:
        json.dump(judge_results, f, indent=2, default=str)
    print(f"  Judge detail saved to {judge_detail_path}")
