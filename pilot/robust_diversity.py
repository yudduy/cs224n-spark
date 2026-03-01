#!/usr/bin/env python3
"""
Robust Diversity Analysis: Fix Judge Root Cause + Paper-Ready Results
=====================================================================

Root cause (Run 23): LLM judge has κ=0.141 at trace level, CV=0.312 on
strategy counts. The problem is asking ONE call to identify strategies AND
classify traces simultaneously. Strategy numbering varies across runs,
making alignment impossible.

Fix: PAIRWISE CONSENSUS. Run judge 3×, extract per-run pairwise
"same/different" matrices (alignment-free), average them. This gives a
continuous, robust consensus distance for each trace pair.

Metrics computed:
  1. Pairwise Consensus Diversity (PCD) — mean consensus distance within model
     Equivalent to Gini-Simpson but estimated from 3-run consensus.
  2. Effective Strategy Count — 1/(1-PCD) = Hill number q=2
  3. Self-consistency κ — from the 3 runs (should be >> 0.141)
  4. Judge-free metrics — pass@k, answer entropy, behavioral profiles

Usage:
  python3 pilot/robust_diversity.py              # full run
  python3 pilot/robust_diversity.py --analyze    # skip judge, use cached

Cost: ~$0.50 (2 additional judge runs × 60 problems × ~25K tokens)
"""

import json
import math
import os
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# ── Paths ──

JUDGE_PATH = Path("data/analysis/llm_judge_pilot.json")
TRACES_DIR = Path("data/modal_runs/gen_traces_full")
PROBLEMS_PATH = TRACES_DIR / "problems.json"
CACHE_DIR = Path("data/analysis/consensus_judge_runs")
OUT_PATH = Path("data/analysis/robust_diversity.json")

MODELS = ["r1-distill", "nemotron-v1", "nemotron-v2", "nemotron-brorl"]
MODEL_LABELS = {
    "r1-distill": "A (R1-Distill, base)",
    "nemotron-v1": "B (v1, 2K RL steps)",
    "nemotron-v2": "C (v2, 3K RL steps)",
    "nemotron-brorl": "D (BroRL, breadth RL)",
}
MODEL_PREFIX = {"r1-distill": "A", "nemotron-v1": "B", "nemotron-v2": "C", "nemotron-brorl": "D"}
SAMPLE_PER_MODEL = 8
LLM_MODEL = "gpt-5-nano"

# ── Load data ──

with open(PROBLEMS_PATH) as f:
    problems = json.load(f)

all_traces = {}
for model in MODELS:
    with open(TRACES_DIR / model / "traces.json") as f:
        all_traces[model] = json.load(f)


# ── OpenAI client ──

def get_client():
    from openai import OpenAI
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
    return OpenAI(api_key=api_key)


# ── Judge prompt (same as llm_judge_pilot.py) ──

def build_prompt(pid):
    prob = problems[pid]
    prob_text = prob["problem"]
    answer = prob.get("answer", "unknown")

    traces_block = []
    trace_labels = []
    for m in MODELS:
        label_prefix = MODEL_LABELS[m].split(" ")[0]
        p_data = all_traces[m]["problems"][pid]
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

    traces_str = "\n".join(traces_block)

    prompt = f"""You are analyzing mathematical reasoning traces from 4 AI model checkpoints on the SAME training trajectory.
Your task: identify how many DISTINCT high-level solution strategies are used across all traces.

IMPORTANT:
- A "strategy" is a fundamentally different MATHEMATICAL APPROACH (e.g., algebraic vs geometric, direct computation vs proof by contradiction, substitution vs factoring)
- Differences in verbosity, formatting, step ordering, or phrasing are NOT different strategies
- Two traces that use the same approach but one is more detailed or rambles more = SAME strategy
- Focus on the CORE mathematical method, not surface presentation

Problem: {prob_text}

Correct answer: {answer}

{traces_str}

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
  "notes": "<any observations about strategy diversity or lack thereof>"
}}"""
    return prompt, trace_labels


# ══════════════════════════════════════════════════════════
# PHASE 1: Run judge 3× (reuse Run 16 as run 0, run 2 new)
# ══════════════════════════════════════════════════════════

def run_judge_passes():
    """Run judge 2 additional times on all 60 problems, with per-problem checkpointing."""
    print("\n" + "=" * 70)
    print("PHASE 1: CONSENSUS JUDGE (3 runs × 60 problems)")
    print("=" * 70)

    # Run 0 = existing Run 16 data
    with open(JUDGE_PATH) as f:
        run0 = json.load(f)
    print(f"  Run 0 (existing): {len(run0)} problems loaded")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()
    total_tokens = 0

    for run_idx in [1, 2]:
        cache_path = CACHE_DIR / f"run_{run_idx}.json"

        # Check cache
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            done_pids = {r["problem_id"] for r in cached if "llm_response" in r}
            print(f"  Run {run_idx}: {len(done_pids)}/60 cached")
            if len(done_pids) >= 60:
                continue
            results = cached
        else:
            results = []
            done_pids = set()

        for pid in range(60):
            if pid in done_pids:
                continue

            prompt, trace_labels = build_prompt(pid)

            # Retry with exponential backoff
            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                    )
                    total_tokens += resp.usage.prompt_tokens + resp.usage.completion_tokens
                    parsed = json.loads(resp.choices[0].message.content)

                    results.append({
                        "problem_id": pid,
                        "tier": problems[pid]["tier"],
                        "subject": problems[pid]["subject"],
                        "llm_response": parsed,
                        "trace_labels": trace_labels,
                    })
                    n_strat = parsed.get("n_strategies", 0)
                    if pid % 10 == 0:
                        print(f"    Run {run_idx} P{pid}: {n_strat} strategies")
                    break

                except Exception as e:
                    err = str(e)
                    if "429" in err or "rate_limit" in err:
                        wait = 2 ** (attempt + 1)
                        print(f"    Run {run_idx} P{pid}: rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"    Run {run_idx} P{pid}: ERROR {e}")
                        results.append({"problem_id": pid, "error": str(e)})
                        break

            time.sleep(1.5)  # rate limit courtesy

            # Checkpoint every 5 problems
            if pid % 5 == 4:
                with open(cache_path, "w") as f:
                    json.dump(results, f, indent=2)

        # Final save
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Run {run_idx}: complete ({len(results)} problems)")

    print(f"  Total new tokens: {total_tokens}")
    cost = total_tokens * 0.50 / 1e6  # rough estimate
    print(f"  Estimated cost: ${cost:.4f}")

    return run0


# ══════════════════════════════════════════════════════════
# PHASE 2: Pairwise Consensus Analysis
# ══════════════════════════════════════════════════════════

def extract_pairwise_same(classifications, trace_labels):
    """From a judge classification dict, extract pairwise same/different matrix."""
    pairs = {}
    for i, ti in enumerate(trace_labels):
        for j, tj in enumerate(trace_labels):
            if i < j:
                si = classifications.get(ti)
                sj = classifications.get(tj)
                if si is not None and sj is not None:
                    pairs[(ti, tj)] = 1 if si == sj else 0
    return pairs


def compute_consensus(run0, run1, run2):
    """Compute pairwise consensus from 3 judge runs."""
    print("\n" + "=" * 70)
    print("PHASE 2: PAIRWISE CONSENSUS COMPUTATION")
    print("=" * 70)

    # Index runs by problem_id
    def index_run(data):
        return {r["problem_id"]: r for r in data if "llm_response" in r}

    r0 = index_run(run0)
    r1 = index_run(run1)
    r2 = index_run(run2)

    all_runs = [r0, r1, r2]
    n_runs = len(all_runs)

    consensus_data = []  # per-problem consensus results
    kappa_per_problem = []

    for pid in range(60):
        # Get classifications from each run
        run_pairs = []
        run_classifications = []
        trace_labels = None

        for run in all_runs:
            if pid not in run:
                continue
            entry = run[pid]
            cls = entry["llm_response"]["classifications"]
            labels = entry.get("trace_labels", list(cls.keys()))
            if trace_labels is None:
                trace_labels = labels
            pairs = extract_pairwise_same(cls, labels)
            run_pairs.append(pairs)
            run_classifications.append(cls)

        if len(run_pairs) < 2 or trace_labels is None:
            continue

        # Consensus: average pairwise agreement across runs
        all_pair_keys = set()
        for rp in run_pairs:
            all_pair_keys.update(rp.keys())

        consensus_pairs = {}
        for pair in all_pair_keys:
            votes = [rp.get(pair) for rp in run_pairs if pair in rp]
            if votes:
                consensus_pairs[pair] = np.mean(votes)  # fraction agreeing "same"

        # Per-model metrics
        per_model = {}
        for model in MODELS:
            prefix = MODEL_PREFIX[model]
            model_labels = [l for l in trace_labels if l.startswith(prefix)]

            if len(model_labels) < 2:
                per_model[model] = {"pcd": 0.0, "eff_strats": 1.0, "n_traces": len(model_labels)}
                continue

            # Pairwise consensus distance within model
            distances = []
            for i, li in enumerate(model_labels):
                for j, lj in enumerate(model_labels):
                    if i < j:
                        pair = (li, lj) if (li, lj) in consensus_pairs else (lj, li)
                        if pair in consensus_pairs:
                            distances.append(1.0 - consensus_pairs[pair])  # distance = 1 - same_prob

            pcd = np.mean(distances) if distances else 0.0
            eff_strats = 1.0 / (1.0 - pcd) if pcd < 1.0 else float('inf')

            per_model[model] = {
                "pcd": round(pcd, 6),
                "eff_strats": round(eff_strats, 4),
                "n_traces": len(model_labels),
                "n_pairs": len(distances),
            }

        # Strategy count consensus (median across runs)
        strat_counts = []
        for run in all_runs:
            if pid in run:
                strat_counts.append(run[pid]["llm_response"].get("n_strategies", 1))
        median_count = int(np.median(strat_counts))
        count_range = (min(strat_counts), max(strat_counts))
        count_agree = strat_counts.count(median_count) / len(strat_counts)

        # Per-problem kappa: agreement on pairwise same/different across runs
        if len(run_pairs) >= 2:
            agreements = []
            for pair in all_pair_keys:
                votes = [rp.get(pair) for rp in run_pairs if pair in rp]
                if len(votes) >= 2:
                    agreements.append(1 if len(set(votes)) == 1 else 0)
            if agreements:
                po = np.mean(agreements)
                # Expected agreement: based on overall same/different rate
                all_votes = [v for rp in run_pairs for v in rp.values()]
                pe = np.mean(all_votes) ** 2 + (1 - np.mean(all_votes)) ** 2
                kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
                kappa_per_problem.append(kappa)
            else:
                kappa = float("nan")
        else:
            kappa = float("nan")

        consensus_data.append({
            "problem_id": pid,
            "tier": problems[pid]["tier"],
            "median_strat_count": median_count,
            "strat_count_range": count_range,
            "strat_count_agreement": round(count_agree, 2),
            "pairwise_kappa": round(kappa, 4) if not np.isnan(kappa) else None,
            "per_model": per_model,
        })

    # Summary
    mean_kappa = np.mean([k for k in kappa_per_problem if not np.isnan(k)]) if kappa_per_problem else 0
    print(f"\n  Pairwise consensus κ: {mean_kappa:.3f} (vs 0.141 before)")
    print(f"  Problems analyzed: {len(consensus_data)}")

    # Multi-strategy count
    multi = [c for c in consensus_data if c["median_strat_count"] > 1]
    print(f"  Multi-strategy problems (median): {len(multi)}/60")

    return consensus_data, mean_kappa


# ══════════════════════════════════════════════════════════
# PHASE 3: Judge-Free Metrics
# ══════════════════════════════════════════════════════════

def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_judge_free_metrics():
    """Compute deterministic metrics that need no LLM judge."""
    print("\n" + "=" * 70)
    print("PHASE 3: JUDGE-FREE METRICS")
    print("=" * 70)

    results = {model: {
        "pass_at_k": {}, "mean_correct": 0, "mean_unique_answers": 0,
        "mean_trace_length": 0, "answer_entropy_mean": 0,
        "per_tier": {},
    } for model in MODELS}

    K_VALUES = [1, 4, 8, 16, 32, 64]

    # Behavioral profiles: per-model × per-problem pass@1
    profiles = {model: [] for model in MODELS}

    for model in MODELS:
        pass_ks = {str(k): [] for k in K_VALUES}
        n_corrects = []
        unique_answers = []
        trace_lengths = []
        answer_entropies = []
        tier_data = {tier: {"pass_at_1": [], "n_correct": []} for tier in ["easy", "medium", "hard"]}

        for pid in range(60):
            p_data = all_traces[model]["problems"][pid]
            nc = p_data["n_correct"]
            n_corrects.append(nc)
            profiles[model].append(pass_at_k(64, nc, 1))

            for k in K_VALUES:
                pass_ks[str(k)].append(pass_at_k(64, nc, k))

            # Unique answers (all rollouts)
            answers = set()
            lengths = []
            answer_counts = Counter()
            for r in p_data["rollouts"]:
                if r["final_answer"]:
                    ans = r["final_answer"].strip().replace(" ", "").lower()
                    answers.add(ans)
                    answer_counts[ans] += 1
                lengths.append(r["n_tokens"])

            unique_answers.append(len(answers))
            trace_lengths.extend(lengths)

            # Answer entropy (Shannon)
            total = sum(answer_counts.values())
            if total > 0:
                probs = [c / total for c in answer_counts.values()]
                h = -sum(p * np.log2(p) for p in probs if p > 0)
                answer_entropies.append(h)

            tier = problems[pid]["tier"]
            tier_data[tier]["pass_at_1"].append(pass_at_k(64, nc, 1))
            tier_data[tier]["n_correct"].append(nc)

        results[model]["pass_at_k"] = {k: round(np.mean(v), 4) for k, v in pass_ks.items()}
        results[model]["mean_correct"] = round(np.mean(n_corrects), 2)
        results[model]["mean_unique_answers"] = round(np.mean(unique_answers), 2)
        results[model]["mean_trace_length"] = round(np.mean(trace_lengths), 0)
        results[model]["answer_entropy_mean"] = round(np.mean(answer_entropies), 4)

        for tier, td in tier_data.items():
            results[model]["per_tier"][tier] = {
                "pass_at_1": round(np.mean(td["pass_at_1"]), 4),
                "mean_correct": round(np.mean(td["n_correct"]), 2),
            }

    # Behavioral distance between models (1 - Spearman correlation of pass@1 profiles)
    behavioral_distances = {}
    for mi, mj in combinations(MODELS, 2):
        rho, _ = sp_stats.spearmanr(profiles[mi], profiles[mj])
        behavioral_distances[f"{mi}_vs_{mj}"] = round(1 - rho, 4)

    # Print summary
    print(f"\n  {'Model':<20} {'p@1':>6} {'p@8':>6} {'p@64':>6} {'unique':>6} {'H(ans)':>6} {'len':>6}")
    print("  " + "-" * 60)
    for model in MODELS:
        r = results[model]
        pk = r["pass_at_k"]
        print(f"  {model:<20} {pk['1']:>6.3f} {pk['8']:>6.3f} {pk['64']:>6.3f} {r['mean_unique_answers']:>6.1f} {r['answer_entropy_mean']:>6.3f} {r['mean_trace_length']:>6.0f}")

    print(f"\n  Behavioral distances (1 - Spearman ρ):")
    for pair, d in sorted(behavioral_distances.items()):
        print(f"    {pair}: {d:.3f}")

    return results, behavioral_distances, profiles


# ══════════════════════════════════════════════════════════
# PHASE 4: Paper-Ready Analysis
# ══════════════════════════════════════════════════════════

def paper_analysis(consensus_data, jf_metrics, behavioral_distances):
    """Combine consensus judge + judge-free metrics into paper tables."""
    print("\n" + "=" * 70)
    print("PHASE 4: PAPER-READY ANALYSIS")
    print("=" * 70)

    # ── Table 1: Per-model diversity (consensus) ──
    print("\n  TABLE 1: Strategy Diversity (Pairwise Consensus, 3-run)")
    print(f"  {'Model':<20} {'PCD':>8} {'EffStrats':>10} {'p@1':>6} {'p@64':>6}")
    print("  " + "-" * 55)

    model_pcds = {model: [] for model in MODELS}
    for entry in consensus_data:
        for model in MODELS:
            if model in entry["per_model"]:
                model_pcds[model].append(entry["per_model"][model]["pcd"])

    table1 = {}
    for model in MODELS:
        pcd_vals = model_pcds[model]
        mean_pcd = np.mean(pcd_vals) if pcd_vals else 0
        eff = 1.0 / (1.0 - mean_pcd) if mean_pcd < 1.0 else float('inf')
        pk = jf_metrics[model]["pass_at_k"]
        print(f"  {model:<20} {mean_pcd:>8.4f} {eff:>10.2f} {pk['1']:>6.3f} {pk['64']:>6.3f}")
        table1[model] = {"pcd": round(mean_pcd, 6), "eff_strats": round(eff, 4)}

    # ── Table 2: Per-tier breakdown ──
    print("\n  TABLE 2: Diversity by Difficulty Tier")
    tier_pcds = {tier: {model: [] for model in MODELS} for tier in ["easy", "medium", "hard"]}
    for entry in consensus_data:
        tier = entry["tier"]
        for model in MODELS:
            if model in entry["per_model"]:
                tier_pcds[tier][model].append(entry["per_model"][model]["pcd"])

    for tier in ["easy", "medium", "hard"]:
        print(f"\n  {tier.upper()}:")
        print(f"  {'Model':<20} {'PCD':>8} {'EffStrats':>10} {'p@1':>6}")
        print("  " + "-" * 45)
        for model in MODELS:
            vals = tier_pcds[tier][model]
            mean_pcd = np.mean(vals) if vals else 0
            eff = 1.0 / (1.0 - mean_pcd) if mean_pcd < 1.0 else float('inf')
            tier_p1 = jf_metrics[model]["per_tier"][tier]["pass_at_1"]
            print(f"  {model:<20} {mean_pcd:>8.4f} {eff:>10.2f} {tier_p1:>6.3f}")

    # ── Table 3: Multi-strategy problems ──
    multi = [c for c in consensus_data if c["median_strat_count"] > 1]
    print(f"\n  TABLE 3: Multi-Strategy Problems ({len(multi)}/60)")
    print(f"  {'PID':>4} {'Tier':<7} {'N_strat':>7} {'Range':>8} {'Agree':>6}  " +
          "  ".join(f"{m[:8]:>8}" for m in MODELS))
    print("  " + "-" * 80)
    for entry in sorted(multi, key=lambda x: x["problem_id"]):
        pid = entry["problem_id"]
        pcds = "  ".join(f"{entry['per_model'].get(m, {}).get('pcd', 0):>8.3f}" for m in MODELS)
        r = entry["strat_count_range"]
        print(f"  {pid:>4} {entry['tier']:<7} {entry['median_strat_count']:>7} {f'{r[0]}-{r[1]}':>8} {entry['strat_count_agreement']:>5.0%}  {pcds}")

    # ── Key claims verification ──
    print("\n  " + "=" * 70)
    print("  KEY CLAIMS VERIFICATION")
    print("  " + "=" * 70)

    # Claim 1: R1-Distill is most diverse
    r1_pcd = table1["r1-distill"]["pcd"]
    others_pcds = {m: table1[m]["pcd"] for m in MODELS if m != "r1-distill"}
    r1_most_diverse = all(r1_pcd > v for v in others_pcds.values())
    print(f"\n  Claim 1: R1-Distill is most diverse")
    print(f"    R1 PCD = {r1_pcd:.4f}")
    for m, v in others_pcds.items():
        print(f"    {m} PCD = {v:.4f}  (R1 {'>' if r1_pcd > v else '<='} {m})")
    print(f"    VERDICT: {'CONFIRMED' if r1_most_diverse else 'REJECTED'}")

    # Claim 2: ProRL narrows strategies
    r1_pcd_vals = model_pcds["r1-distill"]
    v2_pcd_vals = model_pcds["nemotron-v2"]
    # Paired test (same problems)
    if len(r1_pcd_vals) == len(v2_pcd_vals):
        stat, pval = sp_stats.wilcoxon(r1_pcd_vals, v2_pcd_vals, alternative="greater")
        print(f"\n  Claim 2: ProRL narrows strategies (R1 > v2)")
        print(f"    Wilcoxon signed-rank: W={stat:.0f}, p={pval:.6f}")
        print(f"    VERDICT: {'CONFIRMED (p<0.05)' if pval < 0.05 else 'NOT SIGNIFICANT'}")
    else:
        print(f"\n  Claim 2: SKIP (unequal sizes)")

    # Claim 3: v1 anomaly (dip then recovery)
    v1_pcd = table1["nemotron-v1"]["pcd"]
    v2_pcd = table1["nemotron-v2"]["pcd"]
    brorl_pcd = table1["nemotron-brorl"]["pcd"]
    print(f"\n  Claim 3: Strategy evolution trajectory")
    print(f"    R1→v1: {r1_pcd:.4f} → {v1_pcd:.4f} ({'↓' if v1_pcd < r1_pcd else '↑'})")
    print(f"    v1→v2: {v1_pcd:.4f} → {v2_pcd:.4f} ({'↓' if v2_pcd < v1_pcd else '↑'})")
    print(f"    v2→BroRL: {v2_pcd:.4f} → {brorl_pcd:.4f} ({'↓' if brorl_pcd < v2_pcd else '↑'})")

    # Claim 4: Diversity-accuracy tradeoff
    pcd_means = [table1[m]["pcd"] for m in MODELS]
    p1_means = [jf_metrics[m]["pass_at_k"]["1"] for m in MODELS]
    rho, pval = sp_stats.spearmanr(pcd_means, p1_means)
    print(f"\n  Claim 4: Diversity-accuracy tradeoff")
    print(f"    Spearman ρ(PCD, p@1) = {rho:.3f} (p={pval:.4f})")
    tradeoff = rho < -0.5
    print(f"    VERDICT: {'TRADEOFF CONFIRMED' if tradeoff else 'NO CLEAR TRADEOFF'}")

    # Claim 5: 4× inference efficiency (R1@4 ≈ v2@1)
    r1_p4 = jf_metrics["r1-distill"]["pass_at_k"]["4"]
    v2_p1 = jf_metrics["nemotron-v2"]["pass_at_k"]["1"]
    print(f"\n  Claim 5: Inference efficiency (R1@4 ≈ v2@1)")
    print(f"    R1 pass@4 = {r1_p4:.4f}")
    print(f"    v2 pass@1 = {v2_p1:.4f}")
    print(f"    Ratio: {abs(r1_p4 - v2_p1) / v2_p1:.1%} difference")

    return table1, multi


# ══════════════════════════════════════════════════════════
# PHASE 5: Validation of Consensus Approach
# ══════════════════════════════════════════════════════════

def validate_consensus(consensus_data, kappa):
    """Quick validation of the consensus approach."""
    print("\n" + "=" * 70)
    print("PHASE 5: CONSENSUS VALIDATION")
    print("=" * 70)

    # Strategy count agreement
    agreements = [c["strat_count_agreement"] for c in consensus_data]
    mean_agree = np.mean(agreements)
    print(f"  Strategy count agreement (across 3 runs): {mean_agree:.1%}")
    perfect = sum(1 for a in agreements if a == 1.0)
    print(f"  Problems with perfect count agreement: {perfect}/60")

    # Pairwise κ
    kappas = [c["pairwise_kappa"] for c in consensus_data if c["pairwise_kappa"] is not None]
    print(f"  Mean pairwise κ: {np.mean(kappas):.3f} (was 0.141 in single-run validation)")
    print(f"  Median pairwise κ: {np.median(kappas):.3f}")

    # PCD stability: bootstrap CI on the model ranking
    print(f"\n  PCD rank stability (bootstrap):")
    model_pcds = {model: [] for model in MODELS}
    for entry in consensus_data:
        for model in MODELS:
            if model in entry["per_model"]:
                model_pcds[model].append(entry["per_model"][model]["pcd"])

    rng = np.random.RandomState(42)
    rank_counts = {model: Counter() for model in MODELS}
    for _ in range(1000):
        idx = rng.choice(60, size=60, replace=True)
        means = {}
        for model in MODELS:
            vals = [model_pcds[model][i] for i in idx if i < len(model_pcds[model])]
            means[model] = np.mean(vals) if vals else 0
        ranked = sorted(MODELS, key=lambda m: means[m], reverse=True)
        for rank, model in enumerate(ranked):
            rank_counts[model][rank + 1] += 1

    for model in MODELS:
        dist = rank_counts[model]
        mode_rank = dist.most_common(1)[0][0]
        mode_pct = dist.most_common(1)[0][1] / 10
        print(f"    {model:<20}: rank {mode_rank} in {mode_pct:.0f}% of bootstraps")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    analyze_only = "--analyze" in sys.argv

    print("=" * 70)
    print("ROBUST DIVERSITY ANALYSIS")
    print("Fix: Pairwise Consensus Judge (3 runs, alignment-free)")
    print("=" * 70)

    # Phase 1: Run judge
    if not analyze_only:
        run0 = run_judge_passes()
    else:
        with open(JUDGE_PATH) as f:
            run0 = json.load(f)
        print(f"  Loaded existing Run 0: {len(run0)} problems")

    # Load all 3 runs
    with open(JUDGE_PATH) as f:
        run0_data = json.load(f)

    run1_path = CACHE_DIR / "run_1.json"
    run2_path = CACHE_DIR / "run_2.json"

    if not run1_path.exists() or not run2_path.exists():
        print("ERROR: Judge runs not complete. Run without --analyze first.")
        sys.exit(1)

    with open(run1_path) as f:
        run1_data = json.load(f)
    with open(run2_path) as f:
        run2_data = json.load(f)

    # Phase 2: Consensus
    consensus_data, mean_kappa = compute_consensus(run0_data, run1_data, run2_data)

    # Phase 3: Judge-free
    jf_metrics, behavioral_distances, profiles = compute_judge_free_metrics()

    # Phase 4: Paper analysis
    table1, multi = paper_analysis(consensus_data, jf_metrics, behavioral_distances)

    # Phase 5: Validation
    validate_consensus(consensus_data, mean_kappa)

    # ── Save everything ──
    output = {
        "consensus": consensus_data,
        "judge_free": jf_metrics,
        "behavioral_distances": behavioral_distances,
        "table1_summary": table1,
        "n_multi_strategy": len(multi),
        "mean_pairwise_kappa": round(mean_kappa, 4),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  All results saved to {OUT_PATH}")
    print("  Done.")
