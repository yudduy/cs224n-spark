#!/usr/bin/env python3
"""
Rao's Q Internal Validation (Experiments 1a–1f)
================================================
All tests use existing data only — no API calls, no cost.

1a. Rarefaction curves — strategy count vs subsample size
1b. Bootstrap CIs — BCa 95% confidence intervals for Rao's Q
1c. Convergent validity — correlation with Shannon/Vendi/answer diversity
1d. Discriminant validity — non-correlation with length/accuracy/difficulty
1e. Predictive validity — diversity predicts pass@k scaling gap
1f. Replication principle — pooled Q > individual Q

Usage:
  python3 pilot/validate_rao_q.py
"""

import json
import math
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# ── Paths ──

JUDGE_PATH = Path("data/analysis/llm_judge_pilot.json")
TRACES_DIR = Path("data/modal_runs/gen_traces_full")
PROBLEMS_PATH = TRACES_DIR / "problems.json"
OUT_PATH = Path("data/analysis/rao_q_validation.json")

MODELS = ["r1-distill", "nemotron-v1", "nemotron-v2", "nemotron-brorl"]
MODEL_PREFIX = {"r1-distill": "A", "nemotron-v1": "B", "nemotron-v2": "C", "nemotron-brorl": "D"}

# ── Load data ──

with open(JUDGE_PATH) as f:
    judge_data = json.load(f)

with open(PROBLEMS_PATH) as f:
    problems = json.load(f)

all_traces = {}
for model in MODELS:
    with open(TRACES_DIR / model / "traces.json") as f:
        all_traces[model] = json.load(f)

multi_strat = [j for j in judge_data if j.get("llm_response", {}).get("n_strategies", 0) > 1]
multi_strat_ids = [j["problem_id"] for j in multi_strat]

print(f"Loaded: {len(judge_data)} problems, {len(multi_strat)} multi-strategy")
print(f"Multi-strategy IDs: {multi_strat_ids}")


# ── Reuse from rao_q_prototype.py ──

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
    "symmetry": ["symmetr", "palindrom", "invariant"],
    "case_analysis": ["case", "case analysis", "cases"],
    "formula": ["formula", "closed-form", "well-known", "theorem", "identity"],
    "scaling": ["scale", "proportion", "ratio", "percent"],
    "pattern": ["pattern", "sequence", "recurrence"],
    "optimization": ["optim", "minimize", "maximize", "lagrange", "am-gm", "inequality"],
    "polynomial": ["polynomial", "roots", "coefficients", "vieta"],
    "probability": ["probabilit", "expectation", "random", "expected value"],
    "logarithmic": ["logarithm", "log", "ln", "exponenti"],
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


def get_model_strategy_freq(judge_entry, model):
    prefix = MODEL_PREFIX[model]
    classifications = judge_entry["llm_response"]["classifications"]
    counts = Counter()
    total = 0
    for label, strat_id in classifications.items():
        if label.startswith(prefix):
            counts[strat_id] += 1
            total += 1
    if total == 0:
        return {}
    return {s: c / total for s, c in counts.items()}


def gini_simpson(freq_dict):
    if not freq_dict:
        return 0.0
    return 1.0 - sum(p**2 for p in freq_dict.values())


def rao_q_jaccard(judge_entry, model):
    freq = get_model_strategy_freq(judge_entry, model)
    strategies = {s["id"]: s for s in judge_entry["llm_response"]["strategies"]}
    if len(freq) <= 1:
        return 0.0
    strat_tags = {}
    for sid, s in strategies.items():
        strat_tags[sid] = extract_tags(s["name"] + " " + s["description"])
    q = 0.0
    for si, pi in freq.items():
        for sj, pj in freq.items():
            if si != sj:
                d = jaccard_distance(strat_tags.get(si, set()), strat_tags.get(sj, set()))
                q += d * pi * pj
    return q


def rao_q_gower(judge_entry, model):
    freq = get_model_strategy_freq(judge_entry, model)
    strategies = {s["id"]: s for s in judge_entry["llm_response"]["strategies"]}
    if len(freq) <= 1:
        return 0.0
    strat_traits = {sid: extract_strategy_traits(s) for sid, s in strategies.items()}
    q = 0.0
    for si, pi in freq.items():
        for sj, pj in freq.items():
            if si != sj:
                d = gower_distance(strat_traits.get(si, {}), strat_traits.get(sj, {}))
                q += d * pi * pj
    return q


def rao_q_from_freq_and_dists(freq, dist_matrix):
    """Generic Rao's Q from frequency dict and distance matrix dict."""
    if len(freq) <= 1:
        return 0.0
    q = 0.0
    for si, pi in freq.items():
        for sj, pj in freq.items():
            if si != sj:
                q += dist_matrix.get((si, sj), 1.0) * pi * pj
    return q


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ══════════════════════════════════════════════════════════
# EXPERIMENT 1a: Rarefaction Curves
# ══════════════════════════════════════════════════════════

def run_rarefaction():
    """Subsample k traces from 64, count strategies. Check if curves plateau."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1a: RAREFACTION CURVES")
    print("=" * 70)

    K_VALUES = [4, 8, 16, 32, 64]
    N_RESAMPLES = 200
    rng = np.random.RandomState(42)

    rarefaction = {}  # model -> problem_id -> {k: [strategy_counts]}

    for entry in multi_strat:
        pid = entry["problem_id"]
        classifications = entry["llm_response"]["classifications"]

        for model in MODELS:
            prefix = MODEL_PREFIX[model]
            # Get all trace classifications for this model
            model_cls = [(label, strat_id) for label, strat_id in classifications.items()
                         if label.startswith(prefix)]

            if len(model_cls) == 0:
                continue

            if model not in rarefaction:
                rarefaction[model] = {}
            if pid not in rarefaction[model]:
                rarefaction[model][pid] = {}

            for k in K_VALUES:
                if k > len(model_cls):
                    # Can't subsample more than available
                    rarefaction[model][pid][k] = [len(set(s for _, s in model_cls))]
                    continue

                counts = []
                for _ in range(N_RESAMPLES):
                    idx = rng.choice(len(model_cls), size=k, replace=False)
                    sampled_strats = set(model_cls[i][1] for i in idx)
                    counts.append(len(sampled_strats))
                rarefaction[model][pid][k] = counts

    # Summarize: for each model, average across problems
    print(f"\n  Mean strategy count at each subsample size (averaged over {len(multi_strat)} problems):")
    print(f"  {'Model':<20}", end="")
    for k in K_VALUES:
        print(f" {'k='+str(k):>8}", end="")
    print()
    print("  " + "-" * 65)

    plateau_results = {}
    for model in MODELS:
        means = []
        for k in K_VALUES:
            vals = []
            for pid in rarefaction.get(model, {}):
                vals.extend(rarefaction[model][pid].get(k, []))
            mean = np.mean(vals) if vals else 0
            means.append(mean)
        print(f"  {model:<20}", end="")
        for m in means:
            print(f" {m:>8.2f}", end="")
        print()

        # Plateau test: is the gain from k=32 to k=64 < 5% of k=4 to k=64?
        if len(means) >= 2 and means[-1] > 0:
            total_gain = means[-1] - means[0]
            last_gain = means[-1] - means[-2]  # k=32 to k=64
            plateau_ratio = last_gain / total_gain if total_gain > 0 else 0
            plateau_results[model] = {
                "total_gain": round(total_gain, 4),
                "last_gain": round(last_gain, 4),
                "plateau_ratio": round(plateau_ratio, 4),
                "plateaued": plateau_ratio < 0.05,
            }
            status = "PLATEAU" if plateau_ratio < 0.05 else f"still rising ({plateau_ratio:.1%})"
            print(f"    → {status}")

    all_plateaued = all(v["plateaued"] for v in plateau_results.values())
    verdict = "PASS" if all_plateaued else "PARTIAL"
    print(f"\n  Verdict: {verdict} — {'All curves plateau by k=32' if all_plateaued else 'Some curves still rising'}")

    return {
        "rarefaction_means": {
            model: {str(k): round(np.mean([
                np.mean(rarefaction[model][pid].get(k, [0]))
                for pid in rarefaction.get(model, {})
            ]), 4) for k in K_VALUES}
            for model in MODELS
        },
        "plateau_results": plateau_results,
        "verdict": verdict,
    }


# ══════════════════════════════════════════════════════════
# EXPERIMENT 1b: Bootstrap CIs (BCa)
# ══════════════════════════════════════════════════════════

def bca_ci(data, stat_func, n_boot=1000, alpha=0.05, rng=None):
    """Bias-corrected accelerated bootstrap confidence interval."""
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(data)
    observed = stat_func(data)

    # Bootstrap distribution
    boot_stats = np.array([stat_func(data[rng.choice(n, size=n, replace=True)]) for _ in range(n_boot)])

    # Bias correction
    z0 = sp_stats.norm.ppf(np.mean(boot_stats < observed))

    # Acceleration (jackknife)
    jackknife = np.array([stat_func(np.delete(data, i)) for i in range(n)])
    jack_mean = np.mean(jackknife)
    diffs = jack_mean - jackknife
    a = np.sum(diffs ** 3) / (6.0 * (np.sum(diffs ** 2)) ** 1.5) if np.sum(diffs ** 2) > 0 else 0

    # Adjusted percentiles
    z_alpha = sp_stats.norm.ppf(alpha / 2)
    z_1alpha = sp_stats.norm.ppf(1 - alpha / 2)

    def adjusted_percentile(z):
        p = sp_stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        return np.clip(p, 0.001, 0.999)

    lo = np.percentile(boot_stats, 100 * adjusted_percentile(z_alpha))
    hi = np.percentile(boot_stats, 100 * adjusted_percentile(z_1alpha))

    return observed, lo, hi


def run_bootstrap_cis():
    """Bootstrap CIs for Rao's Q (Gower) per model."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1b: BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)

    N_BOOT = 2000
    rng = np.random.RandomState(42)

    results = {}

    for model in MODELS:
        # Collect per-problem Rao's Q values (Gower)
        q_values = []
        for entry in multi_strat:
            q = rao_q_gower(entry, model)
            q_values.append(q)
        q_arr = np.array(q_values)

        obs, lo, hi = bca_ci(q_arr, np.mean, n_boot=N_BOOT, rng=rng)
        results[model] = {
            "mean": round(obs, 6),
            "ci_lo": round(lo, 6),
            "ci_hi": round(hi, 6),
            "n_problems": len(q_values),
        }
        print(f"  {model:<20}: Q={obs:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

    # Check overlap between R1-Distill and others
    r1_hi = results["r1-distill"]["ci_hi"]
    r1_lo = results["r1-distill"]["ci_lo"]
    overlaps = {}
    for model in MODELS:
        if model == "r1-distill":
            continue
        m_lo = results[model]["ci_lo"]
        m_hi = results[model]["ci_hi"]
        overlap = r1_lo < m_hi and m_lo < r1_hi
        overlaps[model] = overlap
        status = "OVERLAP" if overlap else "NO OVERLAP"
        print(f"    R1 vs {model}: {status}")

    any_overlap = any(overlaps.values())
    verdict = "FAIL" if any_overlap else "PASS"
    print(f"\n  Verdict: {verdict} — {'CIs overlap' if any_overlap else 'R1 CI separated from all others'}")

    return {"cis": results, "r1_overlaps": overlaps, "verdict": verdict}


# ══════════════════════════════════════════════════════════
# EXPERIMENT 1c: Convergent Validity
# ══════════════════════════════════════════════════════════

def shannon_entropy(freq_dict):
    """Shannon entropy of strategy distribution."""
    if not freq_dict:
        return 0.0
    return -sum(p * np.log2(p) for p in freq_dict.values() if p > 0)


def vendi_score(freq_dict, dist_matrix):
    """Vendi Score = exp(Shannon entropy of eigenvalues of similarity kernel).
    Kernel K_ij = 1 - d(i,j), where d is the distance matrix."""
    if len(freq_dict) <= 1:
        return 1.0

    strat_ids = sorted(freq_dict.keys())
    n = len(strat_ids)
    # Build similarity kernel weighted by frequencies
    # K_ij = sqrt(p_i * p_j) * (1 - d(i,j))
    K = np.zeros((n, n))
    for i, si in enumerate(strat_ids):
        for j, sj in enumerate(strat_ids):
            if si == sj:
                sim = 1.0
            else:
                sim = 1.0 - dist_matrix.get((si, sj), 1.0)
            K[i, j] = np.sqrt(freq_dict[si] * freq_dict[sj]) * sim

    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.clip(eigvals, 1e-12, None)
    eigvals = eigvals / eigvals.sum()  # normalize
    H = -np.sum(eigvals * np.log(eigvals + 1e-12))
    return np.exp(H)


def run_convergent_validity():
    """Correlation of Rao's Q (Gower) with Shannon, Vendi, answer diversity."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1c: CONVERGENT VALIDITY")
    print("=" * 70)

    rao_q_vals = []
    shannon_vals = []
    vendi_vals = []
    answer_div_vals = []

    for entry in multi_strat:
        pid = entry["problem_id"]
        strategies = {s["id"]: s for s in entry["llm_response"]["strategies"]}
        strat_traits = {sid: extract_strategy_traits(s) for sid, s in strategies.items()}

        # Build Gower distance matrix
        dist_matrix = {}
        for si, sj in combinations(strategies.keys(), 2):
            d = gower_distance(strat_traits[si], strat_traits[sj])
            dist_matrix[(si, sj)] = d
            dist_matrix[(sj, si)] = d

        for model in MODELS:
            freq = get_model_strategy_freq(entry, model)
            if not freq:
                continue

            q = rao_q_gower(entry, model)
            h = shannon_entropy(freq)
            v = vendi_score(freq, dist_matrix)

            # Number of distinct correct answers
            p_data = all_traces[model]["problems"][pid]
            unique_correct_answers = set()
            for r in p_data["rollouts"]:
                if r["is_correct"] and r["final_answer"]:
                    ans = r["final_answer"].strip().replace(" ", "").lower()
                    unique_correct_answers.add(ans)
            n_answers = len(unique_correct_answers)

            rao_q_vals.append(q)
            shannon_vals.append(h)
            vendi_vals.append(v)
            answer_div_vals.append(n_answers)

    # Spearman correlations
    print(f"\n  N data points: {len(rao_q_vals)} (model × problem pairs)")
    correlations = {}
    for name, vals in [("Shannon", shannon_vals), ("Vendi", vendi_vals), ("AnswerDiv", answer_div_vals)]:
        rho, pval = sp_stats.spearmanr(rao_q_vals, vals)
        correlations[name] = {"rho": round(rho, 4), "pval": round(pval, 6)}
        status = "GOOD" if abs(rho) > 0.7 else ("OK" if abs(rho) > 0.5 else "WEAK")
        print(f"  Rao's Q vs {name:<12}: rho={rho:.3f}  p={pval:.4f}  [{status}]")

    pass_count = sum(1 for v in correlations.values() if abs(v["rho"]) > 0.7)
    verdict = "PASS" if pass_count >= 2 else ("PARTIAL" if pass_count >= 1 else "FAIL")
    print(f"\n  Verdict: {verdict} — {pass_count}/3 correlations > 0.7")

    return {"correlations": correlations, "n_points": len(rao_q_vals), "verdict": verdict}


# ══════════════════════════════════════════════════════════
# EXPERIMENT 1d: Discriminant Validity
# ══════════════════════════════════════════════════════════

def run_discriminant_validity():
    """Rao's Q should NOT correlate with trace length, accuracy, difficulty."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1d: DISCRIMINANT VALIDITY")
    print("=" * 70)

    rao_q_vals = []
    trace_len_vals = []
    accuracy_vals = []
    difficulty_vals = []

    tier_to_num = {"easy": 1, "medium": 2, "hard": 3}

    for entry in multi_strat:
        pid = entry["problem_id"]
        prob = problems[pid]
        tier_num = tier_to_num.get(prob["tier"], 2)

        for model in MODELS:
            freq = get_model_strategy_freq(entry, model)
            if not freq:
                continue

            q = rao_q_gower(entry, model)

            # Mean trace length (tokens) for correct traces
            p_data = all_traces[model]["problems"][pid]
            correct_lens = [r["n_tokens"] for r in p_data["rollouts"] if r["is_correct"]]
            mean_len = np.mean(correct_lens) if correct_lens else 0

            # pass@1
            n_correct = p_data["n_correct"]
            acc = n_correct / 64.0

            rao_q_vals.append(q)
            trace_len_vals.append(mean_len)
            accuracy_vals.append(acc)
            difficulty_vals.append(tier_num)

    print(f"\n  N data points: {len(rao_q_vals)}")
    correlations = {}
    for name, vals in [("TraceLength", trace_len_vals), ("Accuracy", accuracy_vals), ("Difficulty", difficulty_vals)]:
        rho, pval = sp_stats.spearmanr(rao_q_vals, vals)
        correlations[name] = {"rho": round(rho, 4), "pval": round(pval, 6)}
        status = "GOOD (low)" if abs(rho) < 0.3 else ("CONCERN" if abs(rho) < 0.5 else "FAIL (high)")
        print(f"  Rao's Q vs {name:<12}: rho={rho:.3f}  p={pval:.4f}  [{status}]")

    pass_count = sum(1 for v in correlations.values() if abs(v["rho"]) < 0.3)
    verdict = "PASS" if pass_count == 3 else ("PARTIAL" if pass_count >= 2 else "FAIL")
    print(f"\n  Verdict: {verdict} — {pass_count}/3 correlations < 0.3")

    return {"correlations": correlations, "n_points": len(rao_q_vals), "verdict": verdict}


# ══════════════════════════════════════════════════════════
# EXPERIMENT 1e: Predictive Validity
# ══════════════════════════════════════════════════════════

def run_predictive_validity():
    """Does strategy diversity predict the pass@k scaling gap (pass@64 - pass@1)?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1e: PREDICTIVE VALIDITY")
    print("=" * 70)

    rao_q_vals = []
    scaling_gap_vals = []
    coverage_vals = []
    reliability_vals = []

    for entry in judge_data:  # ALL problems, not just multi-strat
        pid = entry["problem_id"]
        if "llm_response" not in entry:
            continue

        for model in MODELS:
            freq = get_model_strategy_freq(entry, model)
            q = rao_q_gower(entry, model) if freq and len(freq) > 1 else 0.0

            p_data = all_traces[model]["problems"][pid]
            n_correct = p_data["n_correct"]

            p1 = pass_at_k(64, n_correct, 1)
            p64 = pass_at_k(64, n_correct, 64)
            gap = p64 - p1

            # Coverage: 1 if solved at all (pass@64 > 0), 0 otherwise
            coverage = 1.0 if p64 > 0 else 0.0
            # Reliability: pass@1 given that problem is solvable
            reliability = p1 if p64 > 0 else np.nan

            rao_q_vals.append(q)
            scaling_gap_vals.append(gap)
            coverage_vals.append(coverage)
            if not np.isnan(reliability):
                reliability_vals.append((q, reliability))

    # Main test: diversity predicts scaling gap
    rho_gap, pval_gap = sp_stats.spearmanr(rao_q_vals, scaling_gap_vals)
    print(f"\n  N data points: {len(rao_q_vals)} (all 60 problems × 4 models)")
    print(f"  Rao's Q vs ScalingGap (p@64-p@1): rho={rho_gap:.3f}  p={pval_gap:.4f}")

    # Coverage decomposition
    rho_cov, pval_cov = sp_stats.spearmanr(rao_q_vals, coverage_vals)
    print(f"  Rao's Q vs Coverage (any solve):   rho={rho_cov:.3f}  p={pval_cov:.4f}")

    # Reliability (only for solvable problems)
    if reliability_vals:
        rel_q, rel_r = zip(*reliability_vals)
        rho_rel, pval_rel = sp_stats.spearmanr(rel_q, rel_r)
        print(f"  Rao's Q vs Reliability (p@1|solved): rho={rho_rel:.3f}  p={pval_rel:.4f}")
    else:
        rho_rel, pval_rel = 0, 1

    verdict = "PASS" if abs(rho_gap) > 0.3 else ("PARTIAL" if abs(rho_gap) > 0.15 else "FAIL")
    print(f"\n  Verdict: {verdict} — {'diversity predicts scaling headroom' if abs(rho_gap) > 0.3 else 'weak/no prediction'}")

    return {
        "scaling_gap": {"rho": round(rho_gap, 4), "pval": round(pval_gap, 6)},
        "coverage": {"rho": round(rho_cov, 4), "pval": round(pval_cov, 6)},
        "reliability": {"rho": round(rho_rel, 4), "pval": round(pval_rel, 6)},
        "n_points": len(rao_q_vals),
        "verdict": verdict,
    }


# ══════════════════════════════════════════════════════════
# EXPERIMENT 1f: Replication Principle
# ══════════════════════════════════════════════════════════

def run_replication_principle():
    """Pool traces from two models with different strategies. Pooled Q must exceed both individual Q's."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1f: REPLICATION PRINCIPLE")
    print("=" * 70)

    # Find problems where R1-Distill and v2 use different strategies
    test_cases = []
    for entry in multi_strat:
        pid = entry["problem_id"]
        strategies = {s["id"]: s for s in entry["llm_response"]["strategies"]}

        freq_r1 = get_model_strategy_freq(entry, "r1-distill")
        freq_v2 = get_model_strategy_freq(entry, "nemotron-v2")

        if not freq_r1 or not freq_v2:
            continue

        # Check if they use different dominant strategies
        top_r1 = max(freq_r1, key=freq_r1.get)
        top_v2 = max(freq_v2, key=freq_v2.get)

        if top_r1 != top_v2:
            test_cases.append((pid, entry, freq_r1, freq_v2))

    print(f"  Found {len(test_cases)} problems where R1 and v2 have different dominant strategies")

    violations = 0
    results_detail = []

    for pid, entry, freq_r1, freq_v2 in test_cases:
        strategies = {s["id"]: s for s in entry["llm_response"]["strategies"]}
        strat_traits = {sid: extract_strategy_traits(s) for sid, s in strategies.items()}

        # Compute individual Q's
        q_r1 = rao_q_gower(entry, "r1-distill")
        q_v2 = rao_q_gower(entry, "nemotron-v2")

        # Pool: average the frequency distributions
        all_strats = set(freq_r1.keys()) | set(freq_v2.keys())
        pooled_freq = {}
        for s in all_strats:
            pooled_freq[s] = (freq_r1.get(s, 0) + freq_v2.get(s, 0)) / 2.0

        # Compute pooled Q using Gower
        dist_matrix = {}
        for si, sj in combinations(strategies.keys(), 2):
            d = gower_distance(strat_traits.get(si, {}), strat_traits.get(sj, {}))
            dist_matrix[(si, sj)] = d
            dist_matrix[(sj, si)] = d

        q_pooled = rao_q_from_freq_and_dists(pooled_freq, dist_matrix)

        passed = q_pooled > max(q_r1, q_v2)
        if not passed:
            violations += 1

        status = "PASS" if passed else "VIOLATION"
        print(f"  P{pid}: Q_r1={q_r1:.4f}  Q_v2={q_v2:.4f}  Q_pooled={q_pooled:.4f}  [{status}]")
        results_detail.append({
            "problem_id": pid,
            "q_r1": round(q_r1, 6),
            "q_v2": round(q_v2, 6),
            "q_pooled": round(q_pooled, 6),
            "passed": passed,
        })

    if not test_cases:
        verdict = "SKIP"
        print("  No suitable test cases found")
    elif violations == 0:
        verdict = "PASS"
        print(f"\n  Verdict: PASS — Replication principle holds for all {len(test_cases)} test cases")
    else:
        verdict = "FAIL"
        print(f"\n  Verdict: FAIL — {violations}/{len(test_cases)} violations")

    return {"n_cases": len(test_cases), "violations": violations, "detail": results_detail, "verdict": verdict}


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("RAO'S Q INTERNAL VALIDATION (Experiments 1a–1f)")
    print("=" * 70)

    all_results = {}

    all_results["1a_rarefaction"] = run_rarefaction()
    all_results["1b_bootstrap_cis"] = run_bootstrap_cis()
    all_results["1c_convergent"] = run_convergent_validity()
    all_results["1d_discriminant"] = run_discriminant_validity()
    all_results["1e_predictive"] = run_predictive_validity()
    all_results["1f_replication"] = run_replication_principle()

    # ── Overall verdict ──

    print("\n" + "=" * 70)
    print("OVERALL VALIDATION SUMMARY")
    print("=" * 70)

    tests = [
        ("1a Rarefaction", all_results["1a_rarefaction"]["verdict"]),
        ("1b Bootstrap CIs", all_results["1b_bootstrap_cis"]["verdict"]),
        ("1c Convergent", all_results["1c_convergent"]["verdict"]),
        ("1d Discriminant", all_results["1d_discriminant"]["verdict"]),
        ("1e Predictive", all_results["1e_predictive"]["verdict"]),
        ("1f Replication", all_results["1f_replication"]["verdict"]),
    ]

    pass_count = sum(1 for _, v in tests if v == "PASS")
    partial_count = sum(1 for _, v in tests if v == "PARTIAL")
    fail_count = sum(1 for _, v in tests if v == "FAIL")

    for name, verdict in tests:
        marker = {"PASS": "+", "PARTIAL": "~", "FAIL": "-", "SKIP": "?"}[verdict]
        print(f"  [{marker}] {name}: {verdict}")

    print(f"\n  Score: {pass_count} PASS, {partial_count} PARTIAL, {fail_count} FAIL")

    if pass_count >= 5:
        print("  => Metric is ROBUST. Ready for paper.")
    elif pass_count >= 3:
        print("  => Metric WORKS but needs CAVEATS.")
    else:
        print("  => Metric is UNRELIABLE. Back to drawing board.")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUT_PATH}")
