#!/usr/bin/env python3
"""
Rao's Q Prototype: Strategy Diversity with Continuous Distance
==============================================================
Computes Rao's Q = Σ d(i,j) · p_i · p_j three ways:
  A. Jaccard distance on LLM-extracted strategy tags
  B. Method-description embedding distance (cosine)
  C. Gower distance on mixed traits

Uses Run 16 judge data (14 multi-strategy problems, 4 models).
Compares binary Gini-Simpson (d=1 for all pairs) vs continuous Rao's Q.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import Counter

# ── Load data ──

JUDGE_PATH = Path("data/analysis/llm_judge_pilot.json")
TRACES_DIR = Path("data/modal_runs/gen_traces_full")

with open(JUDGE_PATH) as f:
    judge_data = json.load(f)

MODELS = ["r1-distill", "nemotron-v1", "nemotron-v2", "nemotron-brorl"]
MODEL_PREFIX = {"r1-distill": "A", "nemotron-v1": "B", "nemotron-v2": "C", "nemotron-brorl": "D"}

# Load all traces
all_traces = {}
for model in MODELS:
    with open(TRACES_DIR / model / "traces.json") as f:
        all_traces[model] = json.load(f)

# ── Filter to multi-strategy problems ──

multi_strat = [j for j in judge_data if j["llm_response"]["n_strategies"] > 1]
print(f"Multi-strategy problems: {len(multi_strat)}")

# ── Helper: compute strategy frequencies per model ──

def get_model_strategy_freq(judge_entry, model):
    """Get strategy frequency distribution for a model on a problem."""
    prefix = MODEL_PREFIX[model]
    classifications = judge_entry["llm_response"]["classifications"]
    n_strategies = judge_entry["llm_response"]["n_strategies"]

    counts = Counter()
    total = 0
    for label, strat_id in classifications.items():
        if label.startswith(prefix):
            counts[strat_id] += 1
            total += 1

    if total == 0:
        return {}

    return {s: c / total for s, c in counts.items()}


# ══════════════════════════════════════════════════════════
# APPROACH 0: Binary Gini-Simpson (baseline, d=1 for all i≠j)
# ══════════════════════════════════════════════════════════

def gini_simpson(freq_dict):
    """Q with d(i,j)=1 for all i≠j. Reduces to 1 - Σ p_i^2."""
    if not freq_dict:
        return 0.0
    return 1.0 - sum(p**2 for p in freq_dict.values())


# ══════════════════════════════════════════════════════════
# APPROACH A: Jaccard distance on strategy descriptions
# ══════════════════════════════════════════════════════════

# Extract tags from strategy names/descriptions
MATH_TAGS = {
    # Proof techniques
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
    "long_division": ["long division", "division algorithm"],
    "power_of_ten": ["power of ten", "decimal", "denominator.*100", "denominator.*10"],
    "pattern": ["pattern", "sequence", "recurrence"],
    "estimation": ["estimat", "approximat", "bound"],
    # Additional coverage
    "optimization": ["optim", "minimize", "maximize", "lagrange", "am-gm", "inequality"],
    "inequality": ["inequalit", "am-gm", "cauchy", "schwarz", "jensen"],
    "polynomial": ["polynomial", "roots", "coefficients", "vieta"],
    "matrix": ["matrix", "determinant", "eigenvalue", "linear system"],
    "probability": ["probabilit", "expectation", "random", "expected value"],
    "logarithmic": ["logarithm", "log", "ln", "exponenti"],
    "complex_numbers": ["complex", "imaginar", "euler", "polar form"],
    "modular_arithmetic": ["mod ", "modulo", "remainder", "congruent"],
    "pigeonhole": ["pigeonhole", "pigeonhole principle"],
    "generating_function": ["generating function", "power series"],
    "parity": ["parity", "even", "odd", "divisib"],
    "bounding": ["bound", "upper bound", "lower bound", "sandwich"],
    "conversion": ["convert", "transform", "rewrite", "equivalent form"],
}

import re

def extract_tags(strategy_desc):
    """Extract mathematical tags from a strategy name + description."""
    text = strategy_desc.lower()
    tags = set()
    for tag, keywords in MATH_TAGS.items():
        for kw in keywords:
            if re.search(kw, text):
                tags.add(tag)
                break
    return tags


def jaccard_distance(tags_i, tags_j):
    """Jaccard distance: 1 - |intersection|/|union|."""
    if not tags_i and not tags_j:
        return 0.0
    union = tags_i | tags_j
    if not union:
        return 0.0
    return 1.0 - len(tags_i & tags_j) / len(union)


def rao_q_jaccard(judge_entry, model):
    """Compute Rao's Q using Jaccard distance on strategy tags."""
    freq = get_model_strategy_freq(judge_entry, model)
    strategies = {s["id"]: s for s in judge_entry["llm_response"]["strategies"]}

    if len(freq) <= 1:
        return 0.0

    # Extract tags for each strategy
    strat_tags = {}
    for sid, s in strategies.items():
        strat_tags[sid] = extract_tags(s["name"] + " " + s["description"])

    # Compute Rao's Q = Σ d(i,j) · p_i · p_j
    q = 0.0
    for si, pi in freq.items():
        for sj, pj in freq.items():
            if si != sj:
                d = jaccard_distance(strat_tags.get(si, set()), strat_tags.get(sj, set()))
                q += d * pi * pj
    return q


# ══════════════════════════════════════════════════════════
# APPROACH B: Method-description embeddings
# ══════════════════════════════════════════════════════════

def compute_embedding_distances(judge_entry):
    """Compute cosine distances between strategy method descriptions.
    Uses OpenAI text-embedding-3-small on METHOD DESCRIPTIONS, not full solutions.
    The key insight: embedding the method description (stripped of problem content)
    should have much better signal than embedding the full solution."""
    from openai import OpenAI

    env_path = Path("/Users/duy/Documents/build/eigen/seigen/.env.ecloud")
    # Read API key directly
    api_key = None
    with open(env_path) as ef:
        for line in ef:
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
                break
    client = OpenAI(api_key=api_key)

    strategies = judge_entry["llm_response"]["strategies"]
    if len(strategies) <= 1:
        return {}

    # Use ONLY the method name + description (not the problem text!)
    texts = []
    sids = []
    for s in strategies:
        texts.append(f"{s['name']}: {s['description']}")
        sids.append(s["id"])

    resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = np.array([d.embedding for d in resp.data])
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Compute pairwise cosine distances
    distances = {}
    for i in range(len(sids)):
        for j in range(len(sids)):
            if i != j:
                sim = np.dot(embeddings[i], embeddings[j])
                distances[(sids[i], sids[j])] = 1.0 - sim
    return distances


def rao_q_embedding(judge_entry, model, embed_dists):
    """Compute Rao's Q using embedding distances on method descriptions."""
    freq = get_model_strategy_freq(judge_entry, model)

    if len(freq) <= 1 or not embed_dists:
        return 0.0

    q = 0.0
    for si, pi in freq.items():
        for sj, pj in freq.items():
            if si != sj:
                d = embed_dists.get((si, sj), 1.0)
                q += d * pi * pj
    return q


# ══════════════════════════════════════════════════════════
# APPROACH C: Gower distance on mixed traits
# ══════════════════════════════════════════════════════════

# Define traits for each strategy
REPRESENTATION_TYPES = ["algebraic", "geometric", "numerical", "trigonometric", "combinatorial", "analytic"]
TECHNIQUE_TYPES = ["direct_computation", "substitution", "factoring", "formula_application",
                   "case_analysis", "algebraic_manipulation", "scaling", "pattern_recognition"]

def extract_strategy_traits(strategy):
    """Extract mixed traits from a strategy description."""
    text = (strategy["name"] + " " + strategy["description"]).lower()

    # Categorical: representation type
    rep = "algebraic"  # default
    for r in REPRESENTATION_TYPES:
        if r[:4] in text:
            rep = r
            break

    # Categorical: technique type
    tech = "direct_computation"  # default
    for t in TECHNIQUE_TYPES:
        short = t.split("_")[0][:4]
        if short in text:
            tech = t
            break

    # Continuous: description length (proxy for complexity)
    desc_len = len(strategy["description"].split())

    # Binary: uses known formula?
    uses_formula = 1 if any(w in text for w in ["formula", "theorem", "identity", "well-known"]) else 0

    # Binary: involves transformation?
    uses_transform = 1 if any(w in text for w in ["convert", "transform", "rewrite", "substitute"]) else 0

    return {
        "representation": rep,
        "technique": tech,
        "desc_length": desc_len,
        "uses_formula": uses_formula,
        "uses_transform": uses_transform,
    }


def gower_distance(traits_i, traits_j):
    """Gower's distance for mixed trait types."""
    dists = []

    # Categorical traits: 0 if same, 1 if different
    for cat_trait in ["representation", "technique"]:
        dists.append(0.0 if traits_i[cat_trait] == traits_j[cat_trait] else 1.0)

    # Continuous traits: |x-y| / range (use max_range = 50 words for desc_length)
    max_range = 50.0
    dists.append(min(abs(traits_i["desc_length"] - traits_j["desc_length"]) / max_range, 1.0))

    # Binary traits: 0 if same, 1 if different
    for bin_trait in ["uses_formula", "uses_transform"]:
        dists.append(0.0 if traits_i[bin_trait] == traits_j[bin_trait] else 1.0)

    return np.mean(dists)


def rao_q_gower(judge_entry, model):
    """Compute Rao's Q using Gower distance on strategy traits."""
    freq = get_model_strategy_freq(judge_entry, model)
    strategies = {s["id"]: s for s in judge_entry["llm_response"]["strategies"]}

    if len(freq) <= 1:
        return 0.0

    # Extract traits
    strat_traits = {sid: extract_strategy_traits(s) for sid, s in strategies.items()}

    q = 0.0
    for si, pi in freq.items():
        for sj, pj in freq.items():
            if si != sj:
                d = gower_distance(strat_traits.get(si, {}), strat_traits.get(sj, {}))
                q += d * pi * pj
    return q


# ══════════════════════════════════════════════════════════
# MAIN: Compute all metrics and compare
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("RAO'S Q PROTOTYPE: Strategy Diversity with Continuous Distance")
    print("=" * 80)

    # Per-model results across multi-strategy problems
    results = {model: {"gini_simpson": [], "jaccard": [], "embedding": [], "gower": []}
               for model in MODELS}

    # Also store per-problem distances for inspection
    problem_details = []

    for entry in multi_strat:
        pid = entry["problem_id"]
        strategies = entry["llm_response"]["strategies"]
        n_strat = len(strategies)

        # Pre-compute embedding distances for this problem (shared across models)
        embed_dists = compute_embedding_distances(entry)

        # Pre-compute tag distances for inspection
        strat_tags = {}
        for s in strategies:
            strat_tags[s["id"]] = extract_tags(s["name"] + " " + s["description"])

        # Pre-compute trait distances for inspection
        strat_traits = {}
        for s in strategies:
            strat_traits[s["id"]] = extract_strategy_traits(s)

        detail = {"problem_id": pid, "n_strategies": n_strat, "strategies": []}

        # Show strategy distance matrix
        print(f"\n{'─' * 80}")
        print(f"Problem {pid}: {n_strat} strategies")
        for s in strategies:
            tags = strat_tags[s["id"]]
            print(f"  S{s['id']}: {s['name']}")
            print(f"        Tags: {tags}")

        # Distance matrices
        print(f"\n  Distance matrices (Jaccard / Embedding / Gower):")
        for si, sj in combinations([s["id"] for s in strategies], 2):
            d_jac = jaccard_distance(strat_tags.get(si, set()), strat_tags.get(sj, set()))
            d_emb = embed_dists.get((si, sj), 1.0) if embed_dists else float('nan')
            d_gow = gower_distance(strat_traits.get(si, {}), strat_traits.get(sj, {}))
            print(f"    S{si}-S{sj}: Jaccard={d_jac:.3f}  Embed={d_emb:.3f}  Gower={d_gow:.3f}")
            detail["strategies"].append({
                "pair": f"S{si}-S{sj}",
                "jaccard": round(d_jac, 4),
                "embedding": round(d_emb, 4) if not np.isnan(d_emb) else None,
                "gower": round(d_gow, 4),
            })

        # Per-model Rao's Q
        print(f"\n  Per-model Rao's Q:")
        print(f"  {'Model':<20} {'GiniSimpson':>12} {'Jaccard':>12} {'Embedding':>12} {'Gower':>12}")
        for model in MODELS:
            gs = gini_simpson(get_model_strategy_freq(entry, model))
            jac = rao_q_jaccard(entry, model)
            emb = rao_q_embedding(entry, model, embed_dists)
            gow = rao_q_gower(entry, model)

            results[model]["gini_simpson"].append(gs)
            results[model]["jaccard"].append(jac)
            results[model]["embedding"].append(emb)
            results[model]["gower"].append(gow)

            print(f"  {model:<20} {gs:>12.4f} {jac:>12.4f} {emb:>12.4f} {gow:>12.4f}")

        problem_details.append(detail)

    # ── Summary across all multi-strategy problems ──

    print(f"\n{'=' * 80}")
    print("SUMMARY: Mean Rao's Q across 14 multi-strategy problems")
    print(f"{'=' * 80}")
    print(f"{'Model':<20} {'GiniSimpson':>12} {'Jaccard':>12} {'Embedding':>12} {'Gower':>12}")
    print("-" * 80)

    summary = {}
    for model in MODELS:
        gs = np.mean(results[model]["gini_simpson"])
        jac = np.mean(results[model]["jaccard"])
        emb = np.mean(results[model]["embedding"])
        gow = np.mean(results[model]["gower"])
        print(f"{model:<20} {gs:>12.4f} {jac:>12.4f} {emb:>12.4f} {gow:>12.4f}")
        summary[model] = {"gini_simpson": gs, "jaccard": jac, "embedding": emb, "gower": gow}

    # ── Ranking comparison ──

    print(f"\n{'=' * 80}")
    print("RANKING COMPARISON (highest diversity = rank 1)")
    print(f"{'=' * 80}")

    for metric in ["gini_simpson", "jaccard", "embedding", "gower"]:
        ranked = sorted(MODELS, key=lambda m: summary[m][metric], reverse=True)
        ranking_str = " > ".join(f"{m}({summary[m][metric]:.4f})" for m in ranked)
        print(f"  {metric:<15}: {ranking_str}")

    # ── Correlation between metrics ──

    print(f"\n{'=' * 80}")
    print("CORRELATION BETWEEN METRICS (Spearman, per-problem)")
    print(f"{'=' * 80}")

    from scipy import stats as sp_stats

    # Flatten: all (model, problem) pairs
    all_gs, all_jac, all_emb, all_gow = [], [], [], []
    for model in MODELS:
        all_gs.extend(results[model]["gini_simpson"])
        all_jac.extend(results[model]["jaccard"])
        all_emb.extend(results[model]["embedding"])
        all_gow.extend(results[model]["gower"])

    metrics_flat = {"GiniSimpson": all_gs, "Jaccard": all_jac, "Embedding": all_emb, "Gower": all_gow}
    metric_names = list(metrics_flat.keys())

    print(f"  {'':>15}", end="")
    for name in metric_names:
        print(f" {name:>12}", end="")
    print()

    for ni in metric_names:
        print(f"  {ni:>15}", end="")
        for nj in metric_names:
            if ni == nj:
                print(f" {'1.000':>12}", end="")
            else:
                rho, _ = sp_stats.spearmanr(metrics_flat[ni], metrics_flat[nj])
                print(f" {rho:>12.3f}", end="")
        print()

    # ── Key finding: Does continuous distance change the story? ──

    print(f"\n{'=' * 80}")
    print("VERDICT: Does continuous d(i,j) change the diversity ranking?")
    print(f"{'=' * 80}")

    gs_rank = sorted(MODELS, key=lambda m: summary[m]["gini_simpson"], reverse=True)
    jac_rank = sorted(MODELS, key=lambda m: summary[m]["jaccard"], reverse=True)
    emb_rank = sorted(MODELS, key=lambda m: summary[m]["embedding"], reverse=True)
    gow_rank = sorted(MODELS, key=lambda m: summary[m]["gower"], reverse=True)

    print(f"  Binary (Gini-Simpson) ranking: {' > '.join(gs_rank)}")
    print(f"  Jaccard ranking:               {' > '.join(jac_rank)}")
    print(f"  Embedding ranking:             {' > '.join(emb_rank)}")
    print(f"  Gower ranking:                 {' > '.join(gow_rank)}")

    agree = (gs_rank == jac_rank == emb_rank == gow_rank)
    print(f"\n  All metrics agree on ranking: {'YES' if agree else 'NO'}")

    if not agree:
        print("  → Rankings DIVERGE — the choice of distance metric matters!")
        print("    This is itself a finding: binary same/different misses real structure.")
    else:
        print("  → Rankings AGREE — continuous distance confirms the binary finding.")
        print("    The ranking is robust to choice of d(i,j).")

    # ── Distance distribution analysis ──

    print(f"\n{'=' * 80}")
    print("DISTANCE DISTRIBUTION: How different are strategies from each other?")
    print(f"{'=' * 80}")

    all_jac_dists = []
    all_emb_dists = []
    all_gow_dists = []

    for detail in problem_details:
        for pair in detail["strategies"]:
            all_jac_dists.append(pair["jaccard"])
            if pair["embedding"] is not None:
                all_emb_dists.append(pair["embedding"])
            all_gow_dists.append(pair["gower"])

    print(f"  {'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>5}")
    print(f"  {'-'*50}")
    for name, dists in [("Jaccard", all_jac_dists), ("Embedding", all_emb_dists), ("Gower", all_gow_dists)]:
        arr = np.array(dists)
        if len(arr) == 0:
            print(f"  {name:<12} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>5}")
            continue
        print(f"  {name:<12} {arr.mean():>8.3f} {arr.std():>8.3f} {arr.min():>8.3f} {arr.max():>8.3f} {len(arr):>5}")

    # ── Check: embedding distance on descriptions vs full solutions ──
    print(f"\n  Key question: Are method-description embeddings more discriminative")
    print(f"  than full-solution embeddings? (Previous MiniLM on full solutions showed")
    print(f"  0.92-0.96 cosine. OpenAI on descriptions should show wider spread.)")
    if all_emb_dists:
        print(f"  → Embedding distance range: [{min(all_emb_dists):.3f}, {max(all_emb_dists):.3f}]")
        if max(all_emb_dists) > 0.15:
            print(f"  → YES: Method descriptions produce meaningful distance variation!")
        else:
            print(f"  → NO: Even method descriptions are too similar.")
    else:
        print(f"  → No embedding distances computed.")

    # Save results
    def safe_stats(dists):
        if len(dists) == 0:
            return {"mean": None, "std": None}
        return {"mean": round(float(np.mean(dists)), 4), "std": round(float(np.std(dists)), 4)}

    output = {
        "summary": {m: {k: round(float(v), 6) for k, v in summary[m].items()} for m in MODELS},
        "problem_details": problem_details,
        "distance_stats": {
            "jaccard": safe_stats(all_jac_dists),
            "embedding": safe_stats(all_emb_dists),
            "gower": safe_stats(all_gow_dists),
        }
    }

    out_path = Path("data/analysis/rao_q_prototype.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")
