#!/usr/bin/env python3
"""
Phase 2: Strategy Classification Validation + Cross-Model Analysis
==================================================================
Analyzes 15,360 traces from Phase 1 to determine:
1. Whether embedding clusters correspond to genuine strategy differences
2. How strategy diversity evolves across the ProRL training trajectory
3. Whether the pattern varies by problem difficulty

Input:  data/modal_runs/gen_traces_full/
Output: data/analysis/phase2_results.json + printed report

Usage:
  python3 pilot/analyze_strategies.py
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/modal_runs/gen_traces_full")
OUT_DIR = Path("data/analysis")
MODELS = ["r1-distill", "nemotron-v1", "nemotron-v2", "nemotron-brorl"]
MODEL_LABELS = {
    "r1-distill": "R1-Distill",
    "nemotron-v1": "v1 (2K)",
    "nemotron-v2": "v2 (3K)",
    "nemotron-brorl": "BroRL",
}


# ── Data Loading ────────────────────────────────────────────────


def load_all():
    """Load problems, traces, and embeddings for all models."""
    with open(DATA_DIR / "problems.json") as f:
        problems = json.load(f)

    traces = {}
    embeddings = {}
    for model in MODELS:
        mdir = DATA_DIR / model
        with open(mdir / "traces.json") as f:
            traces[model] = json.load(f)
        embeddings[model] = np.load(mdir / "embeddings.npy")

    K = traces[MODELS[0]]["K"]
    n_problems = len(problems)

    print(f"Loaded: {n_problems} problems × {len(MODELS)} models × K={K}")
    print(f"Embedding shape per model: {embeddings[MODELS[0]].shape}")
    return problems, traces, embeddings, K


def get_problem_embeddings(embeddings, problem_idx, K):
    """Get (K, 384) embeddings for one problem from one model."""
    start = problem_idx * K
    return embeddings[start : start + K]


def get_problem_rollouts(traces, problem_idx):
    """Get rollout metadata for one problem from one model's traces."""
    return traces["problems"][problem_idx]["rollouts"]


# ── Clustering ──────────────────────────────────────────────────


def cluster_problem(all_embs, max_k=10):
    """Cluster embeddings for one problem using agglomerative clustering.

    Args:
        all_embs: (N, 384) array — all models' embeddings stacked
        max_k: maximum number of clusters to try

    Returns:
        dict with labels, n_clusters, silhouette_score
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    N = all_embs.shape[0]
    if N < 4:
        return {"labels": np.zeros(N, dtype=int), "n_clusters": 1, "silhouette": 0.0}

    # Cosine distance matrix
    # Embeddings are already L2-normalized, so cosine_dist = 1 - dot product
    sim = all_embs @ all_embs.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, 2)  # numerical safety

    best_score = -1
    best_labels = None
    best_k = 2

    upper_k = min(max_k, N // 2)
    for k in range(2, upper_k + 1):
        clust = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )
        labels = clust.fit_predict(dist)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(dist, labels, metric="precomputed")
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    if best_labels is None:
        best_labels = np.zeros(N, dtype=int)
        best_k = 1
        best_score = 0.0

    return {
        "labels": best_labels,
        "n_clusters": best_k,
        "silhouette": round(float(best_score), 4),
    }


# ── Validation Metrics ──────────────────────────────────────────


def cluster_answer_alignment(labels, answers):
    """Measure how well clusters align with final answers.

    Returns:
        answer_purity: fraction of within-cluster pairs sharing same answer
        answer_separation: fraction of between-cluster pairs with different answers
    """
    n = len(labels)
    within_same = 0
    within_total = 0
    between_diff = 0
    between_total = 0

    for i in range(n):
        for j in range(i + 1, n):
            a_i = answers[i] or ""
            a_j = answers[j] or ""
            same_cluster = labels[i] == labels[j]
            same_answer = a_i == a_j and a_i != ""

            if same_cluster:
                within_total += 1
                if same_answer:
                    within_same += 1
            else:
                between_total += 1
                if not same_answer:
                    between_diff += 1

    purity = within_same / max(within_total, 1)
    separation = between_diff / max(between_total, 1)
    return round(purity, 4), round(separation, 4)


def cluster_correctness_alignment(labels, correct_flags):
    """Measure whether clusters separate correct from incorrect traces."""
    clusters = defaultdict(list)
    for i, lab in enumerate(labels):
        clusters[lab].append(correct_flags[i])

    # For each cluster: what fraction is correct?
    cluster_purities = []
    for lab, flags in clusters.items():
        rate = sum(flags) / len(flags)
        cluster_purities.append(max(rate, 1 - rate))  # majority purity

    return round(float(np.mean(cluster_purities)), 4)


# ── Strategy Diversity Metrics ──────────────────────────────────


def shannon_entropy(counts):
    """Shannon entropy of a distribution (from counts)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log(p) for p in probs)


def model_strategy_profile(labels, model_start, K):
    """Get cluster distribution for one model within a cross-model clustering.

    Args:
        labels: cluster labels for all 4 models stacked (4*K,)
        model_start: start index for this model
        K: rollouts per model

    Returns:
        dict mapping cluster_id -> count
    """
    model_labels = labels[model_start : model_start + K]
    return dict(Counter(model_labels))


def strategy_diversity_metrics(profile, K):
    """Compute diversity metrics from a model's cluster profile."""
    counts = list(profile.values())
    n_strategies = len(counts)
    entropy = shannon_entropy(counts)
    max_entropy = math.log(K) if K > 0 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Simpson's diversity
    total = sum(counts)
    if total > 1:
        simpson = 1 - sum(c * (c - 1) for c in counts) / (total * (total - 1))
    else:
        simpson = 0.0

    # Dominant strategy share
    dominant = max(counts) / total if total > 0 else 1.0

    return {
        "n_strategies": n_strategies,
        "entropy": round(entropy, 4),
        "norm_entropy": round(normalized_entropy, 4),
        "simpson": round(simpson, 4),
        "dominant_share": round(dominant, 4),
    }


def novel_strategies(profiles, model_names):
    """Find strategies unique to each model (present in X, absent in all others)."""
    all_clusters = set()
    for p in profiles.values():
        all_clusters.update(p.keys())

    novel = {}
    for name in model_names:
        mine = set(profiles[name].keys())
        others = set()
        for other_name in model_names:
            if other_name != name:
                others.update(profiles[other_name].keys())
        unique = mine - others
        novel[name] = {
            "unique_clusters": list(unique),
            "n_unique": len(unique),
            "unique_rollouts": sum(profiles[name].get(c, 0) for c in unique),
        }
    return novel


# ── Main Analysis ───────────────────────────────────────────────


def analyze():
    problems, traces, embeddings, K = load_all()
    n_problems = len(problems)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("PHASE 2: STRATEGY CLASSIFICATION + CROSS-MODEL ANALYSIS")
    print(f"{'='*70}")

    # ── Per-problem analysis ────────────────────────────────────
    problem_results = []
    tier_agg = defaultdict(lambda: defaultdict(list))  # tier -> model -> [metrics]

    for pid in range(n_problems):
        prob = problems[pid]
        tier = prob["tier"]

        # Stack embeddings from all models: (4*K, 384)
        embs_list = []
        answers_list = []
        correct_list = []
        model_starts = {}

        for mi, model in enumerate(MODELS):
            emb = get_problem_embeddings(embeddings[model], pid, K)
            rollouts = get_problem_rollouts(traces[model], pid)
            embs_list.append(emb)
            model_starts[model] = mi * K

            for r in rollouts:
                ans = r.get("final_answer") or ""
                answers_list.append(ans.strip().replace(" ", "").lower())
                correct_list.append(r["is_correct"])

        all_embs = np.vstack(embs_list)  # (4*K, 384)

        # Cluster
        clust = cluster_problem(all_embs, max_k=12)
        labels = clust["labels"]

        # Validation: cluster-answer alignment
        purity, separation = cluster_answer_alignment(labels, answers_list)
        correctness_purity = cluster_correctness_alignment(labels, correct_list)

        # Per-model strategy profiles
        profiles = {}
        model_metrics = {}
        for model in MODELS:
            profile = model_strategy_profile(labels, model_starts[model], K)
            profiles[model] = profile
            metrics = strategy_diversity_metrics(profile, K)
            model_metrics[model] = metrics
            tier_agg[tier][model].append(metrics)

        # Novel strategies
        novel = novel_strategies(profiles, MODELS)

        result = {
            "problem_id": pid,
            "problem_text": prob["problem"][:120],
            "tier": tier,
            "subject": prob["subject"],
            "n_clusters": clust["n_clusters"],
            "silhouette": clust["silhouette"],
            "answer_purity": purity,
            "answer_separation": separation,
            "correctness_purity": correctness_purity,
            "model_metrics": model_metrics,
            "novel_strategies": novel,
        }
        problem_results.append(result)

        if pid % 15 == 0:
            print(f"\n  Problem {pid} [{tier}/{prob['subject']}]: "
                  f"{clust['n_clusters']} clusters (sil={clust['silhouette']:.2f}), "
                  f"purity={purity:.2f}, sep={separation:.2f}")
            for m in MODELS:
                mm = model_metrics[m]
                print(f"    {MODEL_LABELS[m]:>12}: {mm['n_strategies']} strategies, "
                      f"entropy={mm['entropy']:.2f}, dominant={mm['dominant_share']:.2f}")

    # ── Aggregate Results ───────────────────────────────────────

    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    # Overall validation
    all_sil = [r["silhouette"] for r in problem_results]
    all_purity = [r["answer_purity"] for r in problem_results]
    all_sep = [r["answer_separation"] for r in problem_results]
    all_corr_pur = [r["correctness_purity"] for r in problem_results]
    all_n_clust = [r["n_clusters"] for r in problem_results]

    print(f"\n  Clustering Quality (across {n_problems} problems):")
    print(f"    Mean clusters:      {np.mean(all_n_clust):.1f} (range {min(all_n_clust)}-{max(all_n_clust)})")
    print(f"    Mean silhouette:    {np.mean(all_sil):.3f}")
    print(f"    Mean answer purity: {np.mean(all_purity):.3f}")
    print(f"    Mean answer sep:    {np.mean(all_sep):.3f}")
    print(f"    Mean correct pur:   {np.mean(all_corr_pur):.3f}")

    # Validation verdict
    mean_purity = np.mean(all_purity)
    mean_sep = np.mean(all_sep)
    print(f"\n  VALIDATION GATE:")
    print(f"    Answer purity > 0.5?  {mean_purity:.3f} {'PASS' if mean_purity > 0.5 else 'FAIL'}")
    print(f"    Answer separation > 0.4? {mean_sep:.3f} {'PASS' if mean_sep > 0.4 else 'FAIL'}")

    # Cross-model strategy diversity
    print(f"\n  {'Model':<14} {'strategies':>10} {'entropy':>10} {'norm_ent':>10} {'simpson':>10} {'dominant':>10}")
    print(f"  {'-'*64}")

    model_agg = {}
    for model in MODELS:
        all_metrics = []
        for pid in range(n_problems):
            all_metrics.append(problem_results[pid]["model_metrics"][model])

        agg = {
            "mean_strategies": round(np.mean([m["n_strategies"] for m in all_metrics]), 2),
            "mean_entropy": round(np.mean([m["entropy"] for m in all_metrics]), 4),
            "mean_norm_entropy": round(np.mean([m["norm_entropy"] for m in all_metrics]), 4),
            "mean_simpson": round(np.mean([m["simpson"] for m in all_metrics]), 4),
            "mean_dominant_share": round(np.mean([m["dominant_share"] for m in all_metrics]), 4),
        }
        model_agg[model] = agg

        print(f"  {MODEL_LABELS[model]:<14} {agg['mean_strategies']:>10.1f} "
              f"{agg['mean_entropy']:>10.3f} {agg['mean_norm_entropy']:>10.3f} "
              f"{agg['mean_simpson']:>10.3f} {agg['mean_dominant_share']:>10.3f}")

    # Per-tier breakdown
    print(f"\n  Per-Difficulty Strategy Counts (mean strategies per problem):")
    print(f"  {'Tier':<10}", end="")
    for m in MODELS:
        print(f" {MODEL_LABELS[m]:>12}", end="")
    print()
    print(f"  {'-'*58}")

    tier_summary = {}
    for tier in ["easy", "medium", "hard"]:
        print(f"  {tier:<10}", end="")
        tier_summary[tier] = {}
        for model in MODELS:
            metrics_list = tier_agg[tier][model]
            if metrics_list:
                mean_strat = np.mean([m["n_strategies"] for m in metrics_list])
                mean_ent = np.mean([m["entropy"] for m in metrics_list])
                mean_dom = np.mean([m["dominant_share"] for m in metrics_list])
                tier_summary[tier][model] = {
                    "mean_strategies": round(float(mean_strat), 2),
                    "mean_entropy": round(float(mean_ent), 4),
                    "mean_dominant": round(float(mean_dom), 4),
                }
                print(f" {mean_strat:>12.1f}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()

    # Per-tier entropy
    print(f"\n  Per-Difficulty Entropy:")
    print(f"  {'Tier':<10}", end="")
    for m in MODELS:
        print(f" {MODEL_LABELS[m]:>12}", end="")
    print()
    print(f"  {'-'*58}")
    for tier in ["easy", "medium", "hard"]:
        print(f"  {tier:<10}", end="")
        for model in MODELS:
            ts = tier_summary.get(tier, {}).get(model, {})
            ent = ts.get("mean_entropy", 0)
            print(f" {ent:>12.3f}", end="")
        print()

    # Novel strategy analysis
    print(f"\n  Novel Strategies (clusters unique to one model, mean per problem):")
    novel_agg = defaultdict(list)
    for pid in range(n_problems):
        for model in MODELS:
            novel_agg[model].append(
                problem_results[pid]["novel_strategies"][model]["n_unique"]
            )
    for model in MODELS:
        mean_novel = np.mean(novel_agg[model])
        total_novel = sum(novel_agg[model])
        print(f"    {MODEL_LABELS[model]:>12}: {mean_novel:.2f} per problem ({total_novel} total)")

    # ── Detailed inspection: top-separating problems ────────────

    print(f"\n{'='*70}")
    print("TOP 5 PROBLEMS WITH HIGHEST STRATEGY DIVERGENCE ACROSS MODELS")
    print(f"{'='*70}")

    # Score each problem by variance in strategy count across models
    divergence_scores = []
    for pid in range(n_problems):
        counts = [problem_results[pid]["model_metrics"][m]["n_strategies"] for m in MODELS]
        divergence_scores.append((np.std(counts), pid))
    divergence_scores.sort(reverse=True)

    for rank, (score, pid) in enumerate(divergence_scores[:5]):
        r = problem_results[pid]
        print(f"\n  #{rank+1} Problem {pid} [{r['tier']}/{r['subject']}] "
              f"(divergence={score:.2f}, clusters={r['n_clusters']})")
        print(f"    {r['problem_text']}...")
        for m in MODELS:
            mm = r["model_metrics"][m]
            nov = r["novel_strategies"][m]
            print(f"    {MODEL_LABELS[m]:>12}: {mm['n_strategies']} strat, "
                  f"entropy={mm['entropy']:.2f}, novel={nov['n_unique']}")

    # ── Correct-only analysis ───────────────────────────────────

    print(f"\n{'='*70}")
    print("CORRECT-ONLY STRATEGY ANALYSIS")
    print(f"{'='*70}")
    print("  (Clustering only correct traces to measure strategy diversity in solutions)")

    correct_tier_strats = defaultdict(lambda: defaultdict(list))

    for pid in range(n_problems):
        prob = problems[pid]
        tier = prob["tier"]

        # Collect correct-only embeddings per model
        correct_embs = []
        correct_model_ids = []
        model_correct_counts = {}

        for mi, model in enumerate(MODELS):
            emb = get_problem_embeddings(embeddings[model], pid, K)
            rollouts = get_problem_rollouts(traces[model], pid)
            count = 0
            for ri, r in enumerate(rollouts):
                if r["is_correct"]:
                    correct_embs.append(emb[ri])
                    correct_model_ids.append(mi)
                    count += 1
            model_correct_counts[model] = count

        if len(correct_embs) < 4:
            # Not enough correct traces to cluster
            for model in MODELS:
                correct_tier_strats[tier][model].append(0)
            continue

        correct_embs = np.array(correct_embs)
        clust = cluster_problem(correct_embs, max_k=8)
        labels = clust["labels"]

        # Per-model strategy count among correct traces
        for mi, model in enumerate(MODELS):
            model_mask = [i for i, mid in enumerate(correct_model_ids) if mid == mi]
            if model_mask:
                model_labels = labels[model_mask]
                n_strat = len(set(model_labels))
            else:
                n_strat = 0
            correct_tier_strats[tier][model].append(n_strat)

    print(f"\n  Mean strategy count (correct traces only):")
    print(f"  {'Tier':<10}", end="")
    for m in MODELS:
        print(f" {MODEL_LABELS[m]:>12}", end="")
    print()
    print(f"  {'-'*58}")

    correct_summary = {}
    for tier in ["easy", "medium", "hard"]:
        print(f"  {tier:<10}", end="")
        correct_summary[tier] = {}
        for model in MODELS:
            vals = correct_tier_strats[tier][model]
            mean_s = np.mean(vals) if vals else 0
            correct_summary[tier][model] = round(float(mean_s), 2)
            print(f" {mean_s:>12.1f}", end="")
        print()

    # ── Save results ────────────────────────────────────────────

    output = {
        "validation": {
            "mean_clusters": round(float(np.mean(all_n_clust)), 2),
            "mean_silhouette": round(float(np.mean(all_sil)), 4),
            "mean_answer_purity": round(float(np.mean(all_purity)), 4),
            "mean_answer_separation": round(float(np.mean(all_sep)), 4),
            "mean_correctness_purity": round(float(np.mean(all_corr_pur)), 4),
            "gate_purity_pass": bool(mean_purity > 0.5),
            "gate_separation_pass": bool(mean_sep > 0.4),
        },
        "model_aggregate": model_agg,
        "tier_summary": tier_summary,
        "correct_only_tier": correct_summary,
        "novel_strategies": {
            m: {
                "mean_per_problem": round(float(np.mean(novel_agg[m])), 2),
                "total": int(sum(novel_agg[m])),
            }
            for m in MODELS
        },
        "problems": problem_results,
    }

    out_path = OUT_DIR / "phase2_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return output


if __name__ == "__main__":
    analyze()
