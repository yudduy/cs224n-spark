#!/usr/bin/env python3
"""
Rigorous statistical analysis of Phase 1 pass@k data.
Bootstrap CIs, significance tests, sampling efficiency, coverage analysis.

Usage: python3 pilot/analyze_rigorous.py
"""

import json
import numpy as np
from math import comb
from pathlib import Path
from scipy import stats

DATA_DIR = Path("data/modal_runs/gen_traces_full")
OUT_DIR = Path("data/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["r1-distill", "nemotron-v1", "nemotron-v2", "nemotron-brorl"]
LABELS = ["R1-Distill", "v1 (2K)", "v2 (3K)", "BroRL"]
K = 64
N_BOOTSTRAP = 10000
np.random.seed(42)


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def bootstrap_pass_at_k(corrects, n, k, n_boot=N_BOOTSTRAP):
    """Bootstrap CI for mean pass@k across problems."""
    n_probs = len(corrects)
    pak = np.array([pass_at_k(n, c, k) for c in corrects])
    boots = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.random.randint(0, n_probs, n_probs)
        boots[b] = pak[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return pak.mean(), lo, hi


def load_data():
    with open(DATA_DIR / "problems.json") as f:
        problems = json.load(f)
    all_data = {}
    for m in MODELS:
        with open(DATA_DIR / m / "traces.json") as f:
            all_data[m] = json.load(f)
    return problems, all_data


def main():
    problems, all_data = load_data()
    results = {}

    # ── 1. Bootstrap CIs on pass@k ──────────────────────────────
    print("=" * 80)
    print("1. PASS@K WITH 95% BOOTSTRAP CIs (10K resamples)")
    print("=" * 80)
    print()

    ks = [1, 2, 4, 8, 16, 32, 64]
    ci_data = {}

    for tier in ["easy", "medium", "hard", "all"]:
        pids = list(range(60)) if tier == "all" else [
            i for i in range(60) if problems[i]["tier"] == tier
        ]
        print(f"  {tier.upper()} ({len(pids)} problems):")
        ci_data[tier] = {}

        for mi, m in enumerate(MODELS):
            corrects = [all_data[m]["problems"][p]["n_correct"] for p in pids]
            row = f"    {LABELS[mi]:12s}:"
            ci_data[tier][m] = {}
            for k in ks:
                mean, lo, hi = bootstrap_pass_at_k(corrects, K, k)
                ci_data[tier][m][str(k)] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
                row += f"  {mean:.3f}[{lo:.3f},{hi:.3f}]"
            print(row)
        print()

    results["pass_at_k_ci"] = ci_data

    # ── 2. v1 Anomaly: Paired test ─────────────────────────────
    print("=" * 80)
    print("2. v1 ANOMALY: Is v1 significantly worse than R1-Distill at k≥8?")
    print("=" * 80)
    print()

    anomaly_results = {}
    for tier in ["medium", "hard"]:
        pids = [i for i in range(60) if problems[i]["tier"] == tier]
        print(f"  {tier.upper()}:")

        for k in [8, 16, 32, 64]:
            r1_paks = [pass_at_k(K, all_data["r1-distill"]["problems"][p]["n_correct"], k) for p in pids]
            v1_paks = [pass_at_k(K, all_data["nemotron-v1"]["problems"][p]["n_correct"], k) for p in pids]

            diffs = [r - v for r, v in zip(r1_paks, v1_paks)]
            mean_diff = np.mean(diffs)

            # Wilcoxon signed-rank (paired, non-parametric)
            # Only on non-zero diffs
            nonzero = [d for d in diffs if d != 0]
            if len(nonzero) >= 5:
                stat, p_val = stats.wilcoxon(nonzero, alternative="greater")
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            else:
                p_val = float("nan")
                sig = "n/a"

            key = f"{tier}_k{k}"
            anomaly_results[key] = {
                "mean_diff": mean_diff,
                "p_value": p_val,
                "significant": sig,
                "n_r1_better": sum(1 for d in diffs if d > 0),
                "n_v1_better": sum(1 for d in diffs if d < 0),
                "n_tied": sum(1 for d in diffs if d == 0),
            }
            print(f"    pass@{k:2d}: R1-v1 diff = {mean_diff:+.4f}, p={p_val:.4f} {sig}")
            print(f"             R1 better: {anomaly_results[key]['n_r1_better']}, v1 better: {anomaly_results[key]['n_v1_better']}, tied: {anomaly_results[key]['n_tied']}")
        print()

    results["v1_anomaly"] = anomaly_results

    # ── 3. v2 vs R1: Coverage expansion significance ────────────
    print("=" * 80)
    print("3. v2 vs R1-DISTILL: Paired comparison")
    print("=" * 80)
    print()

    v2_vs_r1 = {}
    for tier in ["easy", "medium", "hard"]:
        pids = [i for i in range(60) if problems[i]["tier"] == tier]
        print(f"  {tier.upper()}:")

        for k in [1, 8, 64]:
            r1_paks = [pass_at_k(K, all_data["r1-distill"]["problems"][p]["n_correct"], k) for p in pids]
            v2_paks = [pass_at_k(K, all_data["nemotron-v2"]["problems"][p]["n_correct"], k) for p in pids]

            diffs = [v - r for r, v in zip(r1_paks, v2_paks)]
            mean_diff = np.mean(diffs)

            nonzero = [d for d in diffs if d != 0]
            if len(nonzero) >= 5:
                stat, p_val = stats.wilcoxon(nonzero, alternative="greater")
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            else:
                p_val = float("nan")
                sig = "n/a"

            # Effect size: matched-pairs rank-biserial
            key = f"{tier}_k{k}"
            v2_vs_r1[key] = {
                "mean_diff": mean_diff,
                "p_value": p_val,
                "significant": sig,
                "n_v2_better": sum(1 for d in diffs if d > 0),
                "n_r1_better": sum(1 for d in diffs if d < 0),
            }
            print(f"    pass@{k:2d}: v2-R1 diff = {mean_diff:+.4f}, p={p_val:.4f} {sig}, v2 better on {v2_vs_r1[key]['n_v2_better']}/{len(pids)}")
        print()

    results["v2_vs_r1"] = v2_vs_r1

    # ── 4. Sampling efficiency ──────────────────────────────────
    print("=" * 80)
    print("4. SAMPLING EFFICIENCY: How many samples of model X to match model Y at pass@1?")
    print("=" * 80)
    print()

    efficiency = {}
    ks_fine = list(range(1, 65))

    for tier in ["easy", "medium", "hard"]:
        pids = [i for i in range(60) if problems[i]["tier"] == tier]
        print(f"  {tier.upper()}:")

        # For each target model's pass@1, find how many samples the other needs
        for target_m, target_l in zip(MODELS, LABELS):
            target_p1 = np.mean([
                pass_at_k(K, all_data[target_m]["problems"][p]["n_correct"], 1)
                for p in pids
            ])

            for source_m, source_l in zip(MODELS, LABELS):
                if source_m == target_m:
                    continue
                for k in ks_fine:
                    source_pk = np.mean([
                        pass_at_k(K, all_data[source_m]["problems"][p]["n_correct"], k)
                        for p in pids
                    ])
                    if source_pk >= target_p1:
                        key = f"{tier}_{source_m}_to_match_{target_m}"
                        efficiency[key] = {"k_needed": k, "source_pk": source_pk, "target_p1": target_p1}
                        break

        # Print key comparisons
        for key_pair in [("r1-distill", "nemotron-v2"), ("nemotron-v1", "nemotron-v2")]:
            src, tgt = key_pair
            ekey = f"{tier}_{src}_to_match_{tgt}"
            if ekey in efficiency:
                e = efficiency[ekey]
                src_l = LABELS[MODELS.index(src)]
                tgt_l = LABELS[MODELS.index(tgt)]
                print(f"    {src_l}@{e['k_needed']} = {e['source_pk']:.3f} >= {tgt_l}@1 = {e['target_p1']:.3f}")
            else:
                src_l = LABELS[MODELS.index(src)]
                tgt_l = LABELS[MODELS.index(tgt)]
                print(f"    {src_l}@64 < {tgt_l}@1 (can't match!)")
        print()

    results["sampling_efficiency"] = efficiency

    # ── 5. Coverage ceiling (Venn-like) ─────────────────────────
    print("=" * 80)
    print("5. COVERAGE ANALYSIS: Which problems does each model solve?")
    print("=" * 80)
    print()

    coverage = {}
    for tier in ["easy", "medium", "hard"]:
        pids = [i for i in range(60) if problems[i]["tier"] == tier]
        solved = {}
        for m in MODELS:
            solved[m] = set(p for p in pids if all_data[m]["problems"][p]["n_correct"] > 0)

        # Intersection/difference analysis
        r1 = solved["r1-distill"]
        v2 = solved["nemotron-v2"]
        both = r1 & v2
        r1_only = r1 - v2
        v2_only = v2 - r1
        neither = set(pids) - r1 - v2

        coverage[tier] = {
            "r1_solves": len(r1),
            "v2_solves": len(v2),
            "both_solve": len(both),
            "r1_only": sorted(r1_only),
            "v2_only": sorted(v2_only),
            "neither": sorted(neither),
        }

        print(f"  {tier.upper()} ({len(pids)} problems):")
        print(f"    R1 solves: {len(r1)}, v2 solves: {len(v2)}")
        print(f"    Both: {len(both)}, R1-only: {sorted(r1_only)}, v2-only: {sorted(v2_only)}")
        print(f"    Neither: {sorted(neither)}")

        # For v2-only problems: how many correct does v2 get?
        if v2_only:
            for p in sorted(v2_only):
                nc = all_data["nemotron-v2"]["problems"][p]["n_correct"]
                print(f"      Problem {p} [{problems[p]['subject']}]: v2 gets {nc}/64 correct")
        # For r1-only problems
        if r1_only:
            for p in sorted(r1_only):
                nc = all_data["r1-distill"]["problems"][p]["n_correct"]
                print(f"      Problem {p} [{problems[p]['subject']}]: R1 gets {nc}/64 correct")
        print()

    results["coverage"] = coverage

    # ── 6. Per-problem n_correct correlation ────────────────────
    print("=" * 80)
    print("6. PER-PROBLEM ACCURACY CORRELATION (Spearman)")
    print("=" * 80)
    print()

    for mi, m1 in enumerate(MODELS):
        for mj, m2 in enumerate(MODELS):
            if mj <= mi:
                continue
            c1 = [all_data[m1]["problems"][p]["n_correct"] for p in range(60)]
            c2 = [all_data[m2]["problems"][p]["n_correct"] for p in range(60)]
            rho, p_val = stats.spearmanr(c1, c2)
            print(f"  {LABELS[mi]:12s} vs {LABELS[mj]:12s}: rho={rho:.3f}, p={p_val:.2e}")

    # Also per tier
    print()
    for tier in ["easy", "medium", "hard"]:
        pids = [i for i in range(60) if problems[i]["tier"] == tier]
        c1 = [all_data["r1-distill"]["problems"][p]["n_correct"] for p in pids]
        c2 = [all_data["nemotron-v2"]["problems"][p]["n_correct"] for p in pids]
        rho, p_val = stats.spearmanr(c1, c2)
        print(f"  R1 vs v2 [{tier:6s}]: rho={rho:.3f}, p={p_val:.2e}")

    # ── 7. Phase transition: v1→v2 gain ─────────────────────────
    print()
    print("=" * 80)
    print("7. PHASE TRANSITION: Per-problem improvement v1→v2")
    print("=" * 80)
    print()

    for tier in ["easy", "medium", "hard"]:
        pids = [i for i in range(60) if problems[i]["tier"] == tier]
        v1_c = [all_data["nemotron-v1"]["problems"][p]["n_correct"] for p in pids]
        v2_c = [all_data["nemotron-v2"]["problems"][p]["n_correct"] for p in pids]
        diffs = [v2_c[i] - v1_c[i] for i in range(len(pids))]
        print(f"  {tier.upper()}: v2-v1 mean diff = {np.mean(diffs):+.1f}/64 correct")
        print(f"    v2 better: {sum(1 for d in diffs if d > 0)}, v1 better: {sum(1 for d in diffs if d < 0)}, tied: {sum(1 for d in diffs if d == 0)}")
        # Problems with biggest v2 improvement
        sorted_by_gain = sorted(zip(pids, diffs), key=lambda x: -x[1])
        print(f"    Biggest v2 gains: ", end="")
        for p, d in sorted_by_gain[:3]:
            print(f"P{p}(+{d})", end=" ")
        print()
        print(f"    v1 holds:          ", end="")
        for p, d in sorted_by_gain[-3:]:
            if d < 0:
                print(f"P{p}({d})", end=" ")
        print()
        print()

    # ── 8. BroRL vs v2 ─────────────────────────────────────────
    print("=" * 80)
    print("8. BroRL vs v2: Does breadth scaling help on MATH?")
    print("=" * 80)
    print()

    for tier in ["easy", "medium", "hard"]:
        pids = [i for i in range(60) if problems[i]["tier"] == tier]
        for k in [1, 8, 64]:
            v2_paks = [pass_at_k(K, all_data["nemotron-v2"]["problems"][p]["n_correct"], k) for p in pids]
            br_paks = [pass_at_k(K, all_data["nemotron-brorl"]["problems"][p]["n_correct"], k) for p in pids]
            diffs = [v - b for v, b in zip(v2_paks, br_paks)]
            print(f"  {tier:6s} pass@{k:2d}: v2={np.mean(v2_paks):.3f}, BroRL={np.mean(br_paks):.3f}, diff={np.mean(diffs):+.4f}")
    print()

    # Interesting: problems where BroRL > v2
    print("  Problems where BroRL > v2 (n_correct):")
    for p in range(60):
        v2_c = all_data["nemotron-v2"]["problems"][p]["n_correct"]
        br_c = all_data["nemotron-brorl"]["problems"][p]["n_correct"]
        if br_c > v2_c:
            print(f"    P{p} [{problems[p]['tier']}/{problems[p]['subject']}]: BroRL={br_c}, v2={v2_c}")
    print()

    # ── Save ────────────────────────────────────────────────────
    results["coverage"] = coverage
    with open(OUT_DIR / "rigorous_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {OUT_DIR / 'rigorous_results.json'}")


if __name__ == "__main__":
    main()
