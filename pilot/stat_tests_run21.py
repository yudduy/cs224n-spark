#!/usr/bin/env python3
"""
Run 21 Statistical Significance Tests
======================================
Tests whether Student-R1 vs Student-v2 differences are real.
"""

import json
import math
import numpy as np
from pathlib import Path
from scipy import stats

# ── Load data ──

STUDENT_DIR = Path("data/modal_runs/distill_pilot/full/traces")
TEACHER_DIR = Path("data/modal_runs/gen_traces_full")
JUDGE_PATH = Path("data/analysis/distillation_judge.json")
BASELINE_JUDGE = Path("data/analysis/llm_judge_pilot.json")

def load_traces(path):
    with open(path) as f:
        return json.load(f)

def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

problems = load_traces(STUDENT_DIR / "problems.json")
sr1 = load_traces(STUDENT_DIR / "student-r1" / "traces.json")
sv2 = load_traces(STUDENT_DIR / "student-v2" / "traces.json")
tr1 = load_traces(TEACHER_DIR / "r1-distill" / "traces.json")
tv2 = load_traces(TEACHER_DIR / "nemotron-v2" / "traces.json")
judge = load_traces(JUDGE_PATH)

K = 64

# ── Per-problem metrics ──

sr1_p1 = [pass_at_k(K, p["n_correct"], 1) for p in sr1["problems"]]
sv2_p1 = [pass_at_k(K, p["n_correct"], 1) for p in sv2["problems"]]
tr1_p1 = [pass_at_k(K, p["n_correct"], 1) for p in tr1["problems"]]
tv2_p1 = [pass_at_k(K, p["n_correct"], 1) for p in tv2["problems"]]

sr1_correct = [p["n_correct"] for p in sr1["problems"]]
sv2_correct = [p["n_correct"] for p in sv2["problems"]]
tr1_correct = [p["n_correct"] for p in tr1["problems"]]
tv2_correct = [p["n_correct"] for p in tv2["problems"]]

sr1_uniq = [p["n_unique_answers"] for p in sr1["problems"]]
sv2_uniq = [p["n_unique_answers"] for p in sv2["problems"]]
tr1_uniq = [p["n_unique_answers"] for p in tr1["problems"]]
tv2_uniq = [p["n_unique_answers"] for p in tv2["problems"]]

tiers = [p["tier"] for p in problems]

# ── Strategy counts from judge ──

sr1_strats = []
sv2_strats = []
for r in judge:
    if r.get("skipped") or r.get("error"):
        sr1_strats.append(0)
        sv2_strats.append(0)
        continue
    s = r.get("student_strats", {})
    sr1_strats.append(len(s.get("student-r1", [])))
    sv2_strats.append(len(s.get("student-v2", [])))

# ── Helper ──

def wilcoxon_test(a, b, label):
    a, b = np.array(a), np.array(b)
    diff = a - b
    nonzero = diff[diff != 0]
    if len(nonzero) < 5:
        print(f"  {label}: too few non-tied pairs ({len(nonzero)}), cannot test")
        return None
    stat, p = stats.wilcoxon(nonzero, alternative='two-sided')
    n = len(nonzero)
    # rank-biserial r = 1 - 2T/(n(n+1)/2)
    r_rb = 1 - (2 * stat) / (n * (n + 1) / 2)
    mean_a, mean_b = np.mean(a), np.mean(b)
    print(f"  {label}: mean_a={mean_a:.4f}, mean_b={mean_b:.4f}, diff={mean_a-mean_b:+.4f}")
    print(f"    Wilcoxon W={stat:.0f}, p={p:.6f}, n_nonzero={n}, r_rb={r_rb:+.3f}")
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"    → {sig}")
    return p

def bootstrap_ci(a, b, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    a, b = np.array(a), np.array(b)
    diffs = []
    n = len(a)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo, hi

# ── Tests ──

print("=" * 70)
print("RUN 21: STATISTICAL SIGNIFICANCE TESTS")
print("=" * 70)

# 1. Student-R1 vs Student-v2 accuracy
print("\n1. ACCURACY: Student-R1 vs Student-v2 (pass@1, n=60 problems)")
wilcoxon_test(sr1_p1, sv2_p1, "S-R1 vs S-v2 pass@1")
lo, hi = bootstrap_ci(sr1_p1, sv2_p1)
print(f"    Bootstrap 95% CI on (S-R1 - S-v2): [{lo:+.4f}, {hi:+.4f}]")

# Also test on raw correct counts
print("\n   Raw correct counts (out of 64):")
wilcoxon_test(sr1_correct, sv2_correct, "S-R1 vs S-v2 n_correct")
lo, hi = bootstrap_ci(sr1_correct, sv2_correct)
print(f"    Bootstrap 95% CI on diff: [{lo:+.2f}, {hi:+.2f}]")

# 2. Strategy diversity
print("\n2. STRATEGY DIVERSITY: Student-R1 vs Student-v2")
# Only count problems where judge ran (not skipped)
valid_idx = [i for i, r in enumerate(judge) if not r.get("skipped") and not r.get("error")]
sr1_s_valid = [sr1_strats[i] for i in valid_idx]
sv2_s_valid = [sv2_strats[i] for i in valid_idx]
print(f"   n={len(valid_idx)} problems with judge results")
n_r1_wins = sum(1 for a, b in zip(sr1_s_valid, sv2_s_valid) if a > b)
n_v2_wins = sum(1 for a, b in zip(sr1_s_valid, sv2_s_valid) if b > a)
n_tied = sum(1 for a, b in zip(sr1_s_valid, sv2_s_valid) if a == b)
print(f"   S-R1 > S-v2: {n_r1_wins}, S-v2 > S-R1: {n_v2_wins}, tied: {n_tied}")
wilcoxon_test(sr1_s_valid, sv2_s_valid, "S-R1 vs S-v2 strategies")

# 3. Students vs Teachers
print("\n3. STUDENT vs TEACHER ACCURACY")
wilcoxon_test(sr1_p1, tr1_p1, "Student-R1 vs Teacher-R1 pass@1")
wilcoxon_test(sv2_p1, tv2_p1, "Student-v2 vs Teacher-v2 pass@1")

# 4. Per-tier
print("\n4. PER-TIER: Student-R1 vs Student-v2")
for tier in ["easy", "medium", "hard"]:
    idx = [i for i, t in enumerate(tiers) if t == tier]
    a = [sr1_p1[i] for i in idx]
    b = [sv2_p1[i] for i in idx]
    wilcoxon_test(a, b, f"{tier} (n={len(idx)})")

# 5. Unique answers
print("\n5. UNIQUE ANSWERS: Students vs Teachers")
wilcoxon_test(sr1_uniq, tr1_uniq, "Student-R1 vs Teacher-R1 unique_ans")
wilcoxon_test(sv2_uniq, tv2_uniq, "Student-v2 vs Teacher-v2 unique_ans")

# 6. Summary table
print(f"\n{'=' * 70}")
print("SUMMARY TABLE")
print(f"{'=' * 70}")
print(f"{'Comparison':<45} {'Diff':>8} {'p':>10} {'Sig':>5}")
print("-" * 70)

tests = [
    ("S-R1 vs S-v2 pass@1", sr1_p1, sv2_p1),
    ("S-R1 vs S-v2 n_correct", sr1_correct, sv2_correct),
    ("S-R1 vs S-v2 strategies", sr1_s_valid, sv2_s_valid),
    ("Student-R1 vs Teacher-R1 pass@1", sr1_p1, tr1_p1),
    ("Student-v2 vs Teacher-v2 pass@1", sv2_p1, tv2_p1),
    ("S-R1 vs T-R1 unique_ans", sr1_uniq, tr1_uniq),
    ("S-v2 vs T-v2 unique_ans", sv2_uniq, tv2_uniq),
]

for label, a, b in tests:
    a_arr, b_arr = np.array(a), np.array(b)
    diff_arr = a_arr - b_arr
    nonzero = diff_arr[diff_arr != 0]
    mean_diff = np.mean(a_arr) - np.mean(b_arr)
    if len(nonzero) >= 5:
        _, p = stats.wilcoxon(nonzero, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {label:<45} {mean_diff:>+8.4f} {p:>10.6f} {sig:>5}")
    else:
        print(f"  {label:<45} {mean_diff:>+8.4f} {'N/A':>10} {'—':>5}")
