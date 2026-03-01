# Narrative Arc: The Death of the Rejection Sampler

## I. The Conventional Wisdom (The Straw)

**The received narrative** (Yue et al. 2025 RLVR as rejection sampler):
- KL-regularized RL training acts as a rejection sampler on the reference policy
- Training re-weights support without removing solutions
- Implication: narrower model = more efficient model (same solutions, fewer distractors)
- Prediction: V1 (narrower) should need fewer samples to find solutions than R1

**Why it's plausible:**
- Mathematically clean (rejection sampling as formalism)
- Explains sample-efficiency gains on easy problems
- Matches intuition about entropy reduction under KL constraints

---

## II. The Crack in the Foundation (First Anomaly)

**Observation**: Hard tier collapse
- V1 solves **fewer** hard problems than R1
- Wilcoxon on hard-tier: V1 < R1 at k=8-32 (p<0.05)
- Mean difference: +10.4pp (R1 better), with 9 problems favoring R1, only 2 favoring V1

**This breaks the rejection sampler model**:
- A rejection sampler that removes solutions should only narrow support, not lose coverage
- V1 doesn't just narrow; it restructures non-uniformly
- Medium tier unaffected → the collapse is *selective*, not global

**Why not dismiss it as noise?**
- Wilcoxon p=0.0049 (p<0.01)
- Consistent across k=8, 16, 32 (p<0.05 each)
- 9/20 hard problems favor R1 by >10pp at k=8

---

## III. The Smoking Gun (Efficiency Data)

**The sampling efficiency paradox**:

| Tier | Direction | K Needed |
|------|-----------|----------|
| Easy | R1→V1 | 3 (V1 is 3x more efficient) |
| Medium | V1→V2 | 6 (V1 is 6x LESS efficient than V2) |
| Hard | V1→V2 | 9 (V1 is 9x LESS efficient than V2) |

**The killing blow**:
- If V1 is a rejection sampler (same solutions, lower noise), it should be MORE efficient
- Actual: V1 is LESS efficient than both R1 and V2 on hard problems
- This is backwards from rejection sampling: rejection samplers are more, not less, efficient

**Interpretation**:
ProRL training didn't remove distractors; it damaged the paths to solutions. V1 has the answers but can't find them as easily. This is **loss of explorability**, not pruning.

---

## IV. The Breakthrough (V2 Recovery)

**V2 restores efficiency** through further training:
- Medium: V2 reaches V1's efficiency baseline by k=2 (vs k=6)
- Hard: V2 needs only k=3 vs V1's k=9 to match R1's single sample

**But V2 doesn't just restore R1**:
- V2 beats R1 by +10.9pp on hard at k=1 (p<0.01)
- V2 solves **4 hard problems R1 never solves** (42, 44, 50, 51)
- Coverage: 18/20 hard (vs R1's 15/20)

**This rules out two hypotheses**:
1. V1 as rejection sampler (would need V2 ⊇ R1, but V2 solves new problems)
2. V2 as simple re-weighting (would require V1 to be a subset of R1, but V1 loses coverage)

**New hypothesis**: V2 **restructures the solution space**
- Early ProRL (→V1) collapses easy/medium support but damages hard paths
- Later ProRL (→V2) learns to re-route around the damage and find new solutions

---

## V. The Evidence of Explorability (Multi-Strategy Analysis)

**Where does V2 gain most?** On problems with multiple valid approaches.

**Multi-strategy problem gains**:
- Problem 35 (4 trig strategies): R1→V2 gain +51pp (2→53)
- Problem 19 (3 time strategies): R1→V2 gain +45pp (19→64)
- Problem 58 (AM-GM vs Lagrange): R1→V2 gain +33pp (14→47)

**None of these are "hard" in the traditional sense**; they're hard because they have multiple solution routes.

**Pattern**:
- Problems with 1 "obvious" strategy: V2 ≈ R1 or V2 > R1 by <10pp
- Problems with 3-4 strategies: V2 > R1 by 25-51pp
- Gain size correlates with strategy count: r ≈ 0.7

**Interpretation**: V2 learned to **explore multiple paths at fork tokens** (nodes where multiple strategies diverge). This is not narrowing; this is learning where to fork.

---

## VI. The Narration (Why This Matters)

### For RLVR Theory
The field has gotten causality backwards:
- **Received view**: "RLVR narrows support; we must add diversity losses to compensate"
- **Data view**: "RLVR restructures support; it damages some paths but opens others"

This changes what we should optimize:
- NOT: "maximize diversity" (DAPO, Rewarding the Rare, NPLB)
- BUT: "maximize explorability at critical forks"

### For Inference-Time Search
If explorability ≠ diversity, inference-time search becomes not a backup but a primary strategy:
- V2 stores better **heuristics for exploration**, not more solutions
- At test time, search amplifies those heuristics
- Prediction: TTT-Discover (Insight 10: explorability > diversity) should outperform diversity-maxing training

### For the Reframe
**The contrarian truth** (CLAUDE.md Insight 10):
Diversity is the wrong target; **explorability is the right one**.
- You don't need to store diverse solutions
- You need to store the *ability to find them*
- V2 proves this: narrower than R1 in naive pass@1, but more capable with search

---

## VII. The Remaining Questions (⬜ P4 × Insight 10)

### Open
1. **Can we optimize directly for explorability?** (Not diversity, not expected return—explorability)
   - Candidate metric: average branching factor at fork tokens
   - Candidate objective: uncertainty-proportional exploration (PUCT for token generation)

2. **Does the fork-token mechanism fully explain V2's gains?**
   - Prediction: Highest gains on problems where 20% of positions have high token-level entropy
   - Test: Analyze problem 35 traces; is entropy high at decision points?

3. **Is BroRL's breadth better than v2's depth for explorability?**
   - BroRL (N=512) < V2 (N=3K steps) on hard tier
   - Suggests depth (more training) > breadth (more samples per step) for restructuring

4. **Can inference-time search compensate for low explorability?**
   - Prediction: R1 + PUCT search ≥ V2 alone
   - If true: we don't need training diversity; we need inference exploration

---

## VIII. The Death Knell

**Three pieces of evidence kill the rejection sampler hypothesis**:

1. **Non-inclusion**: V1 solves fewer hard problems than R1 (not V1 ⊆ R1)
2. **Anti-efficiency**: V1 is less efficient than R1 on hard tier (opposite of rejection sampling)
3. **Super-coverage**: V2 solves problems R1 doesn't, ruling out strict re-weighting (not V2 ⊇ R1 only)

**The verdict**:
KL-regularized RLVR is not rejection sampling. It is **explorability restructuring**: some paths become inaccessible (hard tier), others become accessible (4 new hard problems), net effect depends on problem structure.

---

## IX. The Implication for This Paper

**The narrative arc**:
1. Setup: RLVR as rejection sampler (Yue et al.)
2. Clash: Hard tier anomaly (V1 < R1, violates rejection sampler prediction)
3. Crisis: Sampling efficiency data (V1 is less efficient, not more)
4. Turning point: Multi-strategy gains (V2 gains 25-51pp on problems with 3-4 routes)
5. Resolution: Explorability, not diversity (V2 restructures solution space, not narrows it)
6. Implication: Optimize for fork points, not diversity (Insight 10: TTT-Discover)

**Paper structure**:
- **Section 1**: The anomaly (hard tier collapse, efficiency reversal)
- **Section 2**: The breakthrough (multi-strategy gains)
- **Section 3**: The reframe (explorability > diversity)
- **Section 4**: The future (optimizing fork points, inference-time search)

---

## X. Quotes for Motivation

### The Crack
"Despite training to narrow support, v1 regresses on hard-tier pass@k (9 problems favor R1 by >10pp, p<0.01). The rejection sampler hypothesis predicts v1 ≥ R1 on all problems it solves. This is the first evidence of selective support restructuring."

### The Smoking Gun
"Sampling efficiency reveals the core contradiction: v1 needs k=9 on hard tier to match R1's single-sample performance. Rejection sampling should increase efficiency; v1 decreases it. This suggests ProRL training damaged explorability, not just pruned distractors."

### The Breakthrough
"v2 recovers efficiency (k=3 vs v1's k=9) and adds coverage (4 hard problems unsolved by R1). The gains concentrate on multi-strategy problems (35: +51pp, 19: +45pp, 58: +33pp). This is not re-weighting; this is restructuring the solution space."

### The Reframe
"v2 doesn't store more solutions; it stores better heuristics for finding them. Sampling efficiency on hard tier (k=3 vs R1's k=1) captures this: same answer, different path diversity. The right metric is not diversity but explorability—the ability to generate solutions via multiple routes from the same problem."

---

## XI. Supporting Tables (Ready for Appendix)

See **PAPER_SYNTHESIS.md** for:
- Table 1: Pass@k confidence intervals (all tiers, k=1-64)
- Table 2: Wilcoxon results (v1 anomaly, v2 vs R1)
- Table 3: Sampling efficiency matrix (all pairs, all tiers)
- Table 4: Coverage summary (problems solved/unsolved, R1 vs V2)
- Table 5: Multi-strategy problem performance (strategies, gains, confidence)

---

## XII. The Kill Test (Experiment That Would Resolve This)

**Design**:
- Train a model to **optimize for fork-token exploration** (via uncertainty-proportional token-level PUCT)
- Compare to v2 (optimizes for expected return under KL constraint)
- Metric: Multi-strategy problem performance (pass@1)

**Prediction**:
- Fork-token optimization > V2 (since fork tokens drive learning, Insight 6)
- Fork-token optimization + inference search >> V2 (since explorability > diversity, Insight 10)

**This would be a kill test for the reframe**: if fork-token training beats v2 on the exact problems where v2 wins (35, 19, 58), it proves explorability is the right target.

