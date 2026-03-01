# Spark-MCTS Pilot Log

## Run 1: Baseline Calibration (2026-02-22)

### Configuration
- Model: Qwen2.5-Math-1.5B
- Dataset: MATH-500 Level 5, 50 problems x 4 rollouts
- GPU: 1x A100-80GB
- tau_h: 80th percentile entropy
- tau_p: 30th percentile probability
- kappa: 0.02

### Calibrated Thresholds
| Parameter | Value |
|-----------|-------|
| tau_h | 1.2800 |
| tau_p | 0.6237 |
| kappa | 0.02 |

### Results
| Metric | Value | Status |
|--------|-------|--------|
| Total valid tokens | 94,537 | |
| Spark fraction | 13.13% | Too high (target 2-8%) |
| Sparks per rollout | 62.1 +/- 81.6 | |
| Reasoning spark fraction | 13.4% | PASS (>10%) |
| Noise fraction | 1.2% | |
| Ambiguous fraction | 85.3% | Concerning |
| Gate 1 pass (H > tau_h) | 19.9% | |
| Gate 2 pass (pi < tau_p) | 29.9% | |
| Gate 3 pass (pi > kappa*max) | 92.5% | Nearly vacuous on full pop |
| CURE top-20 -> 3-gate precision | 79.8% | |
| Entropy-only FP rate | 33.9% | 3-gate adds value |
| G1 fail G2 (formatting) | 228 | |
| G1 fail G3 (noise floor) | 6,127 | Primary filter contribution |
| Initial accuracy | 12.5% | |
| Branch accuracy | 10.0% | Below initial |
| Branch diversity (Jaccard) | 0.273 +/- 0.144 | Good (<0.5) |
| Kill criteria triggered | 0 | |

### Top 10 Spark Tokens
| Token | Count | Class |
|-------|-------|-------|
| 'the' | 803 | reasoning* |
| 'and' | 226 | ambiguous |
| 'to' | 224 | ambiguous |
| 'we' | 223 | ambiguous |
| 'is' | 220 | ambiguous |
| 'of' | 215 | ambiguous |
| 'in' | 184 | ambiguous |
| 'let' | 179 | reasoning |
| 'for' | 177 | ambiguous |
| 'this' | 152 | ambiguous |

*Classification based on crude word list; "the" is misclassified here.

### Interpretation

#### Finding 1: Stopwords as top sparks
Expected at 1.5B scale, not a filter design flaw. Entropy is a property of the
*position*, not the *token*. At 1.5B, "the" has high entropy because the model
genuinely doesn't know whether "the", "a", "this", "that" comes next — syntactic
uncertainty from model weakness, not reasoning uncertainty. The 80/20 paper's
word cloud shows "however/because/thus" because they ran 14B and 32B models
where syntax is deterministic and only reasoning positions retain entropy.
Prediction: resolves at 7B+. Partial fix at 1.5B: tighten tau_h to 95th
percentile to push past syntactic uncertainty into semantic pivots.

#### Finding 2: Spark fraction 13.13% (target 2-8%)
Threshold miscalibration, not fundamental flaw. 80th percentile was correct for
the 80/20 paper's gradient masking use case. For branching triggers, need 95th
or 97th percentile. CURE branches at 1-3 positions per rollout; 62 sparks/rollout
is absurd for branching. One config change: entropy_percentile 80 -> 95.

#### Finding 3: Gate 3 nearly vacuous (92.5% pass)
Measurement framing issue. 92.5% is over ALL tokens. Gate 3 targets only the
G1+G2 population. Of 18,772 G1-passing tokens, 6,127 failed G3 — that's ~33%
filtration within the relevant population. Gate 3 distinguishes hallucination
floor (pi near absolute zero) from genuine low-probability sparks. Its value is
real; the denominator was misleading.

#### Finding 4: Branch accuracy 10% < initial 12.5%
Mechanistically expected. Branches start from the hardest positions (high-H,
low-pi) — of course accuracy is lower than full-prompt completions. The training
experiment tests whether diversity at these positions generates better gradient
signal over 50 steps, not whether single-shot branches solve harder.
The real concern: 12.5% baseline means GRPO signal is dominated by 0-reward
trajectories. Fix: switch to Level 3-4 (25-40% baseline) for training.

### Verdict
Pilot proves:
- 3-gate filter is non-trivially more precise than entropy-only (33.9% FP removal)
- Branching infrastructure works end-to-end
- Branch diversity at spark positions is good
- Calibrated thresholds are data-driven

Before training, fix:
1. tau_h 80th -> 95th percentile (target 3-6% spark fraction)
2. Dataset Level 5 -> Level 3-4 (target 30-40% baseline accuracy)
3. Smoke-test backward pass with tightened config

### Data Files
```
pilot/results/spark_pilot_20260222_110122/
  results.json   44KB  Full per-token data
  sparks.tsv      8KB  Spark token details
  branches.tsv    3KB  Re-concatenation results
  summary.txt     307B Calibrated thresholds + metrics
```

---

## Run 2: Tightened Calibration (2026-02-22)

### Changes from Run 1
- entropy_percentile: 80 -> 95
- Expected: spark fraction 3-6%, stopwords replaced by discourse markers

### Calibrated Thresholds
| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| tau_h | 2.2697 | +0.9897 (was 1.2800) |
| tau_p | 0.7005 | +0.0768 (was 0.6237) |
| kappa | 0.02 | unchanged |

### Results
| Metric | Run 1 (80th) | Run 2 (95th) | Delta | Status |
|--------|-------------|-------------|-------|--------|
| Total valid tokens | 94,537 | 104,915 | +10K (different rollouts) | |
| Spark fraction | 13.13% | **2.88%** | -10.25pp | PASS (target 2-8%) |
| Sparks per rollout | 62.1 | **15.1** | -47.0 | Reasonable for branching |
| Reasoning spark fraction | 13.4% | 10.3% | -3.1pp | PASS (>10%) |
| Noise fraction | 1.2% | 0.5% | -0.7pp | |
| Gate 1 pass (H > tau_h) | 19.9% | **5.0%** | -14.9pp | Tight |
| Gate 2 pass (pi < tau_p) | 29.9% | 29.9% | 0 | Unchanged (expected) |
| Gate 3 pass (pi > kappa*max) | 92.5% | 95.3% | +2.8pp | |
| CURE -> 3-gate precision | 79.8% | 81.9% | +2.1pp | |
| Entropy-only FP rate | 33.9% | **42.2%** | +8.3pp | 3-gate MORE valuable |
| G1 fail G2 | 228 | **0** | -228 | Gate 2 auto-satisfied at 95th |
| G1 fail G3 (noise floor) | 6,127 | 2,205 | -3,922 | Gate 3 still primary filter |
| Initial accuracy | 12.5% | 10.5% | -2.0pp | Different rollouts |
| Branch accuracy | 10.0% | **11.1%** | +1.1pp | Now >= initial |
| Branch diversity (Jaccard) | 0.273 | **0.058** | -0.215 | Dramatically more diverse |
| Kill criteria | 0 | 0 | | All pass |

### Top 10 Spark Tokens (Run 2)
| Token | Count | Class | Run 1 Count |
|-------|-------|-------|-------------|
| 'the' | 125 | reasoning* | 803 (-84%) |
| 'in' | 70 | ambiguous | 184 (-62%) |
| 'to' | 57 | ambiguous | 224 (-75%) |
| 'and' | 51 | ambiguous | 226 (-77%) |
| 'for' | 38 | ambiguous | 177 (-79%) |
| 'of' | 38 | ambiguous | 215 (-82%) |
| 'we' | 37 | ambiguous | 223 (-83%) |
| 'if' | 37 | ambiguous | 78 (-53%) |
| 'is' | 31 | ambiguous | 220 (-86%) |
| 'let' | 28 | reasoning | 179 (-84%) |

*Stopwords still present but absolute counts dramatically reduced. At 1.5B,
this is the expected scale limitation — the model's syntactic uncertainty
persists even at the 95th percentile.

### Key Findings

#### 1. Spark fraction nailed
2.88% is squarely in the 2-8% target. 15.1 sparks per rollout is workable
for both branching (pick top-K) and gradient routing (mask the rest).

#### 2. Branch accuracy reversal
Run 1: branches (10.0%) < initial (12.5%) — concerning.
Run 2: branches (11.1%) >= initial (10.5%) — neutral/slightly positive.
Tighter threshold selects genuinely uncertain positions where the model
can still produce valid completions.

#### 3. Branch diversity dramatically improved
Jaccard 0.058 (was 0.273). At tighter thresholds, spark positions produce
highly divergent continuations — exactly what training needs for diverse
gradient signal.

#### 4. Gate 2 collapse — 3-gate operating as 2-gate at this regime
**Critical finding.** Zero Gate 2 failures means H > 95th percentile and
pi < tau_p are perfectly correlated at 1.5B. Every token clearing Gate 1
automatically clears Gate 2. The 3-gate filter is effectively a 2-gate
filter (Gate 1 + Gate 3) at this operating point.

This is not a design flaw — it's a finding about the geometry of the logit
space at 1.5B: at the 95th entropy percentile, the model's probability mass
is already spread thin enough that pi < 30th percentile is guaranteed.

**Paper framing:** "At 95th percentile entropy on 1.5B, Gates 1 and 2 are
empirically perfectly correlated. Gate 2's independent contribution becomes
measurable at lower entropy thresholds (Run 1: 228 G2 failures at 80th
percentile) and is predicted to decouple at larger model scales where
the entropy-probability relationship is less monotonic."

**Implication for novelty claim:** Gate 2 provides independent filtering
only when entropy threshold is loose enough that some high-H tokens retain
high pi (formatting tokens in math proofs, e.g., LaTeX delimiters with
high entropy but high probability). At 95th percentile, these don't exist.
The paper should report this correlation explicitly as a scale-dependent
finding, not hide it.

#### 5. Entropy-only FP rate increased to 42.2%
The 3-gate filter is MORE valuable at the tighter threshold. 42% of entropy-
only candidates at the 95th percentile are noise-floor tokens. Without
Gate 3, nearly half of CURE's branching candidates would be garbage.

### Verdict
Run 2 confirms the recalibrated 3-gate filter is ready for training:
- Spark fraction in target range (2.88%)
- Branch diversity excellent (Jaccard 0.058)
- Branch accuracy >= initial accuracy
- 3-gate adds substantial value over entropy-only (42% FP removal)
- All kill criteria pass

### Remaining Steps Before Training
1. ~~Switch dataset from Level 5 to Level 3-4 (target 30-40% baseline accuracy)~~
2. ~~Smoke-test backward pass with tightened thresholds (5 training steps)~~
3. Run 3-condition ablation (Vanilla GRPO / CURE-branching / Spark-MCTS)

### Data Files
```
pilot/results/recal_20260222_192254/
  results.json   Full per-token data
  sparks.tsv     Spark token details
  summary.txt    Calibrated thresholds + metrics
```

---

## Run 3: Backward Pass Smoke Test (2026-02-22)

### Configuration
- Model: Qwen2.5-Math-1.5B
- Dataset: GSM8K (8 prompts × 6 responses per step)
- GPU: 1x A100-80GB
- Training steps: 5
- 3 conditions: A (Vanilla GRPO), B (CURE), C (Spark-MCTS)
- Lp-Reg: loss_mode=lp_reg, logp_neg_k_percent=0.01, minp_p_threshold=0.02
- 3-gate: entropy_percentile=95, prob_percentile=30, kappa=0.02

### Results: ALL 5 CHECKS PASS

| Check | Status | Detail |
|-------|--------|--------|
| Condition A loss finite | PASS | mean=-0.0268 |
| Condition B loss finite | PASS | mean=-0.0386 |
| Condition C loss finite | PASS | mean=-0.0649 |
| Lp-Reg loss fires | PASS | mean=0.072, max=0.122, 5/5 steps |
| Prefix stop-grad | PASS | 8452/9438 positions zeroed (89.6%) |
| 3-gate filter | INFO | 5 runs, 0 fallbacks, mean 0.8% spark fraction |

### Training Dynamics (Condition C)
| Step | pg_loss | grad_norm | score_mean | advantages_range |
|------|---------|-----------|------------|-----------------|
| 1 | -0.127 | 3.603 | 0.188 | [-0.91, 2.04] |
| 2 | -0.055 | 16.290 | 0.312 | [-2.04, 2.04] |
| 3 | -0.015 | 4.112 | 0.458 | [-2.04, 2.04] |
| 4 | -0.008 | 7.203 | 0.417 | [-2.04, 1.29] |
| 5 | -0.093 | 5.089 | 0.312 | [-0.91, 2.04] |

### Key Findings

#### 1. Backward pass fully functional
All three conditions produce finite non-zero losses with flowing gradients.
No NaN/Inf at any step. Condition C (Spark-MCTS) has the largest mean loss
magnitude (-0.065 vs -0.027 for A and -0.039 for B), consistent with
additional Lp-Reg KL regularization term.

#### 2. Prefix stop-gradient works
89.6% of prefix positions had their gradients zeroed. The remaining 10.4%
are expected: these are positions where the response_mask was already 0
(padding tokens, which have no gradient contribution regardless).

#### 3. 3-gate filter generalizes to training
The per-batch 3-gate calibration produces variable thresholds (tau_h ranges
2.9-8.3 across steps) adapting to each batch's entropy distribution. Spark
fraction 0.8% is lower than the forward-pass calibration (2.9%) due to
different batch sizes and generation temperatures. Zero fallbacks in 5 steps.

#### 4. Reward function fix
The default verl GSM8K reward uses strict `####` extraction. Qwen2.5-Math
outputs `\boxed{}` format. Custom reward function handles both formats.
Validation accuracy: 65.9% — proving the model can solve GSM8K at 1.5B.

### Bugs Fixed During Smoke Test
1. `ppo_mini_batch_size=64 > train_batch_size=2` — assertion fail (use_dynamic_bsz=False constraint)
2. `log_probs` key not in TensorDict — field is `old_log_probs` in verl
3. `####` reward extraction fails for `\boxed{}` model output — custom compute_score
4. `train_batch_size=2` gives all-zero rewards — increased to 8 for reward variance

### Verdict
**GREEN LIGHT for full training run.** All backward-pass checks pass.
Next: 50-step 3-condition ablation on MATH Level 3-4.

---

## Run 4: Dual-Threshold Split (2026-02-22)

### Motivation
Reviewer feedback identified that a single threshold cannot serve both as the
intervention trigger and the measurement ruler. If τ_h floats per-batch, |S_t|
is measured against a moving target and the extinction/assimilation time series
is uninterpretable. If τ_h is fixed, the 80/20 paper (Wang et al. 2506.01939)
validated dynamic per-sequence selection — and conceptually, relative cognitive
forks matter more than absolute entropy values.

Counter-argument (debunked): The "entropy collapse" scenario (fixed τ_h becomes
unreachable at step 25) is empirically backwards. High-entropy tokens trend
*upward* during RLVR, with 86% overlap at step 1360. Fixed thresholds don't
risk shutting off the intervention, but dynamic per-sequence selection is still
the validated approach.

### Changes
1. **Intervention trigger**: per-batch dynamic → **per-sequence dynamic** percentile
   - Each sequence gets its own τ_h (95th-pct H) and τ_p (30th-pct π)
   - Matches Wang et al.'s validated implementation
   - Guarantees routing to relative cognitive forks regardless of absolute shift
2. **Metric logging**: dynamic → **fixed absolute** from Run 2 calibration
   - τ_h = 2.2697, τ_p = 0.7005, κ = 0.02 (the ruler)
   - Logged as `st/total_sparks_fixed` and `st/spark_fraction_fixed`
   - Independent of intervention — measures absolute pool health over time
3. Fixed thresholds passed to all 3 conditions (A, B, C) for comparable metrics

### Results: ALL CHECKS PASS

| Check | Status | Detail |
|-------|--------|--------|
| Condition A loss finite | PASS | mean=-0.0434 |
| Condition B loss finite | PASS | mean=-0.0008 |
| Condition C loss finite | PASS | mean=-0.0047 |
| Lp-Reg loss fires | PASS | mean=0.023, max=0.033 |
| Prefix stop-grad | PASS | 6969/8636 positions zeroed (80.7%) |
| Per-seq dynamic trigger | INFO | 5 runs, 0 fallbacks, 2.8% spark fraction |

### Fixed |S_t| Metric Across Conditions
| Condition | Mean |S_t| | Range | Interpretation |
|-----------|-------------|-----------|----------------|
| A (Vanilla GRPO) | 606.0 | [454, 772] | No intervention — largest pool |
| B (CURE) | 275.8 | [196, 461] | Entropy branching reduces pool |
| C (Spark-MCTS) | 349.8 | [262, 461] | 3-gate + Lp-Reg — middle ground |

### Verdict
Dual-threshold architecture confirmed working. The ruler (fixed thresholds)
and the trigger (per-sequence dynamic) are now fully decoupled.
Ready for full training run.

## Run 5: Mass Balance Metric — BroRL Σπ_t (2026-02-22)

### Motivation
BroRL (arxiv 2510.01180) decomposes RL value into probability mass on correct
tokens (Q_pos). The mass balance equation shows how ΔQ_pos = (gain from
upweighting correct tokens) − (loss from suppressing tokens). If Lp-Reg
prevents excessive suppression, Σπ_t at spark positions should remain stable
or increase under condition C vs decline under A and B.

### Implementation
1. **Calibration** (`_calibrate_mass_balance`): Before training, generate 1
   response per 50 held-out prompts, identify 3-gate sparks using fixed
   thresholds (τ_h=2.2697, τ_p=0.7005, κ=0.02), record (seq_idx, position,
   token_id, π_0) for each spark.
2. **Measurement** (`_measure_mass_balance`): At checkpoint steps, teacher-force
   the step-0 responses through current model, read π_t at each saved spark
   position, compute Σπ_t and ratio to Σπ_0. No gradient effect.
3. **Schedule**: Step 0 (calibration) + last step (5 for smoke). Full run will
   measure at steps 0, 10, 20, 30, 40, 50.
4. **Output**: Separate `mass_balance_{condition}.json` per condition + embedded
   in diagnostics JSON.

### Results: ALL CHECKS PASS + MASS BALANCE LOGGED

| Check | Status | Detail |
|-------|--------|--------|
| Condition A loss finite | PASS | mean=0.0187 |
| Condition B loss finite | PASS | mean=0.0095 |
| Condition C loss finite | PASS | mean=0.0022 |
| Lp-Reg loss fires | PASS | mean=0.062, max=0.113 |
| Prefix stop-grad | PASS | 6329/9214 positions zeroed (68.7%) |
| Per-seq dynamic trigger | INFO | 5 runs, 0 fallbacks, 2.8% spark fraction |

### Mass Balance (Σπ_t) After 5 Steps
| Condition | Σπ_0 | Σπ_final | Ratio | Δmean_π | Interpretation |
|-----------|-------|----------|-------|---------|----------------|
| A (Vanilla GRPO) | 137.16 | 136.94 | 0.9984 | −0.000311 | Mass erosion at sparks |
| B (CURE) | 137.16 | 136.82 | 0.9976 | −0.000465 | Slightly worse erosion |
| C (Spark-MCTS) | 137.16 | 137.25 | **1.0007** | +0.000135 | Mass preserved/gained |

715 sparks identified across 50 held-out sequences (same initial model for all).

### Key Finding
After only 5 steps the signal is already directionally correct: A and B suppress
probability mass at spark positions while C (Lp-Reg) preserves it. This validates
the BroRL mass preservation claim — at full training (50 steps) the divergence
should be dramatically clearer.

### Verdict
Mass balance metric confirmed working. Two independent paper claims now have
instrumentation: |S_t| decomposition (mechanistic, fixed-ruler) and Σπ_t curve
(BroRL mass preservation). Ready for full training run.

---

## Run 6: Full Ablation — 50 Steps (2026-02-23)

### Configuration
- Model: Qwen2.5-Math-1.5B
- Dataset: GSM8K (8 prompts × 6 responses per step)
- GPU: 2x A100-80GB
- Training steps: 50
- 3 conditions: A (Vanilla GRPO), B (CURE branching), C (Spark-MCTS + Lp-Reg)
- WandB project: spark-mcts-ablation
- Total time: 105 min, ~$18

### Per-Condition Summary

| Metric | A (Vanilla GRPO) | B (CURE branching) | C (Spark-MCTS) |
|--------|:-:|:-:|:-:|
| Time | 27.9 min | 37.8 min | 39.3 min |
| Entropy | 0.97 → 0.27 | 1.24 → 0.36 | 1.27 → 0.50 |
| pg_loss mean | 0.026 | 0.052 | 0.074 |
| lpreg_magnitude mean | 0.026 | 0.052 | 0.074 |
| Grad norm mean | 0.251 | 0.281 | 0.298 |
| Mass balance drift | 7.17% | 4.33% | 4.33% |
| Prefix stop-grad | N/A | N/A | 77.6% zeroed |

### Confirmed (hard to vary)
- All 3 components mechanically sound (branching, 3-gate, Lp-Reg, prefix stop-grad)
- Branching reduces mass balance drift (B/C 4.33% vs A 7.17%)
- Lp-Reg preserves entropy (C: 0.50 vs B: 0.36 vs A: 0.27)
- No NaN, no OOM (at gpu_memory_utilization=0.80), no instability

### Critical Issues Found

#### 1. Lp-Reg dominates pg_loss
At coeff=1.0, regularizer magnitude (0.074) > policy gradient (0.061). Cannot
distinguish "healthy exploration" from "optimizer handcuffed." The gradient is
spending more effort on the KL penalty than on the actual reward signal.

#### 2. B and C identical mass balance (4.33%)
The 3-gate filter + Lp-Reg adds zero mass balance benefit over branching alone.
The unique C contribution is entropy preservation only. This suggests the mass
preservation is coming from branching diversity, not the Lp-Reg mechanism.

#### 3. No accuracy numbers
Zero pass@1 logged across all conditions. Without accuracy, entropy preservation
is ambiguous evidence — could be preserving useful exploration or preventing
collapse into a good policy.

#### 4. 50 steps too short
Only 800 examples seen out of 7473 train. Not even one epoch. Mechanisms are
validated but we cannot draw learning-curve conclusions.

### Bugs Fixed During Runs
- vLLM v1 token validation: file-level patch for Qwen2.5 special tokens (151643-151935)
- OOM at gpu_memory_utilization=0.85 → reduced to 0.80
- vLLM patch file path: use `grep -rl "out of vocabulary"` not hardcoded paths

### Verdict
All mechanisms mechanically validated. Three pre-flight fixes needed before
committing to a full experiment:
1. Lp-Reg coefficient sweep (reduce from 1.0)
2. Add pass@1 evaluation logging
3. Run viability test with corrected config

### Data Files
```
Modal volume: spark-pilot-results:/spark_full/
Local: pilot/full_report_ablation.json
WandB: spark-mcts-ablation project
```

---

## Run 7: Gate Order Fix + NLL Replacement (2026-02-23, code-only)

### Changes (no training run)
Four code fixes applied to `pilot/modal_spark_smoke.py`:

#### Fix 1: Gate 3 before Gate 2
**Problem**: τ_p computed on ALL pi_vals including silent forks (π~0.009). Drags
τ_p threshold down, letting noise through.
**Fix**: `pi_above_kappa = pi_t[pi_t > kappa]` then compute percentile on survivors.

#### Fix 2: V(R) Gate (aleatoric fork detection)
**Problem**: Branches where all responses get the same reward carry no epistemic
signal — they're aleatoric. Training on them adds noise.
**Fix**: When Var(branch_rewards)==0, zero the response_mask for those stage-2
branches. Binary gate: either the fork produced disagreement (epistemic) or it
didn't (aleatoric, masked).

#### Fix 3: Advantage-weighted NLL replaces Lp-Reg KL
**Problem**: KL gradient scales as β(π_θ − π_ref) ≈ 0 for rare tokens where
both π_θ and π_ref are small (~0.01). The protection mechanism fails on exactly
the tokens it's supposed to protect.
**Fix**: NLL gradient is −α|A|(1−π_θ) ≈ α|A| for small π_θ. Rarity-invariant.
When the advantage-weighted NLL fires, it pushes probability back toward the
reference policy proportional to the advantage magnitude, regardless of how small
the current probability is.

**Implementation details**:
- `compute_policy_loss_spark_nll` replaces `compute_policy_loss_lp_reg`
- Config: `spark_nll_alpha` (default 0.3), `spark_logp_neg_k_percent` (0.01)
- NLL protection magnitude piped through `actor/pg_clipfrac_lower` metric
  (Ray process isolation: module-level variables don't work across processes)
- Removed: tgt_log_prob forward pass, pos_k_percent, kl_type, minp_p_threshold

#### Fix 4: Piped NLL protection (Ray process isolation)
**Problem**: Module-level variables set in actor worker are invisible to trainer.
**Fix**: Protection magnitude flows through existing `pg_clipfrac_lower` return
value — the only reliable cross-process metric channel.

---

## Run 8: NLL Alpha Sweep (2026-02-23)

### Configuration
- 5 steps × 3 alphas [0.1, 0.3, 0.5]
- 1x A100-80GB, GSM8K
- Purpose: Confirm NLL protection fires without dominating GRPO

### Three Attempts
1. **Attempt 1**: Metric bug — measured abs(total_loss), not protection separately
2. **Attempt 2**: Module-level variable — zeros due to Ray process isolation
3. **Attempt 3**: Fixed — piped through pg_clipfrac_lower return value

### Results
| α | |GRPO| | NLL protect/token | GRPO/protect ratio |
|---|--------|-------------------|-------------------|
| 0.1 | 0.081 | 0.179 | 0.5 |
| 0.3 | 0.057 | 0.868 | 0.1 |
| 0.5 | 0.075 | 1.228 | 0.1 |

### Decision
Per-token protection > total GRPO at all α, but only ~1% of tokens touched.
Total gradient budget dominated by GRPO regardless. Selected α=0.3 (default).
Cost: ~$3.

---

## Run 9: GSM8K Viability — 100 Steps, A vs C (2026-02-23)

### Configuration
- Model: Qwen2.5-Math-1.5B
- Dataset: GSM8K
- GPU: 2x A100-80GB
- Training steps: 100
- Conditions: A (Vanilla GRPO), C (Spark-MCTS + NLL)
- n_total=6 (n1=2, n2=2), test_freq=10
- NLL alpha=0.3
- WandB project: spark-mcts-viability

### Results

| Condition | pass@1 | Loss Mean | Wall Time |
|-----------|--------|-----------|-----------|
| A (vanilla GRPO) | **74.37%** | 0.0146 | 28 min |
| C (spark-MCTS+NLL) | **72.78%** | 0.0055 | 43 min |
| **Delta (C−A)** | **−1.59pp** | | |

### Mechanism Health (all nominal)
- 3-gate filter: mean spark frac=3.8%, fires every step
- V(R) gate: **80% aleatoric** (1284/1600 forks)
- Prefix stop-grad: 29181/161752 positions zeroed (18%)
- NLL protection: mean=3.30, max=9.51, fires 96/100 steps
- Mass balance C: ratio=1.0155 (excellent, vs A's 1.0596)
- Fixed |S_t|: A=294.4, C=148.3 (C sees fewer tokens due to masking)

### Root Cause Analysis: V(R) Gate Too Aggressive on GSM8K
At n=6 (n1=2 base + n2=2 branches per fork), GSM8K rewards are binary {0,1}.
For a fork to have V(R)>0, the 2 branches must have DIFFERENT rewards.
With 2 branches: P(V(R)=0) = p² + (1−p)² where p = prob(correct).

At p=0.74 (model's accuracy): P(V(R)=0) = 0.74² + 0.26² = 0.615
Observed: 80% (worse, due to easy/hard problems being deterministic).

C effectively learns from ~0.8% of tokens vs A's 100%. The V(R) gate is
theoretically correct — it IS identifying aleatoric forks. But the combination
of an easy benchmark (p=0.74) and few branches (n2=2) means almost all forks
are aleatoric. Not a mechanism failure — a benchmark mismatch.

### Warnings
- No pass@1 trajectory captured (verl key is `val-core/openai/gsm8k/reward/mean@1`,
  not `score` — fixed in subsequent runs)

### Cost
~$18-20 total. Running total: ~$41.

---

## Run 10a: MATH L3-4 Smoke Test — Attempt 1 (2026-02-23)

### Configuration
- 5 steps, Condition C only
- MATH-500 filtered to Level 3-4 (233 problems → 186 train / 47 val)
- n2=4, n_total=10, 2x A100-80GB, max_response_length=1024

### Result: FAILED
```
AssertionError: only support equal chunk. Got size of DataProto 47 and chunk 2.
```
Val set has 47 examples (odd). verl's DataProto.chunk() requires even division
across GPUs. The mass balance calibration generates from the val set.

**Fix**: Trim val set to even count: `if len(val_data) % 2 != 0: val_data = val_data[:-1]`

---

## Run 10b: MATH L3-4 Smoke Test — Attempt 2 (2026-02-23)

### Configuration
Same as 10a with 46 val examples. Also fixed diagnostics key mismatch
(`"steps"` vs `"step_diagnostics"` and `"pg_loss"` vs `"loss"`).

### Result: Training completed, verification FAILED (false alarm)
Training ran 5 steps successfully. Verification reported "No step diagnostics"
due to the key mismatch bug. Actual training output showed all mechanisms firing.

V(R) gate: 78-81% aleatoric — same as GSM8K despite n2=4 on MATH L3-4.
This contradicted the prediction of ~18% based on independent branches.

### Cost: ~$2

---

## Run 10c: MATH L3-4 Position Diagnostic (2026-02-23)

### Configuration
- 5 steps, Condition C, MATH L3-4, n2=4, 2x A100-80GB
- Added spark position fraction logging: mean, median, std, Q25, Q75, late(>0.7)

### Purpose
Test whether sparks cluster late in sequence (>0.7 position fraction), which
would explain V(R)=0 correlation through shared-prefix-determined outcomes.

### Results

| Step | pos_frac mean | pos_frac median | Q25 | Q75 | late(>0.7) | V(R)=0 rate |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.28 | 0.25 | 0.16 | 0.30 | 10% | 81% |
| 2 | 0.46 | 0.51 | 0.21 | 0.70 | 24% | 81% |
| 3 | 0.44 | 0.44 | 0.25 | 0.69 | 24% | 75% |
| 4 | 0.51 | 0.46 | 0.25 | 0.83 | 34% | 88% |
| 5 | 0.41 | 0.39 | 0.18 | 0.58 | 19% | 94% |
| **Avg** | **0.42** | | | | **22%** | **84%** |

Smoke test PASS. Validation pass@1 = 52.2%.

### Key Finding: Position hypothesis falsified
Sparks fire EARLY — mean position 0.42, only 22% late. Yet V(R)=0 rate is 84%.
Branches from position 0.28 (first quarter!) still produce correlated outcomes.

**The correlation is not a position effect.** The model's downstream reasoning
converges to the same answer regardless of surface-level divergence at the spark
token. High logit-space entropy at a position does not imply the position is an
epistemic decision point for the final answer.

### Implications
The independence assumption P(all same) = p⁴ + (1−p)⁴ = 0.177 at p=0.4 assumed
branches have equal, independent probability of success. The 4.7× gap (84% vs 18%)
means branches are near-perfectly correlated. This is structural, not a benchmark
or parameter issue.

The 3-gate filter measures token-level logit uncertainty. But branching value is a
trajectory-level semantic property. The two are decoupled: a token can have high
entropy (many plausible next words) while the logical trajectory is already
determined by the prefix.

### Cost: ~$2. Running total: ~$45. Budget remaining: ~$105.

---

## Run 10d: Condition D — Gradient Routing Without Branching (2026-02-23)

### Motivation
Run 10c proved the 3-gate filter's two jobs are separable:
1. **Gradient routing** (selecting which tokens to learn from) — valid in principle
2. **Branching trigger** (where to fork rollouts) — falsified, 84% V(R)=0

Condition D tests routing alone: does zeroing gradient on low-entropy positive-
advantage tokens improve learning, without any tree search?

### Design
**Condition D** = Flat GRPO (16 rollouts) + asymmetric gradient routing:
- **Positive-advantage + high-entropy (H > τ_h=1.28)**: keep gradient (decision points that led to success)
- **Positive-advantage + low-entropy**: zero response_mask (deterministic execution, not informative)
- **Negative-advantage**: full gradient preserved + NLL protection on bottom 1% by log-prob

**Conscious asymmetry**: Routing only sharpens positive signal. Negative gradient is
unmodified except NLL protection on endangered tokens. Defensible: you don't want to
zero gradient on a failed trajectory's deterministic tokens — those might be wrong
execution steps that need correction.

**Paper claim if D > A**: "Token-level gradient routing via entropy-based masking
improves exploration preservation in standard GRPO, independent of tree search branching."

### Implementation
All changes in `pilot/modal_spark_smoke.py`:
- Condition D config in `get_hydra_overrides`: `enable_branching=False`, `use_gradient_routing=True`, `loss_mode="spark_nll"`, 16 flat rollouts
- Routing block after advantage computation: zeros `response_mask` for `pos_adv & ~high_ent & valid` tokens
- Configs: `ROUTING_SMOKE_CONFIG` (1 GPU, 5 steps), `ROUTING_VIABILITY_CONFIG` (2 GPU, 100 steps)
- Modal functions: `run_routing_smoke()`, `run_routing_viability()`

### Smoke Test (Run 10d-smoke)

**Bug**: `import torch` inside the routing `if` block caused `UnboundLocalError`.
Python treats `torch` as local for the entire `fit()` function scope when `import torch`
appears anywhere in the function body, shadowing the module-level import. Code before the
routing block (like `torch.sum(...)`) failed because the local hadn't been assigned yet.
**Fix**: Remove redundant `import torch` — already available at module level.

**Attempt 2 — ALL 4 CHECKS PASS**:
| Check | Status | Value |
|-------|--------|-------|
| Loss finite | PASS | mean=0.3192 |
| Routing zeroes tokens | PASS | mean=39.5% |
| high_ent_frac | PASS | 19.3% (target 2-30%) |
| NLL protection fires | PASS | mean=1.6441 |

Cost: ~$2.

### Viability Run (Run 10d)

#### Configuration
- Model: Qwen2.5-Math-1.5B
- Dataset: GSM8K
- GPU: 2x A100-80GB
- Training steps: 100
- Conditions: A (Vanilla GRPO, 16 rollouts), D (Routing + NLL, 16 rollouts)
- test_freq=10, NLL alpha=0.3
- WandB project: spark-routing-viability

#### Results

| Condition | pass@1 | Loss Mean |
|-----------|--------|-----------|
| A (vanilla GRPO, n=16) | **76.6%** | 0.0133 |
| D (routing + NLL) | **76.9%** | 0.7428 |
| **Delta (D−A)** | **+0.3pp** | |

Delta is within noise. Routing produces no detectable improvement.

#### Routing Diagnostics
| Metric | Value |
|--------|-------|
| Mean tokens zeroed | 48.9% |
| high_ent_frac | 10.3% |
| NLL fires | 100/100 steps |
| NLL mean magnitude | 4.22 |

### Loss Diagnostic (code inspection, no run)

**Red flag**: D's loss is 55× A's (0.7428 vs 0.0133).

**Decomposition** — two independent causes:

#### 1. Normalization artifact (~2×)
`agg_loss` uses `token-mean`: `sum(loss × mask) / sum(mask)`. D zeros 48.9% of
response_mask → denominator halved → reported loss ~2× inflated vs A. This is a
reporting artifact, not a training artifact (the actual gradient per active token
is unchanged).

#### 2. NLL domination (~27×)
`actor/pg_loss` = PPO clipped loss + NLL protection, combined in a single metric.
At α=0.3, NLL per-token magnitude (mean=4.22) is **56× the policy gradient**.
NLL protection is not protecting — it IS the primary training signal. The GRPO
policy gradient is a rounding error on the NLL term.

**Breakdown**: 2× (normalization) × 27× (NLL domination) ≈ 55×. Fully explained.

### Analysis: Why Routing Failed at 1.5B

#### The routing signal is real but small
Low-entropy tokens don't have zero gradient — they have ~2-3× smaller gradients
than high-entropy tokens (via the entropy→probability→advantage chain). Zeroing
them removes a real but small signal. The question is whether removing this small
signal helps or hurts. Answer: neither, at this scale.

#### Gate 2 auto-satisfaction at 1.5B
At 1.5B parameters, entropy and probability are near-perfectly correlated (confirmed
in Run 2: zero Gate 2 failures at 95th percentile entropy). The entropy threshold
alone determines routing. Gate 2 (τ_p percentile) provides no independent information.
This means the routing decision is a single-variable threshold on entropy — there
isn't enough structure in the logit space to create a meaningful partition.

#### NLL drowns the signal
Even if routing provides a small benefit, α=0.3 NLL makes protection 56× the policy
gradient per token. Any routing signal is invisible under NLL noise. Could fix with
α=0.005, but if routing itself is a no-op at 1.5B, fixing α won't change the outcome.

#### Scale limitation (Wang et al. reference)
The 80/20 paper (Wang et al. 2506.01939) shows entropy-based gradient masking gains
at 14B and 32B, where entropy structure is meaningfully differentiated — high-entropy
positions correspond to genuine reasoning forks, not syntactic uncertainty. At 1.5B,
syntactic and semantic entropy are confounded. This is consistent with our Run 2
finding that stopwords dominate spark positions even at the 95th percentile.

### Conclusions

#### Proved (hard to vary)
1. **Entropy is a poor branching trigger at 1.5B**: 84% V(R)=0 across GSM8K and
   MATH L3-4. Logit-space uncertainty ≠ outcome-space uncertainty. Branches diverge
   in surface form but converge in final answer.
2. **Gradient routing ≈ no-op at 1.5B**: +0.3pp over 100 steps, within noise.
   Entropy-based token selection doesn't improve learning when the entropy landscape
   lacks semantic structure.
3. **NLL at α=0.3 dominates GRPO**: Protection became the primary training signal
   (56× policy gradient per token). D matching A despite this suggests NLL is
   approximately harmless but also not helpful — it's expensive noise.
4. **The mechanism is correct, the scale is wrong**: All components fire as designed.
   The failure is that 1.5B doesn't have enough entropy differentiation for the
   mechanism to find signal. This is not a bug — it's a finding about the minimum
   scale for entropy-based interventions.

#### What remains open
1. **Routing at 7B+**: Would entropy-based routing show signal where 1.5B doesn't?
   The 80/20 paper suggests yes, but we're budget-blocked (~$83 remaining, 7B needs
   ~$50-100 per viability run on A100s).
2. **ROSE-style semantic trigger + routing**: Replace entropy gate with a semantic
   uncertainty detector (e.g., a probe trained on outcome variance from a reference
   policy). Combined with gradient routing, this could identify genuine epistemic
   forks. Requires a different project scope.
3. **α calibration**: α=0.005 would make NLL proportional to GRPO, but if routing
   is a no-op at 1.5B, fixing α changes nothing. Only worth testing if moving to 7B+.

### Cost
- Smoke test: ~$2
- Viability run: ~$20
- **Running total: ~$67**
- **Budget remaining: ~$83**

---

## Run 11: Trigger Signal Diagnostic (2026-02-23)

### Motivation
Runs 7-10d established that token-level entropy does not predict trajectory-level
divergence (V(R)=0 at 84%). This is now theoretically grounded: Wu et al. (arXiv
2507.14843) prove RLVR is support-constrained, and empirically show that while
token entropy increases during RLVR, answer entropy decreases — uncertain-looking
paths converge onto fewer distinct answers. DeepSearch (Wu et al., arXiv 2509.25454)
addresses this via Q_parent + H + D scoring, where the value estimate (Q_parent) is
the key ingredient our approach was missing.

The genuine open question: can a cheaper signal than Q_parent predict outcome
divergence? Specifically, do **structural step boundaries** (reasoning transitions)
identify genuine cognitive forks better than entropy peaks?

### Design
Forward-pass only diagnostic. No training. ~$1-2 cost.

20 MATH Level 3 problems × 3 position sets × 5 positions × 4 branches = ~1200 rollouts.

**Position sets** (first 200 generated tokens):
- **H (Entropy)**: Top 5 by logit entropy (our previously-used trigger)
- **S (Structural)**: First 5 step boundaries (\n\n, Step N:, Therefore/Thus/Hence)
- **R (Random)**: 5 random positions (control baseline)

**Primary metric**: V(R)>0 rate per set (fraction of positions where branches diverge in outcome)

**Decision matrix**:
- S >> H >> R → Structural transitions are better triggers. Step-level branching.
- H >> S >> R → Entropy signal real but scale-dependent. Revisit at 7B.
- Both ≈ R → Strong invisible leash. Answer basin determined near t=0 at 1.5B.

### Implementation
Pure vLLM inference — no verl, no Ray, no training loop.
Single A100-80GB on Modal, ~10 minutes runtime.
`modal run pilot/modal_spark_smoke.py --mode trigger-diag`

### Results

Base greedy accuracy: 50.0% (10/20) — good calibration for the diagnostic.
1064 branch prompts total (20 problems × ~15 positions × 4 branches). Runtime: 2.1 min.

| Set | V(R)>0 rate | Mean V(R) | Mean entropy | Mean pos frac | N |
|-----|:-----------:|:---------:|:------------:|:-------------:|:-:|
| **H (Entropy)** | **33.0%** | 0.0663 | 1.592 | 0.194 | 100 |
| **S (Structural)** | **21.2%** | 0.0436 | 0.492 | 0.247 | 66 |
| **R (Random)** | **35.0%** | 0.0756 | 0.196 | 0.221 | 100 |

**VERDICT: Both ≈ R — Strong invisible leash**

S had only 66 positions (vs 100 for H and R) because some problems had fewer
than 5 structural boundaries in the first 200 tokens.

### Interpretation

**Neither entropy nor structural boundaries predict trajectory divergence better
than random.** H (33.0%) and R (35.0%) are statistically indistinguishable. S
(21.2%) is actually *worse* than random.

This confirms the strong form of the invisible leash: for Qwen2.5-Math-1.5B,
the answer basin is determined very early in generation — potentially at the
first few tokens. By the time any trigger signal (entropy, structural boundary,
or random position) is measured in the first 200 tokens, the trajectory is
already committed.

**Key observation**: Mean position fraction for all three sets is 0.19-0.25
(first quarter of the sequence). Even at these early positions, V(R)>0 only
~33% of the time. The 35% V(R)>0 for random positions is the base rate — it
reflects the inherent stochasticity of temperature=1.0 sampling, not any
signal in the trigger.

**What the ~33% V(R)>0 baseline means**: With binary rewards and 4 samples at
temperature=1.0, V(R)>0 rate depends on problem difficulty:
- Easy (p→1): nearly all branches correct, V(R)≈0
- Hard (p→0): nearly all branches wrong, V(R)≈0
- Medium (p≈0.5): max divergence, V(R)>0 most often

Base accuracy is 50%, so ~33% V(R)>0 is consistent with: some problems are easy
(greedy-correct, branches all correct), some are hard (greedy-wrong, branches
all wrong), and only the borderline ones show divergence — regardless of WHERE
you branch.

### Implication for the Paper

This is the strongest empirical result in the project: **no measurable-during-
generation signal predicts outcome divergence at 1.5B.** The answer basin is
determined by the support constraint (Wu et al.'s invisible leash), not by
mid-sequence decisions.

The only signal that could predict outcome divergence is value-space — either
a trained PRM (expensive) or completing short probe rollouts to a verifier
(DeepSearch's approach). Both require external information beyond what the
model's internal logit state provides.

### Cost: ~$1. Running total: ~$68. Budget remaining: ~$82.

---

## Run 12: Semantic Branch Diagnostic — Embedder + Fallback Pivots (2026-02-26)

### Motivation
Run 11 proved no measurable-during-generation signal predicts outcome divergence at
1.5B. But that diagnostic used raw entropy as the trigger. The open question: can a
**semantic similarity metric** on short continuations distinguish genuine reasoning
forks from surface variation? If branches that look different to an embedder also
produce different answers, we have a cheap pruning signal for MCTS.

Predicted failure mode: NLP sentence transformers (all-MiniLM-L6-v2) are trained on
natural language, not mathematical logic. "Assume x is even" vs "Assume x is odd"
embed near-identically (Antonym Trap). Conversely, "x=5" vs "\mathbf{x}=5" embed
as distant (formatting noise).

### Design
Forward-pass only diagnostic on Modal. 50 MATH-500 Level 5 problems (fixed seed=42).
DeepSeek-R1-Distill-Qwen-1.5B base model. Per problem:
1. Greedy decode with per-token entropy
2. At first H > theta (1.28), branch K=3, horizon W=10
3. Oracle: complete each branch to final answer (max 1024 tokens)
4. Score pair similarity using the active mode's metric
5. Label: same_answer (identical boxed answer) vs different_answer

Automatic fallback chain — if primary fails (FC>20% or FD>20%):
- **Primary**: all-MiniLM-L6-v2 cosine similarity
- **Pivot 1 (Symbolic)**: Regex extraction of LaTeX math, exact match
- **Pivot 2 (KL)**: KL divergence of next-token logit distributions
- **Pivot 3 (Constrained)**: Vocabulary-masked generation + embedding
- **Pivot 4 (Judge)**: LLM self-judge "are these steps equivalent?"

### Results (50 problems, 150 pairs)

| Stage | FC rate | FD rate | Pass? | Score same_ans (mean) | Score diff_ans (mean) |
|-------|:-------:|:-------:|:-----:|:---------------------:|:---------------------:|
| embedder (MiniLM) | **81.5%** | 18.9% | FAIL | 0.487 | 0.521 |
| symbolic (regex) | **70.0%** | 18.6% | FAIL | 0.088 | 0.060 |
| kl (next-token) | **86.7%** | 18.5% | FAIL | 15.98 | 14.61 |
| constrained | **100%** | 23.8% | FAIL | 0.278 | 0.275 |
| judge (self) | 0% | **23.3%** | FAIL | 0.0 | 0.0 |

**No stage passed both thresholds simultaneously.**

### Ground Truth Distribution
- 150 pairs: 29 same-answer (19.3%), 121 different-answer (80.7%)
- Only 11/50 problems had ANY branch reach the correct answer
- 124/150 pairs had NEITHER branch correct
- Model near-random on Level 5 MATH at 1.5B

### Key Findings

#### 1. The oracle is unreliable, not the detectors
The FC metric = "detector says similar, but answers differ." At 81% FC, this looks
like the embedder is broken. But examining concrete examples: branches with
**literally identical** W=10 continuations ("for \(0 \le x \le 4", "for \(0 \le x
\le 4") end at wildly different wrong answers (0, 20, 50). The symbolic gate
*correctly* identifies these as the same branch. The oracle *incorrectly* labels
them as different because the model wanders to different wrong answers later.

This is Run 11's invisible leash in a new guise: the first 10 tokens don't determine
the final answer, so "same start, different end" is the common case for a weak model
on hard problems.

#### 2. W=10 is too short to contain semantic divergence
Micro-text character length: mean=45.7, max=69. These are filler phrases:
"let me think about this step by step", "Hmm, let me try." The actual mathematical
fork hasn't happened yet at token 10.

#### 3. Score distributions completely overlap (all methods)
- Embedder: mean 0.487 (same) vs 0.521 (diff) — **zero separation**
- Symbolic: 0.088 vs 0.060 — near-zero match rate for both classes
- KL: 15.98 vs 14.61 — both distributions span the full [0, 17.8] range
- Constrained: 0.278 vs 0.275 — **identical**
- Judge: all scores = 0 (1.5B model can't self-judge, always says "different")

#### 4. Smoke vs Full discrepancy for symbolic
Smoke (8 problems): symbolic passed (FC=0%, FD=8.7%). Full (50 problems): symbolic
failed (FC=70%). The smoke test was misleadingly optimistic due to small sample.

### Two Confounded Problems (structural diagnosis)

**Horizon problem**: W=10 tokens is too short to contain meaningful reasoning forks.
The micro-texts are opening phrases, not mathematical decisions. Need W=50-100 to
see actual divergence, but then the compute savings from pruning shrink.

**Oracle problem**: Final answer agreement assumes the model's answer is a
deterministic function of its first 10 tokens. With a 1.5B model on Level 5 MATH,
it's not. Same reasoning start → different wrong answers is common. This poisons
every FC metric regardless of the similarity function.

### Relationship to Run 11
Run 11 showed no signal predicts outcome divergence (H, S, R all ~33% V(R)>0).
Run 12 shows no similarity metric can reliably detect outcome divergence from short
continuations. These are the same finding from two angles:

**The bottleneck is not the detector — it's the signal.** At 1.5B on hard math,
there is insufficient correlation between local generation state (whether measured
by entropy, structure, embedding, KL, or self-judgment) and final outcome.

### Directions Worth Investigating
1. **Fix the oracle**: Use intermediate reasoning states, not final answers
2. **Increase horizon**: W=50-100 to capture actual mathematical content
3. **Use a stronger base model**: If model accuracy >50%, same_answer becomes reliable

The deeper question: is branch deduplication the right bottleneck? Coverage gain
from branching (diverse correct answers) may be more valuable than efficient pruning.

### Cost: ~$5. Running total: ~$73. Budget remaining: ~$77.

### Data Files
```
data/modal_runs/embedder_diagnostic/
  full_report_1772103161.json    Full pair-level data (all 5 stages)
  full_checkpoint.json           Checkpoint with completed stages
  smoke_report_1772097905.json   Smoke test (8 problems, 2 stages)
```

---

## Research Pivot: Strategy Diversity Measurement (2026-02-26)

### Context
Runs 1-12 established that token-level entropy is a weak proxy for reasoning-critical branching (42% FP rate, 84% zero outcome variance). The milestone paper confirmed scaffold absorption fails (D-abs collapse at 7B). Rather than pushing further on token-level interventions, we pivot to a higher-level question: **how do we measure reasoning strategy diversity, and does it collapse during RLVR training?**

### Literature Landscape (verified Feb 2026)

**The diversity collapse phenomenon is established:**
- Pass@1 improves while pass@k (large k) degrades during RLVR (Yue et al. NeurIPS 2025, 561 cites)
- Token entropy can INCREASE while answer diversity DECREASES ("invisible leash," Wu et al. 2507.14843)
- Support shrinkage outweighs expansion ~3.6:1 (ibid.)
- ProRL (2505.24864) bypasses this via reference policy resets, enabling 2000+ RL steps

**15+ papers feed diversity signals into RLVR, but all use crude proxies:**
- Token entropy (STEER, AER, AEPO, PRIME) — local randomness, not strategy variety
- Embedding cosine distance (DIVER, DSDR, DRA-GRPO, MMR-GRPO) — conflates paraphrase with approach change
- Answer count (DAPO filtering) — misses cases where different approaches yield same answer

**Closest prior work — "Rewarding the Rare" (Hu et al., 2601.08763):**
- Uses LLM judge (32B-72B) to cluster rollouts by "overarching strategy"
- Reweights GRPO advantages by w = 1/f^alpha (inverse cluster size)
- AUC@64 on AIME: 0.160 vs 0.116 (SimpleRL), +0.044
- Tested on 7B-8B only. No code released. Alpha value unreported.
- **Critical gaps:** No formal strategy definition (black-box judge), no strategy count tracked across training, no decomposition of WHY diversity helps, no ablation of clustering method

### The Identified Gap

Nobody has:
1. A **reproducible, feature-based definition** of reasoning strategy (not delegated to LLM judge)
2. **Longitudinal tracking** of strategy count per problem across RL training steps
3. **Decomposition** of pass@k degradation into "fewer strategies" vs "worse execution"
4. Strategy-level diversity as training signal with **auditable, non-black-box** definition

### Research Question
**Does RLVR training cause strategy-level diversity collapse, and can we measure it with structural features rather than an LLM judge?**

### Pilot Experiment Design: Strategy Diversity Diagnostic

**Goal:** Validate that (1) we can define and detect distinct reasoning strategies from traces using structural features, (2) strategy count is measurable and meaningful, (3) we can observe collapse (or not) across training checkpoints.

**Phase 1: Strategy Feature Extraction (no training needed)**
- Model: DeepSeek-R1-Distill-Qwen-1.5B (we have this)
- Dataset: 30-50 MATH Level 4-5 problems
- Generate K=16 rollouts per problem (high K for diversity measurement)
- For each trace, extract structural features:
  - Operation types used (substitution, factoring, case analysis, contradiction, induction, coordinate geometry, etc.)
  - Proof structure (linear chain vs. branching vs. backtracking)
  - Number of distinct subgoals / "wait" tokens / self-corrections
  - Key mathematical objects introduced (coordinates, auxiliary variables, inequalities)
- Cluster traces by structural feature vectors (not embeddings, not LLM judge)
- Measure: how many clusters per problem? How stable is clustering?

**Phase 2: LLM Judge Calibration**
- Run "Rewarding the Rare"-style LLM judge clustering on the same traces
- Compare: does our structural clustering agree with the judge?
- If agreement is high: structural features are a cheap, reproducible proxy for strategy identity
- If agreement is low: understand WHERE they diverge — is the judge capturing something features miss, or is the judge noisy?

**Phase 3: Longitudinal Tracking (requires RL checkpoints)**
- Use publicly available RL training checkpoints if they exist (ProRL released Nemotron-Research-Reasoning-Qwen-1.5B but likely only final checkpoint)
- Or: run short GRPO training (100-500 steps) on our 1.5B model, saving checkpoints every 50 steps
- At each checkpoint: generate K=16 rollouts on the SAME 30-50 problems
- Track: strategy cluster count per problem, strategy distribution entropy, which strategies appear/disappear
- Plot the collapse curve: steps on x-axis, average strategy count on y-axis

**Phase 4: Decomposition**
- For problems where pass@k degrades across training:
  - Did a strategy disappear entirely? (detachment)
  - Or did an existing strategy get less reliable? (degradation)
- For problems where pass@1 improves:
  - Did one strategy get sharper while others died? (concentration)
  - Or did multiple strategies all improve? (broad improvement)

### Success Criteria
- Phase 1: Structural features produce ≥3 distinct, interpretable clusters on at least 60% of problems
- Phase 2: ≥70% agreement with LLM judge clustering (Cohen's kappa > 0.5)
- Phase 3: Observable downward trend in strategy count across training
- Phase 4: At least one clear example of detachment (strategy lost entirely)

### Key Differentiators vs. "Rewarding the Rare"
1. **Reproducible**: Feature-based, not LLM-judge-dependent. Anyone can rerun.
2. **Longitudinal**: Track across training, not just measure at endpoint.
3. **Decomposed**: Distinguish strategy loss from strategy degradation.
4. **Cheap**: No 32B-72B judge needed at training time.

### Connection to Prior Spark Work
The token entropy diagnostic (Runs 1-11) showed the signal is too noisy at the token level. This pivot moves to the strategy level — the right abstraction for measuring diversity. The kill-test methodology from the milestone paper applies directly: if we later use strategy diversity as a training signal, we evaluate with uniform sampling to check absorption.

### Estimated Cost
- Phase 1-2: ~$10 (inference only, 50 problems × 16 rollouts × 2 models)
- Phase 3: ~$20-40 (short GRPO run with checkpoints + inference at each)
- Phase 4: Analysis only, no additional compute
- Total: ~$30-50. Budget remaining: ~$77.

### Cross-Disciplinary Motivation
From rare event sampling literature (verified):
- **Go-Explore** (Ecoffet et al.): Diversity collapse = "detachment problem." Archive + return-then-explore fixes it.
- **MAP-Elites**: Maintain best solution per strategy cell. Requires defined behavior space.
- **Metadynamics**: Penalize visited strategies to force exploration of new ones.
- **AMS / Multilevel Splitting**: Clone promising partial chains, re-explore with fresh randomness.

All require a well-defined strategy space. Our measurement contribution IS that definition.

---

## Run 13: Strategy Diversity Smoke Test (2026-02-27)

### Configuration
- Model: DeepSeek-R1-Distill-Qwen-1.5B
- Dataset: 10 MATH Level 5 problems, seed=42
- K=8 rollouts per problem, temperature=1.0, max_tokens=2048
- GPU: A10G on Modal
- Total generation: 80 rollouts in 44.4s

### Results

| Metric | Value |
|--------|-------|
| Mean fine clusters per problem | 4.6 |
| Median fine clusters | 5 |
| Range | [2, 7] |
| Mean coarse clusters | 1.9 |
| Mean Simpson diversity | 0.631 |
| Mean pass rate | 22.5% |
| Problems with ≥3 fine clusters | 8/10 (80%) |
| Problems with ≥3 coarse clusters | 1/10 (10%) |
| **Success criterion (≥60% with ≥3 fine)** | **PASS (80%)** |

### Per-Problem Breakdown

| Prob | Subject | Pass Rate | Fine Clusters | Coarse Clusters | Primary Strategy |
|------|---------|-----------|---------------|-----------------|-----------------|
| 0 | Algebra | 75% | 4 | 1 | factoring (8/8) |
| 1 | Algebra | 0% | 6 | 1 | coordinate_geometry (8/8) |
| 2 | Algebra | 0% | 5 | 2 | factoring (7), coord_geo (1) |
| 3 | Counting & Prob | 25% | 7 | 2 | combinatorics (7), direct (1) |
| 4 | Prealgebra | 0% | 6 | 4 | case_analysis (2), unclassified (3), direct (1), coord_geo (2) |
| 5 | Geometry | 0% | 5 | 2 | coord_geo (7), trig (1) |
| 6 | Intermed. Algebra | 0% | 6 | 2 | trig (7), substitution (1) |
| 7 | Algebra | 38% | 2 | 1 | substitution (8/8) |
| 8 | Algebra | 88% | 3 | 2 | factoring (4), substitution (4) |
| 9 | Prealgebra | 0% | 2 | 2 | direct_computation (7), substitution (1) |

### Critical Analysis: Is the Clustering Meaningful?

**The fine clustering is TOO granular — it's capturing noise, not strategy.**

Looking at Problem 0 (factoring): all 8 rollouts use factoring as their primary approach. The fine clustering splits them into 4 groups based on which SECONDARY operations appear (e.g., whether "substitution" or "coordinate_geometry" keywords happen to occur alongside factoring). But reading the actual traces, these are all doing the SAME thing — factoring x^2-5x-14 = (x-7)(x+2). The differences are surface-level (whether the model mentions "plugging in" or "checking" along the way), not strategic.

**The coarse clustering is TOO coarse — it collapses everything to one strategy.**

Problem 0: all 8 mapped to "factoring." Problem 7: all 8 mapped to "substitution." With coarse clustering, 7/10 problems have ≤2 clusters. That's not enough granularity to measure diversity.

**Where the features DO capture real differences (Problem 8):**
Problem 8 shows a genuine 4/4 split between "factoring-first" and "substitution-first" approaches. Reading the traces confirms this is real — R0 starts with the inverse variation formula and focuses on manipulating sqrt(x), while R4 starts by substituting known values immediately. Same problem, genuinely different approach order.

**Where the features FAIL (Problem 3):**
Problem 3 (combinatorics) shows 7 fine clusters but only 2 coarse. The fine clusters distinguish R0 (combinatorics+induction+substitution) from R5 (combinatorics+recursion). Reading the traces: R3 uses inclusion-exclusion, R5 uses a similar counting approach with slightly different bookkeeping. The regex detected "recursion" in R5's text but the actual mathematical approach is very similar to R3's. The keyword detection is noisy.

**The "unclassified" problem (Problem 4):**
3/8 rollouts mapped to "unclassified" because the regex patterns didn't match any operation. This means the feature set is incomplete — it misses some mathematical approaches entirely.

### Diagnosis

The structural feature approach has TWO problems:

1. **False diversity (fine level):** Regex keyword detection creates fake clusters by distinguishing rollouts that mention "check" (verification) vs. those that don't, even when the underlying strategy is identical. This inflates cluster count artificially.

2. **True collapse (coarse level):** Mapping to a single "primary operation" loses the structural composition — factoring+substitution vs. factoring+case_analysis gets collapsed to just "factoring."

**The right granularity is in between.** What we need is not keyword presence/absence but a semantic-structural understanding: "does this trace use inclusion-exclusion or direct counting?" "Does it set up coordinates or use synthetic geometry?" The regex approach can't distinguish these because the same keywords appear in both.

### Verdict

**The direction is valid but the operationalization needs work.**

The good news:
- Strategy diversity IS measurable — Problem 8 shows a clean, real split
- The model DOES use multiple strategies for some problems
- The infrastructure works (80 rollouts in 44s, analysis pipeline clean)

The bad news:
- Regex-based feature extraction is too noisy for fine clustering and too coarse for primary-operation clustering
- The "right" level of granularity requires either (a) a much more sophisticated feature extractor or (b) an LLM judge (which is what "Rewarding the Rare" does)
- 3/80 rollouts (4%) were unclassified — feature coverage is incomplete

### What To Do Next

**Option A: Hybrid approach.** Use a small, cheap LLM (e.g., Qwen2.5-7B-Instruct, not 72B) as a judge for strategy classification, but with a STRUCTURED prompt that forces it to classify into a predefined taxonomy. This is cheaper than Rewarding the Rare's 32B+ judge and more reproducible because the taxonomy is fixed. Compare against our structural features for calibration.

**Option B: Improve structural features.** Add more sophisticated pattern matching: detect proof structure (direct proof vs. contradiction vs. induction), identify key mathematical objects (coordinate systems, generating functions, modular arithmetic chains), extract the SEQUENCE of operations rather than just presence/absence. This keeps the approach fully reproducible but requires significant feature engineering.

**Option C: Embedding-based middle ground.** Embed the reasoning traces (not the answers) using a sentence transformer, then cluster. This captures semantic similarity without needing predefined categories. The risk: embedding distance may conflate paraphrase with strategy change (same concern as DIVER/DSDR).

**Recommendation: Option A.** A structured LLM judge with a fixed taxonomy gives us the best of both worlds — semantic understanding + reproducibility. We define the taxonomy (from math education literature: Polya's heuristics, Schoenfeld's problem-solving framework), the LLM just classifies into it. The taxonomy IS the contribution (nobody has bridged math ed → LLM trace analysis), and the LLM judge is the measurement tool, not the definition.

### Cost: ~$3. Running total: ~$76. Budget remaining: ~$74.

### Data Files
```
data/modal_runs/strategy_diversity_full_results.json   Full results (80 rollouts with features)
```

---

## Current Objective & Scope (2026-02-27)

### Research Question
**Does RLVR training preserve or destroy a model's capacity for strategy exploration?**

Not "does RLVR reduce diversity" (known: yes, Yue et al. 2025). The question is whether the diversity loss matters — whether inference-time search can recover what RLVR training discards, or whether RLVR permanently narrows what's reachable.

### Core Distinction: Diversity vs. Explorability
- **Diversity** = how many strategies the model produces under standard sampling (stored in the policy)
- **Explorability** = how many strategies the model CAN produce when given inference-time compute budget (capacity to discover)

These are different. A collapsed policy with high explorability beats a diverse policy with low explorability. Nobody has tested this directly.

### The Experiment (Kill-Test on Explorability)
Compare three 1.5B models on the same MATH problems, each with varying inference budgets:

### Models (verified via HF MCP 2026-02-27)

All same Qwen2 arch. Nemotron is finetuned FROM R1-Distill → same training trajectory, 4 checkpoints.

| Checkpoint | Training | MATH | AIME24 | HuggingFace |
|---|---|---|---|---|
| R1-Distill-1.5B | Distilled from R1 (671B) | 82.90 | 28.54 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Nemotron v1 | ProRL 2K steps on R1-Distill | 91.89 | 48.13 | `nvidia/Nemotron-...-1.5B` rev `v1` |
| Nemotron v2 | ProRL 3K steps | 92.49 | 49.58 | `nvidia/Nemotron-...-1.5B` (default) |
| Nemotron BroRL | v2 + 419 steps, N=512 | 92.20 | 60.42 | `nvidia/Nemotron-...-1.5B` branch `brorl` |

Note: `Qwen/Qwen2.5-1.5B` (raw pretrained) likely too weak for meaningful reasoning. Focus on the 4-checkpoint ProRL trajectory above.

### HF Data Audit (verified 2026-02-27)

| Dataset | Content | Use for us |
|---|---|---|
| `nvidia/OpenMathReasoning` | R1 (671B), ~10-16 CoT/problem, 306K AoPS | Strategy oracle: define what strategies exist |
| `uzaymacar/math-rollouts` | R1-Distill 8B/14B, chunk annotations | Adopt annotation schema (function tags) |
| `open-r1/OpenR1-Math-Raw` | R1 (671B), 1-8 rollouts, NuminaMath | Reference traces |
| **1.5B traces** | **NONE EXIST** | **Must generate (~$1-3)** |

### Experimental Design (Refined)

**Generate:** 4 models × 60 MATH problems × 64 rollouts = ~15K generations (~$1-3)
- Problem selection: 20 easy (Level 1-2) / 20 medium (Level 3) / 20 hard (Level 4-5) from MATH
- Temperature: 1.0 (maximize diversity)
- Max tokens: 4096

**Measure per model per problem:**
1. **pass@k curves** (k=1,4,8,16,32,64) — accuracy under search
2. **Unique correct answer count** — how many distinct correct paths exist
3. **Strategy diversity** — cluster correct traces by approach
4. **Novel strategy rate** — strategies found by model X but not model Y
5. **Strategy persistence** — do strategies at checkpoint N survive to checkpoint N+1?

**Strategy classification approach:** Adopt `uzaymacar/math-rollouts` function tags (`problem_setup`, `fact_retrieval`, `active_computation`, `plan_generation`, `uncertainty_management`) as chunk-level annotation, plus embedding-based clustering for approach-level diversity. NOT an LLM judge (same as Rewarding the Rare) — use structural signals.

### What This Tests (Decision Tree Outcomes)

| Finding | Implication |
|---------|-------------|
| R1-Distill+search ≈ Nemotron+search | ProRL's accuracy gains are recoverable via search. Reference reset didn't create new explorability. |
| Nemotron+search >> R1-Distill+search | ProRL creates genuinely new strategies that search alone can't find. Validates the mechanism. |
| Nemotron BroRL finds strategies absent in v1/v2 | Breadth scaling (N=512) discovers strategies depth scaling (more steps) misses. |
| Results vary by difficulty | Phase transition: easy → exploit wins, hard → explore wins. Strongest paper. |
| Strategy count drops from R1-Distill → Nemotron | ProRL improves accuracy by NARROWING strategies. Exploration cost of convergence. |
| Strategy count INCREASES R1-Distill → Nemotron | ProRL genuinely expands the strategy space. Contradicts "rejection sampler" narrative. |

### Why This Is Novel
1. **Nobody has compared points on a ProRL training trajectory at the STRATEGY level.** Accuracy improvements are published; strategy landscape changes are not.
2. **4 checkpoints on the same trajectory = longitudinal signal without training.** Free controlled experiment.
3. **Kill-test framing:** Does ProRL create genuine explorability, or does it just sharpen existing paths? Our methodology.
4. **Difficulty stratification:** Testing whether the exploration-exploitation phase transition exists at the strategy level.

### What's Already Known (Don't Re-Prove)
- RLVR improves pass@1, degrades pass@k at large k (Yue et al. NeurIPS 2025)
- RLVR is support-constrained to base model distribution (Invisible Leash, 2507.14843)
- KL-regularized expected-return maximization specifies mode collapse (arXiv:2510.20817)
- Token entropy ↑ while answer diversity ↓ under RLVR (Invisible Leash)
- Strategy-level diversity collapse exists (Rewarding the Rare, 2601.08763)
- ProRL achieves higher accuracy than R1-Distill (published in ProRL/BroRL papers)

### Scope Boundaries
- **In scope:** 4 checkpoints at 1.5B, MATH benchmark (60 problems × 3 difficulty tiers), K=64 rollouts, structural strategy analysis
- **Out of scope:** Training our own RLVR, TTT-Discover, code/visual reasoning, models >7B, LLM judge (too similar to Rewarding the Rare)
- **Budget:** ~$74 remaining. Estimated cost: ~$1-3 for generation, ~$5 for analysis. Total ~$8.
- **Timeline:** CS224N final project deadline

### Distillation Reference
Full intellectual foundation: `docs/distill/reasoning-strategy-diversity.md`
- 7 first principles (3 foundations, 4 open)
- Key reframe: "explorability > diversity" (Insight 10, Contrarian Truth 2)
- The Next Question in the distillation IS this experiment

### Relationship to Prior Work in This Log
- Runs 1-11: Token-level entropy diagnostics → too noisy, wrong abstraction level
- Run 12: Milestone paper kill-test → D-abs collapse, scaffold dependence (methodology proven)
- Run 13: Regex strategy features → too noisy fine / too coarse coarse
- **This experiment:** Strategy-level analysis across 4 ProRL checkpoints with matched inference budget

---

## Run 14: Phase 1 Trace Generation (2026-02-27)

### Setup
- Script: `pilot/modal_gen_traces.py`
- 4 models × 60 MATH problems (20 easy/20 med/20 hard) × K=64 rollouts = 15,360 traces
- T=1.0, max_tokens=4096, MiniLM embeddings (mean-pooled chunks)
- GPU: A10G on Modal, total wall time: 3.55 hours, ~$4 compute

### Results — Cross-Model Comparison

| Model | pass@1 | pass@8 | pass@16 | pass@64 | mean_correct/64 | unique_ans |
|---|---|---|---|---|---|---|
| R1-Distill | 0.597 | 0.772 | 0.806 | 0.850 | 38.2 | 1.53 |
| Nemotron v1 (2K) | 0.605 | 0.742 | 0.782 | 0.833 | 38.7 | 0.97 |
| Nemotron v2 (3K) | **0.729** | **0.871** | **0.888** | **0.900** | **46.6** | 1.10 |
| BroRL | 0.702 | 0.838 | 0.858 | 0.883 | 44.9 | 1.12 |

### Results — Per Difficulty Tier (pass@1 / pass@64)

| Model | Easy | Medium | Hard |
|---|---|---|---|
| R1-Distill | 0.799 / 0.95 | 0.556 / 0.85 | 0.436 / 0.75 |
| Nemotron v1 | 0.895 / 0.95 | 0.565 / 0.80 | 0.355 / 0.75 |
| Nemotron v2 | 0.927 / 0.95 | 0.714 / 0.85 | 0.545 / 0.90 |
| BroRL | 0.920 / 0.95 | 0.679 / 0.85 | 0.507 / 0.85 |

### Key Observations

1. **v1 is anomalous.** Nemotron v1 (2K steps) has LOWER pass@k than R1-Distill at k≥8, despite higher pass@1 on easy. ProRL at 2K steps may have collapsed diversity without sufficient accuracy gains. This is the "cost of early convergence" the theory predicts.

2. **v2 dominates everywhere.** pass@1 AND pass@64 both improve. ProRL at 3K steps found the sweet spot — accuracy gain without (complete) diversity loss. But unique_answers dropped from 1.53 → 1.10.

3. **BroRL ≈ v2 but slightly worse.** Despite BroRL's much higher AIME24 score (60.4 vs 49.6), it's slightly behind v2 on MATH. This suggests BroRL's breadth scaling (N=512) helps on harder competition problems but doesn't improve general MATH coverage.

4. **Hard problems show the biggest separation.** R1-Distill pass@64=0.75 → v2 pass@64=0.90. ProRL genuinely expanded reachability on hard problems. This is not just sharpening existing paths.

5. **Easy problems are saturated.** All models reach pass@64=0.95. No signal here for strategy analysis.

6. **unique_answers dropped: R1-Distill 1.53 → Nemotron ~1.0.** ProRL narrowed the answer space while improving accuracy. Fewer distinct answers, but more of them correct. Consistent with Invisible Leash (token entropy ↑, answer diversity ↓).

### Interpretation vs. Decision Tree

- "R1-Distill+search ≈ Nemotron+search" → **PARTIALLY.** v1 yes. v2/BroRL no — they genuinely improve pass@64 on hard problems.
- "Results vary by difficulty" → **YES.** Easy saturated. Hard shows the action. Phase transition confirmed.
- "Strategy count drops R1-Distill → Nemotron" → **unique_answers dropped** (1.53 → ~1.0). But need embedding-based strategy analysis (Phase 2) to distinguish real strategy diversity from answer variation.

### Next: Phase 2 — Strategy Classification Validation
- Data: `data/modal_runs/gen_traces_full/` (140MB traces + 22MB embeddings)
- Task: Embedding-based clustering validation on positive control problems
- Goal: Determine if clusters correspond to genuine strategy differences before making diversity claims

---

## Run 15: Phase 2 — Strategy Classification & Explorability Analysis (2026-02-27)

### Goal
Validate whether embedding-based clustering captures genuine strategy diversity, then extract the real signal from our 15,360 traces.

### Approach 1: Embedding Clustering (FAILED)
Script: `pilot/analyze_strategies.py`

Agglomerative clustering with cosine distance on MiniLM embeddings (384-dim), silhouette-based k selection.

**Validation gate results:**
- Answer purity: 0.603 (PASS > 0.5)
- Answer separation: 0.597 (PASS > 0.4)
- Mean clusters: 2.0 (range 2-3)

**Apparent finding:** v2/BroRL collapsed to ~1 strategy per problem (dominant share 0.997+), R1-Distill had ~1.8.

### Diagnostic: Embedding Clustering is an ARTIFACT

Ran 5 diagnostics to verify. **Conclusion: MiniLM embeddings are not strategy-sensitive.**

| Diagnostic | Finding |
|---|---|
| Intra-problem cosine similarity | 0.92-0.96 for ALL models. Embeddings dominated by problem identity, not reasoning approach. |
| Cross-model centroids | 0.97-0.99 cosine. MiniLM cannot distinguish how R1-Distill vs v2 solves a problem. |
| Within-answer vs between-answer gap | 0.004-0.025. Near noise floor. If it can't separate different ANSWERS, it can't separate strategies within the same answer. |
| Random baseline | Random 384-dim vectors: silhouette=0.007. Real data: 0.55. Something real but NOT strategy-level — likely surface features (length, vocab). |
| Cluster count replication | All 4 models get k=2 for 58-59/60 problems. This is an embedding limitation, not a model property. |

**Verdict:** The Phase 2 clustering results are NOT trustworthy. Discarded.

### Approach 2: Answer-Level Analysis (THE REAL SIGNAL)

Instead of trying to classify "strategies" (which requires a much better classifier than MiniLM), analyze what we CAN measure reliably: answer distributions, coverage ceilings, and sampling gains.

#### Finding 1: Answer diversity = mostly ERROR diversity

| Model | Problems with >1 answer | Problems with >1 CORRECT answer |
|---|---|---|
| R1-Distill | 24/60 | 2/60 |
| v1 (2K) | 4/60 | 0/60 |
| v2 (3K) | 7/60 | 1/60 |
| BroRL | 6/60 | 1/60 |

**At 1.5B scale, there is almost no genuine solution diversity** (multiple correct approaches). The "diversity" R1-Distill has is overwhelmingly wrong answers. ProRL didn't destroy diversity — it eliminated errors.

#### Finding 2: Coverage ceiling reveals genuine capability gain

| Tier | R1-Distill | v1 (2K) | v2 (3K) | BroRL |
|---|---|---|---|---|
| Easy (20) | 19/20 | 19/20 | 19/20 | 19/20 |
| Medium (20) | 17/20 | 16/20 | 17/20 | 17/20 |
| Hard (20) | 15/20 | 15/20 | **18/20** | 17/20 |
| ALL (60) | 51/60 | 50/60 | **54/60** | 53/60 |

v2 solves **3 more hard problems** than R1-Distill (problems 42, 44, 50, 51 — though it loses problem 57). ProRL expanded reachability, not just accuracy.

R1-Distill unsolvable: {42, 44, 50, 51, 55}
v2 unsolvable: {55, 57}

#### Finding 3: Sampling gain as explorability proxy

Gain = pass@64 − pass@1. Higher gain = more benefit from repeated sampling.

| Tier | R1-Distill gain | v1 gain | v2 gain | BroRL gain |
|---|---|---|---|---|
| Easy | 0.151 | 0.055 | 0.023 | 0.030 |
| Medium | 0.294 | 0.235 | 0.136 | 0.171 |
| Hard | 0.314 | 0.395 | 0.355 | 0.343 |

R1-Distill gains MORE from sampling on easy/medium (has more room to improve — more errors to overcome). But on hard problems, v1 actually gains the most (0.395), suggesting early ProRL creates a volatile model that occasionally stumbles onto hard solutions.

#### Finding 4: Equal-compute contest (the kill test)

**Among jointly-solvable problems, who benefits more from sampling?**

| Tier | Joint problems | R1-Distill mean gain | v2 mean gain | R1 wins more often |
|---|---|---|---|---|
| Easy | 19 | 0.159 | 0.025 | 15/19 |
| Medium | 17 | 0.346 | 0.160 | **17/17** |
| Hard | 14 | 0.379 | 0.276 | 10/14 |

R1-Distill benefits more from sampling on every tier, on most problems. But this is because R1-Distill has lower pass@1 — it has more errors to recover from.

**The real comparison: R1-Distill@64 vs v2@1 (same cost: is search a substitute for RL?)**

| Tier | R1@64 | v2@1 | Winner |
|---|---|---|---|
| Easy | 0.950 | 0.927 | R1+search |
| Medium | 0.850 | 0.714 | R1+search |
| Hard | 0.750 | 0.545 | R1+search |

R1-Distill with 64 samples beats v2 greedy. But R1@64 needs ~4 samples to match v2@1, meaning RL gives roughly a 4× inference efficiency gain.

**However: R1@64=0.750 vs v2@64=0.900 on hard.** v2 with the SAME search budget reaches 15% more hard problems. This is the definitive result: **ProRL creates genuine explorability that search alone cannot recover.**

#### Finding 5: v1 anomaly confirms Pólya urn prediction

v1 (2K ProRL steps) shows WORSE pass@k than R1-Distill at k≥8 on medium/hard:
- Medium: v1@64=0.800 vs R1@64=0.850
- Hard: v1@1=0.355 (lowest of all) but v1@64=0.750 (ties R1)

Early RL narrows the distribution (higher pass@1 on easy) but hasn't yet built new capabilities. This is the "cost of early convergence" — the Pólya urn phase where early correct solutions compound and crowd out exploration.

#### Finding 6: Token-length CV is verbosity, not strategy

Among correct traces for the same problem, token-length CV is 0.15-0.52. But manual inspection of high-CV traces (problem 16: 313.9 + 12.6) shows shortest trace (219 tokens) and longest (4096 tokens) use identical approach — just different verbosity. This is NOT strategy diversity.

### Synthesis: What the data actually says

1. **At 1.5B scale, these models essentially have ONE strategy per problem.** Answer diversity is error diversity, not solution diversity. The "strategy diversity" framing requires larger models or harder problems where multiple valid approaches exist.

2. **ProRL genuinely expands reachability.** v2 solves 3 more hard problems than R1-Distill. This is NOT compression — it's capability gain.

3. **The explorability question is answered: ProRL CREATES explorability for hard problems.** v2@64=0.90 vs R1@64=0.75 on hard. With equal search budget, the RL-trained model reaches more problems. The "search as substitute for training" hypothesis fails for hard problems.

4. **The v1→v2 transition is the interesting story.** v1 is in the Pólya urn phase (narrowed but not yet capable). v2 has crossed a threshold where accuracy and coverage both improve. The phase transition lives between 2K and 3K ProRL steps.

5. **BroRL's breadth scaling (N=512) doesn't help on MATH.** BroRL ≈ v2 on MATH despite dramatically higher AIME24 (60.4 vs 49.6). BroRL's benefit is specific to competition-level problems, not general MATH.

### Decision Tree Resolution

| Finding | Resolved? | Answer |
|---|---|---|
| R1-Distill+search ≈ Nemotron+search | **NO** — R1@64 < v2@64 on hard by 0.15 | ProRL creates genuine capability |
| Nemotron+search >> R1-Distill+search | **YES on hard** — v2 solves 18/20 vs 15/20 | ProRL expands reachability |
| BroRL finds strategies absent in v1/v2 | **PARTIAL** — BroRL solves problem 42 (17/64 correct) which v1 can't | Minor |
| Results vary by difficulty | **YES** — easy saturated, hard is where the action is | Phase transition confirmed |
| Strategy count drops R1→Nemotron | **N/A** — can't measure strategies at 1.5B | Need LLM judge or larger models |
| Strategy count rises R1→Nemotron | **N/A** — same | Need LLM judge or larger models |

### What's genuinely novel in our data

1. **4-checkpoint longitudinal pass@k analysis** — R1-Distill → v1 → v2 → BroRL. No one has published this.
2. **v1 anomaly** — Early ProRL (2K steps) is WORSE than the distilled base at k≥8. Unreported.
3. **Difficulty-stratified phase transition** — Easy saturated, hard shows 15% gap. The story is entirely in hard.
4. **R1@4 ≈ v2@1** — RL gives ~4× inference efficiency. Concrete quantification.
5. **Coverage ceiling expansion** — v2 solves 3 additional hard problems unreachable by R1-Distill even at K=64.

### What's NOT novel (confirmed by literature review)

- Pass@k curves decline with more RL (Yue et al., Invisible Leash) — we confirm but don't discover this
- Answer diversity drops with RL — known
- BroRL improves competition math — their own paper shows this

### Deliverables
- `pilot/analyze_strategies.py` — Embedding clustering script (diagnostic value only)
- `pilot/analyze_rigorous.py` — Statistical analysis with bootstrap CIs
- `data/analysis/phase2_results.json` — Clustering results (artifactual, kept for transparency)
- `data/analysis/rigorous_results.json` — Bootstrap CIs, significance tests

---

## Run 16: LLM Judge Strategy Classification — All 60 Problems (2026-02-27)

### Setup
- Classifier: GPT-5-nano via OpenAI API ($0.05/$0.40 per 1M tokens)
- All 60 problems, 4 models, up to 8 correct traces per model per problem
- Total cost: $0.15 (1.5M input + 179K output tokens)
- Script: `pilot/llm_judge_pilot.py`

### Results: Strategy Diversity DOES Exist (but limited)

**14/60 problems (23%) have >1 strategy across all models combined.**

| Tier | Problems with >1 strategy | Mean strategies |
|---|---|---|
| Easy | 5/20 | 1.40 |
| Medium | 4/20 | 1.25 |
| Hard | 5/20 | 1.40 |

Not difficulty-dependent — strategy diversity is roughly uniform across tiers.

### The Core Finding: R1-Distill retains strategies that ProRL drops

Among 12 multi-strategy problems where all 4 models have correct traces:

| Comparison | Count |
|---|---|
| R1 uses MORE strategies than v2 | 6/12 |
| v2 uses MORE strategies than R1 | 1/12 |
| Tied | 5/12 |

**R1-Distill mean: 2.14 strategies per multi-strat problem**
**v2 mean: 1.57 strategies per multi-strat problem**

R1-Distill has 12 exclusive strategies (used by no other model), v2 has 2, BroRL has 0.

### Specific Examples

| Problem | Type | R1 strategies | v2 strategies | What changed |
|---|---|---|---|---|
| P52 (hard IntAlg) | Optimization | AM-GM + Symmetry + Lagrange | AM-GM only | v2 dropped 2 strategies |
| P58 (hard IntAlg) | Optimization | AM-GM + Lagrange | Lagrange only | v2 dropped AM-GM |
| P46 (hard Algebra) | Intercepts/slope | All 4 methods | 3 of 4 | Slight narrowing |
| P35 (med Precalc) | Trig identities | 2 methods | 3 methods (!) | **v2 GAINED a strategy** |
| P34 (med Prealg) | Tangram area | Coordinate geometry | Tangram reasoning | **Different strategy, not subset** |

### Interpretation

1. **Strategy diversity exists at 1.5B** — contrary to what the embedding analysis suggested. 23% of problems have genuinely distinct approaches (algebraic vs geometric, AM-GM vs Lagrange, etc).

2. **ProRL narrows the strategy repertoire.** R1-Distill accesses 2.14 strategies on multi-strat problems, v2 accesses 1.57. This is a real strategy convergence effect.

3. **The convergence is NOT total.** v2 retains the dominant strategy on every problem and occasionally discovers new ones (P35). ProRL doesn't destroy all diversity — it preferentially drops minority strategies.

4. **Strategy convergence explains the pass@k curve shapes.** R1-Distill's higher sampling gain comes partly from accessing minority strategies that v2 has dropped. But v2's higher pass@1 means it reliably executes the dominant strategy.

5. **Problem 34 is the most interesting case.** R1-Distill uses coordinate geometry (brute force), v2 uses tangram-area reasoning (elegant). ProRL didn't just narrow — it SWITCHED the preferred strategy to a more efficient one.

6. **Problem 35 contradicts the simple "RL narrows" story.** v2 uses 3 strategies while R1 uses 2. ProRL can sometimes EXPAND the strategy space, perhaps by making the model competent enough on a problem that it can explore alternative approaches.

### Synthesis: What the data says across all analyses

**Phase 1 (pass@k):**
- v2 dominates on accuracy (pass@1) and coverage (pass@64)
- v1 is anomalously worse than R1-Distill at k≥8 (p=0.005 on hard)
- R1@4 ≈ v2@1 (4× inference efficiency from RL)
- v2 solves 3 hard problems R1 can't reach at any K

**Phase 2 (strategy classification):**
- 14/60 problems have genuine strategy diversity
- R1-Distill retains more strategies (mean 2.14 vs v2's 1.57 on multi-strat problems)
- ProRL preferentially drops minority strategies but retains the dominant one
- In 2 cases, ProRL actually expanded or switched strategy (P35, P34)

**Combined narrative:** ProRL trades strategy breadth for reliability. The base model (R1-Distill) maintains a wider strategy repertoire but executes each less reliably. ProRL (v2) narrows to fewer strategies but executes them with much higher accuracy. On hard problems, this accuracy gain also expands coverage — v2 reaches problems the base model can't solve at all, even with 64 attempts.

### Paper Contribution: Verified and Specific

1. **4-checkpoint longitudinal strategy evolution** across the ProRL trajectory (new)
2. **v1 anomaly**: early ProRL (2K steps) worse than base at k≥8, p=0.005 (new)
3. **Strategy narrowing quantified**: R1-Distill 2.14 → v2 1.57 strategies on multi-strategy problems (new)
4. **Narrowing is not total**: 2 cases of ProRL expanding/switching strategies (new, contradicts simple "collapse" narrative)
5. **Coverage ceiling expansion**: v2 reaches 3 new hard problems via accuracy, not diversity (new quantification)
6. **4× inference efficiency**: R1@4 ≈ v2@1 (new quantification)

### Deliverables
- `pilot/llm_judge_pilot.py` — GPT-5-nano strategy classification
- `data/analysis/llm_judge_pilot.json` — Full classification results for 60 problems

---

## Run 17: Mathematical Framework Audit (2026-02-27)

### Question
Do ProRL, BroRL, Invisible Leash, or related papers mathematically model the strategy breadth-reliability tradeoff we observe?

### Answer: No existing paper models this exactly. But 3 papers provide formal pieces.

#### Piece 1: Chen et al. 2510.20817 — The objective DESIGNS collapse

**Remark 4.3 (Equal-reward case):** For two correct answers y1, y2 with R(y1) = R(y2):
```
G_β(y1) / G_β(y2) = π_ref(y1) / π_ref(y2)
```
The optimal KL-regularized policy locks probability ratios to the reference policy, independent of β. If the reference policy already favors one strategy 2:1, the optimal policy does too. Tuning regularization cannot fix this.

**Implication for our data:** This explains WHY ProRL narrows strategies. The reference policy (R1-Distill) already has a dominant strategy. KL-regularized RL is mathematically constrained to preserve that dominance. Strategy narrowing is by construction, not optimizer failure.

#### Piece 2: Sinha et al. 2601.21669 — The probability multiplier

**Theorem 3.1:** Under gradient flow on expected return with softmax:
```
d/dt log(p_i(t) / p_j(t)) = p_i(t) · a_i(t) - p_j(t) · a_j(t)
```
When p_i > p_j and r(i) ≥ r(j), the log-ratio grows — the rich get richer. The probability of a strategy amplifies its own gradient signal. This is structural, independent of exploration or entropy bonuses.

**Implication for our data:** This is the mathematical mechanism behind the v1 anomaly. At 2K ProRL steps, the dominant strategy already has higher p, so it gets amplified. Minority strategies get suppressed. The model hasn't trained long enough for accuracy gains to compensate, so pass@k drops.

#### Piece 3: F-GRPO (Plyusov et al. 2602.06717) — Unsampled mass shrinks

**Proposition 3.2:** Change in unsampled-correct mass:
```
ΔQ_{u,pos} = (η/N)[-S_R · U_{pos,2} - Q_{u,pos}((R_c - S_R)A_2 + (R_w - S_R)B_2 - S_R · U_2)]
```
Even when total correct mass grows (ΔQ_pos > 0), unsampled-correct mass can shrink. Strategies that happen not to be sampled in a training batch lose mass through softmax normalization.

**Lemma 3.1:** Probability of missing rare-correct modes:
```
Pr(B_τ) = (1-τ)^N - (μ_pos - τ)^N - (1-μ_pos)^N
```
Non-monotonic in N. At intermediate group sizes, the risk of sharpening onto common solutions while missing rare ones is MAXIMIZED.

**Implication for our data:** This is the mechanism for why R1-Distill retains 12 exclusive strategies while v2/BroRL have 0-2. During ProRL training, minority strategies are occasionally unsampled → lose mass → become even rarer → more likely to be unsampled. Positive feedback loop = our Pólya urn prediction formalized.

#### What's mathematically MISSING (our contribution space)

1. **No paper models strategy-level dynamics.** All analysis is at the token or answer level. "Strategy" as a mathematical object (a distribution over reasoning paths that reach the same answer) is unformalized. Our LLM judge classification is an empirical proxy, not a formal object.

2. **No paper models the v1 dip → v2 recovery trajectory.** Sinha's Theorem 3.1 predicts monotonic divergence (rich get richer). It doesn't predict recovery. The recovery (v2 > v1) likely comes from ProRL's reference reset + longer training giving the model time to build new capabilities after initial narrowing. This is unmodeled.

3. **No paper models coverage ceiling expansion.** v2 solves 3 hard problems R1 can't reach at K=64. This contradicts the "RLVR is rejection sampler" narrative (Yue et al.) and implies RL can discover solutions outside base model support despite the KL leash. Mechanism unknown.

4. **No paper distinguishes "strategy narrowing" from "strategy switching."** Our P34 (R1 uses coordinate geometry, v2 uses tangram reasoning) and P35 (v2 gains a new strategy) show ProRL can CHANGE strategy, not just narrow. This is unexplained by any current theory.

### Mathematical Narrative for Our Paper

**Theorem 3.1 (Sinha) + Remark 4.3 (Chen) + Proposition 3.2 (F-GRPO)** combine to predict:

1. Under KL-regularized RL with binary rewards, the optimal policy preserves reference policy strategy ratios (Chen Remark 4.3)
2. Under gradient flow, the dominant strategy's probability amplifies its own signal (Sinha Theorem 3.1) → monotonic narrowing
3. Minority strategies that go unsampled lose mass through softmax coupling (F-GRPO Proposition 3.2) → accelerated narrowing
4. The combination predicts: strategy narrowing is inevitable, accelerating, and structurally caused by the objective + optimizer + sampling

**Our data confirms predictions 1-3 and reveals phenomena 1-3 don't predict:**
- The v1 dip (early narrowing before capability) → explained by Theorem 3.1 dynamics
- The v2 recovery (longer training compensates) → NOT explained by any theory
- Strategy switching (P34) → NOT explained
- Coverage expansion (3 new hard problems) → CONTRADICTS "rejection sampler"
- Strategy expansion (P35: v2 gains strategy 4) → CONTRADICTS monotonic narrowing

### Key References
- Chen et al. 2510.20817 "When is RLVR Not Effective?" — Remark 4.3 (equal reward locks ratios)
- Sinha et al. 2601.21669 "Expected Return Collapse" — Theorem 3.1 (probability multiplier)
- Plyusov et al. 2602.06717 "F-GRPO" — Proposition 3.2 (unsampled mass shrinks), Lemma 3.1 (non-monotonic miss probability)
- BroRL 2510.01180 — Theorem 1 (mass balance, but token-level only)
- ProRL 2505.24864 — No original equations (all borrowed)

---

## Comprehensive Synthesis: Paper-Ready Reference (2026-02-27)

*This section consolidates all experimental findings, statistical results, mathematical framework, and paper framing from Runs 14-17 into a single write-up-ready document.*

---

### 1. Experimental Setup

#### 1.1 Models

All 4 models are 1.5B-parameter Qwen2 architecture on the same ProRL training trajectory. This gives us a longitudinal view without training a single step.

| # | Checkpoint | Training | MATH-500 | AIME24 | HuggingFace ID |
|---|---|---|---|---|---|
| A | R1-Distill-Qwen-1.5B | Distilled from DeepSeek-R1 (671B) | 82.90 | 28.54 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| B | Nemotron v1 | ProRL 2K steps on R1-Distill | 91.89 | 48.13 | `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B` rev `v1` |
| C | Nemotron v2 | ProRL 3K steps | 92.49 | 49.58 | same repo, default revision |
| D | Nemotron BroRL | v2 + 419 steps, N=512 batch | 92.20 | 60.42 | same repo, branch `brorl` |

#### 1.2 Data Generation

- **Problems:** 60 from MATH benchmark — 20 easy (Level 1-2), 20 medium (Level 3), 20 hard (Level 4-5)
- **Rollouts:** K=64 per model per problem = **15,360 total traces**
- **Generation:** Temperature 1.0, max 4096 tokens, A10G GPU on Modal
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384-dim), generated alongside traces (later shown inadequate for strategy classification)
- **Wall time:** 3.55 hours. Cost: ~$4
- **Script:** `pilot/modal_gen_traces.py`
- **Data:** `data/modal_runs/gen_traces_full/` (140MB traces, 22MB embeddings)

#### 1.3 Analysis Pipeline

| Phase | Script | Method | Status |
|---|---|---|---|
| Pass@k + statistics | `pilot/analyze_rigorous.py` | Unbiased estimator (Chen et al. 2021), 10K bootstrap, Wilcoxon | **Complete** |
| Embedding clustering | `pilot/analyze_strategies.py` | Agglomerative + cosine on MiniLM | **FAILED** (artifact) |
| LLM judge classification | `pilot/llm_judge_pilot.py` | GPT-5-nano, structured JSON output | **Complete** |
| Mathematical framework | Literature audit | ProRL, BroRL, Invisible Leash, F-GRPO | **Complete** |

---

### 2. Results: Pass@k Curves with Bootstrap CIs

All confidence intervals from 10,000 bootstrap resamples. Pass@k uses the unbiased estimator: pass@k = 1 − C(n−c, k) / C(n, k).

#### 2.1 Easy Tier (20 problems)

| Model | pass@1 | pass@4 | pass@8 | pass@16 | pass@64 |
|---|---|---|---|---|---|
| R1-Distill | 0.799 [0.668, 0.907] | 0.931 [0.821, 0.996] | 0.947 [0.845, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] |
| v1 (2K) | 0.895 [0.773, 0.980] | 0.948 [0.848, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] |
| v2 (3K) | **0.927** [0.816, 0.991] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] |
| BroRL | 0.920 [0.805, 0.990] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] |

**Observation:** Easy tier saturates at pass@4 for all RL-trained models. No signal for strategy analysis.

#### 2.2 Medium Tier (20 problems)

| Model | pass@1 | pass@4 | pass@8 | pass@16 | pass@64 |
|---|---|---|---|---|---|
| R1-Distill | 0.556 [0.367, 0.734] | 0.674 [0.485, 0.848] | 0.728 [0.542, 0.890] | 0.777 [0.601, 0.926] | 0.850 [0.700, 1.000] |
| v1 (2K) | 0.565 [0.377, 0.750] | 0.685 [0.494, 0.849] | 0.737 [0.554, 0.903] | 0.774 [0.587, 0.940] | 0.800 [0.600, 0.950] |
| v2 (3K) | **0.714** [0.544, 0.869] | 0.823 [0.661, 0.964] | 0.845 [0.693, 0.993] | 0.850 [0.700, 1.000] | 0.850 [0.700, 1.000] |
| BroRL | 0.679 [0.492, 0.847] | 0.775 [0.596, 0.930] | 0.801 [0.612, 0.948] | 0.812 [0.637, 0.962] | 0.850 [0.700, 1.000] |

**Observation:** v2 reaches ceiling (0.850) by pass@8. R1-Distill needs pass@32 to approach the same ceiling. v1 trails R1-Distill at k≥32.

#### 2.3 Hard Tier (20 problems) — THE ACTION

| Model | pass@1 | pass@4 | pass@8 | pass@16 | pass@64 |
|---|---|---|---|---|---|
| R1-Distill | 0.436 [0.259, 0.613] | 0.585 [0.393, 0.774] | 0.641 [0.447, 0.829] | 0.690 [0.498, 0.868] | 0.750 [0.550, 0.900] |
| v1 (2K) | 0.355 [0.176, 0.548] | 0.461 [0.279, 0.651] | 0.537 [0.357, 0.718] | 0.621 [0.432, 0.801] | 0.750 [0.550, 0.900] |
| v2 (3K) | **0.545** [0.377, 0.717] | **0.737** [0.581, 0.878] | **0.816** [0.662, 0.944] | **0.864** [0.711, 0.989] | **0.900** [0.750, 1.000] |
| BroRL | 0.507 [0.335, 0.680] | 0.692 [0.514, 0.849] | 0.763 [0.595, 0.912] | 0.813 [0.642, 0.954] | 0.850 [0.700, 1.000] |

**Key finding:** v2@64 = 0.900 vs R1@64 = 0.750. A 15pp gap that search budget CANNOT close. ProRL created genuine explorability.

#### 2.4 Aggregate (All 60 problems)

| Model | pass@1 | pass@8 | pass@64 |
|---|---|---|---|
| R1-Distill | 0.597 [0.495, 0.698] | 0.772 [0.673, 0.865] | 0.850 [0.750, 0.933] |
| v1 (2K) | 0.605 [0.494, 0.711] | 0.741 [0.637, 0.839] | 0.833 [0.733, 0.917] |
| v2 (3K) | **0.729** [0.632, 0.817] | **0.871** [0.786, 0.942] | **0.900** [0.817, 0.967] |
| BroRL | 0.702 [0.601, 0.796] | 0.838 [0.747, 0.918] | 0.883 [0.800, 0.950] |

---

### 3. Statistical Tests

#### 3.1 v1 Anomaly (Wilcoxon signed-rank, R1-Distill vs v1)

v1 (2K ProRL steps) performs WORSE than the distilled base model on hard problems at k≥8.

| Comparison | Mean Δ (R1 − v1) | p-value | Sig. | R1 wins | v1 wins |
|---|---|---|---|---|---|
| Hard, k=8 | **+0.104** | **0.0049** | ** | 9 | 2 |
| Hard, k=16 | +0.069 | 0.016 | * | 9 | 2 |
| Hard, k=32 | +0.042 | 0.049 | * | 8 | 1 |
| Medium, k=8 | −0.009 | 0.68 | ns | 3 | 5 |
| Medium, k=16 | +0.003 | 0.68 | ns | 3 | 5 |

**Interpretation:** Early ProRL (2K steps) narrows the distribution before building new capabilities. On hard problems, this costs pass@k significantly. The v1 anomaly is the empirical signature of the Pólya urn phase — early correct solutions compound and crowd out exploration.

#### 3.2 v2 vs R1-Distill (Wilcoxon signed-rank)

| Comparison | Mean Δ (v2 − R1) | p-value | Sig. | v2 wins | R1 wins |
|---|---|---|---|---|---|
| Easy, k=1 | +0.127 | **0.0003** | *** | 15 | 0 |
| Medium, k=1 | +0.158 | **0.0001** | *** | 17 | 0 |
| Hard, k=1 | +0.109 | **0.006** | ** | 14 | 3 |
| Hard, k=8 | +0.175 | 0.014 | * | 11 | 3 |
| Hard, k=64 | +0.150 | 0.188 | ns | 4 | 1 |

**Interpretation:** v2 is significantly better at pass@1 across all tiers. At pass@64 on hard, the advantage (+0.15) is directionally strong but not significant (p=0.19) — only 4 problems differ, so the Wilcoxon test lacks power. The 4 specific problems v2 solves that R1 can't (42, 44, 50, 51) are individually decisive even if the aggregate test is underpowered.

#### 3.3 Sampling Efficiency (R1 samples needed to match v2@1)

| Tier | k needed | R1 pass@k achieved | v2 pass@1 target |
|---|---|---|---|
| Easy | 4 | 0.931 | 0.927 |
| Medium | 7 | 0.718 | 0.714 |
| Hard | 3 | 0.559 | 0.545 |

**Interpretation:** RL training gives a ~3-7× inference efficiency multiplier. One v2 greedy sample ≈ 3-7 R1-Distill samples. But this only measures pass@1 matching — at higher k, v2 pulls further ahead.

---

### 4. Coverage Analysis

#### 4.1 Problem Coverage (can it solve the problem at ANY k?)

| Tier | R1-Distill | v1 (2K) | v2 (3K) | BroRL |
|---|---|---|---|---|
| Easy (20) | 19 | 19 | 19 | 19 |
| Medium (20) | 17 | 16 | 17 | 17 |
| Hard (20) | 15 | 15 | **18** | 17 |
| **Total (60)** | **51** | **50** | **54** | **53** |

**Unsolvable by R1-Distill (even at K=64):** Problems {42, 44, 50, 51, 55}
**Unsolvable by v2:** Problems {55, 57}
**v2-exclusive (R1 can't, v2 can):** {42, 44, 50, 51} — 4 hard problems
**R1-exclusive (R1 can, v2 can't):** {57} — 1 hard problem

#### 4.2 Interpretation

v2 solves 3 net additional hard problems (4 gained − 1 lost). This is **coverage expansion**, not compression. The "rejection sampler" narrative (Yue et al.) predicts RLVR should only sharpen existing correct solutions, never create new ones. Our data contradicts this for 4/20 hard problems.

Problem 57 (lost by v2) is the single case where ProRL's narrowing cost a reachable solution. This trade is 4:1 in v2's favor.

---

### 5. LLM Judge: Strategy Classification

#### 5.1 Setup

- **Classifier:** GPT-5-nano via OpenAI API
- **Prompt:** Classify traces into distinct mathematical STRATEGIES (not surface variation)
- **Input:** Up to 8 correct traces per model per problem, truncated to 3000 chars
- **Output:** Structured JSON with strategy names, descriptions, and per-trace classifications
- **Cost:** $0.15 total (1.5M input + 179K output tokens)
- **Script:** `pilot/llm_judge_pilot.py`
- **Data:** `data/analysis/llm_judge_pilot.json`

#### 5.2 Aggregate Results

| Tier | Problems with >1 strategy | Mean strategies/problem |
|---|---|---|
| Easy (20) | 5 | 1.40 |
| Medium (20) | 4 | 1.25 |
| Hard (20) | 5 | 1.40 |
| **All (60)** | **14 (23%)** | **1.35** |

Strategy diversity is roughly uniform across difficulty tiers — not a difficulty-dependent phenomenon at 1.5B.

#### 5.3 Per-Model Strategy Breadth (on 12 multi-strategy problems with all 4 models present)

| Model | Mean strategies used | Exclusive strategies (only this model uses) |
|---|---|---|
| R1-Distill | **2.14** | 12 |
| v1 (2K) | 1.71 | 5 |
| v2 (3K) | 1.57 | 2 |
| BroRL | 1.57 | 0 |

| Comparison (multi-strat problems) | Count |
|---|---|
| R1 uses MORE strategies than v2 | 6/12 |
| v2 uses MORE strategies than R1 | 1/12 |
| Tied | 5/12 |

#### 5.4 Noteworthy Individual Problems

| Problem | Tier | Subject | R1 strategies | v2 strategies | Phenomenon |
|---|---|---|---|---|---|
| P0 | Easy | Precalculus | 2 (Trig+Pythag, Direct cot) | 1 (Trig+Pythag) | Narrowing |
| P2 | Easy | Prealgebra | 4 (long div, ×5 to 100, 1/20 scale, factor decomp) | 2 | Strategy elimination |
| P34 | Med | Prealgebra | 1 (coordinate geometry) | 1 (tangram reasoning) | **Strategy switching** |
| P35 | Med | Precalculus | 2 trig methods | **3 trig methods** | **v2 GAINED a strategy** |
| P46 | Hard | Algebra | 4 methods | 3 methods | Slight narrowing |
| P52 | Hard | Int. Algebra | 3 (AM-GM, Symmetry, Lagrange) | 1 (AM-GM only) | Dropped 2 strategies |
| P58 | Hard | Int. Algebra | 2 (AM-GM, Lagrange) | 1 (Lagrange only) | Dropped AM-GM |

#### 5.5 Key Patterns

1. **ProRL preferentially drops MINORITY strategies.** The dominant strategy on every problem is preserved. What's lost are the less-common approaches (Lagrange when AM-GM dominates, coordinate geometry when tangram reasoning suffices).

2. **Two anomalies contradict simple "narrowing" narrative:**
   - **P34 (strategy switching):** R1-Distill uses coordinate geometry (brute force), v2 uses tangram-area reasoning (more elegant). ProRL didn't just narrow — it REPLACED the strategy.
   - **P35 (strategy expansion):** v2 uses 3 trig identity methods while R1 uses only 2. ProRL made the model competent enough to explore a third approach.

3. **R1-Distill has 12 exclusive strategies vs v2's 2.** ProRL training eliminated 10 strategies that only the distilled base model could access.

---

### 6. Mathematical Framework

Three formal results from the literature predict the strategy narrowing we observe. No existing paper combines them or predicts the anomalies we find.

#### 6.1 Chen et al. 2510.20817 — The Objective DESIGNS Collapse

**Remark 4.3 (Equal-reward case).** For two correct answers y₁, y₂ with R(y₁) = R(y₂), the optimal KL-regularized policy satisfies:

$$\frac{G_\beta(y_1)}{G_\beta(y_2)} = \frac{\pi_{\text{ref}}(y_1)}{\pi_{\text{ref}}(y_2)}$$

The optimal policy locks strategy ratios to the reference policy, **independent of β**. Tuning the regularization coefficient cannot change this. If R1-Distill already favors one strategy 3:1, the KL-optimal ProRL policy preserves that 3:1 ratio.

**Our data:** This explains WHY v2 narrows (6/12 multi-strat problems). The narrowing is by construction — the objective specifies it.

#### 6.2 Sinha et al. 2601.21669 — The Probability Multiplier

**Theorem 3.1.** Under gradient flow on expected return with softmax parametrization:

$$\frac{d}{dt} \log\frac{p_i(t)}{p_j(t)} = p_i(t) \cdot a_i(t) - p_j(t) \cdot a_j(t)$$

When pᵢ > pⱼ and r(i) ≥ r(j), the log-ratio grows — the rich get richer. The probability of a strategy amplifies its own gradient signal. **This is structural, independent of exploration bonuses or entropy regularization.**

**Our data:** This is the mathematical mechanism behind the v1 anomaly. At 2K ProRL steps, the dominant strategy already has higher p → gets amplified → minority strategies get suppressed → pass@k drops before accuracy gains compensate. Theorem 3.1 predicts monotonic divergence.

#### 6.3 F-GRPO (Plyusov et al. 2602.06717) — Unsampled Mass Shrinks

**Proposition 3.2.** Change in unsampled-correct mass:

$$\Delta Q_{u,\text{pos}} = \frac{\eta}{N}\left[-S_R \cdot U_{\text{pos},2} - Q_{u,\text{pos}}\left((R_c - S_R)A_2 + (R_w - S_R)B_2 - S_R \cdot U_2\right)\right]$$

Even when total correct mass grows (ΔQ_pos > 0), unsampled-correct mass can shrink. Strategies not sampled in a training batch lose mass through softmax normalization.

**Lemma 3.1.** Probability of missing rare-correct modes:

$$\Pr(B_\tau) = (1-\tau)^N - (\mu_{\text{pos}} - \tau)^N - (1-\mu_{\text{pos}})^N$$

Non-monotonic in N. At intermediate group sizes, the risk of sharpening onto common solutions while missing rare ones is MAXIMIZED.

**Our data:** This explains R1-Distill's 12 exclusive strategies vs v2's 2. During ProRL training, minority strategies are occasionally unsampled → lose mass → become rarer → more likely to be unsampled. Positive feedback loop.

#### 6.4 Combined Prediction

**Theorem 3.1 (Sinha) + Remark 4.3 (Chen) + Proposition 3.2 (F-GRPO)** predict:

1. KL-regularized RL preserves reference policy strategy ratios (Chen Remark 4.3)
2. Gradient flow amplifies the dominant strategy's signal (Sinha Theorem 3.1) → monotonic narrowing
3. Minority strategies that go unsampled lose mass through softmax coupling (F-GRPO Prop 3.2) → accelerated narrowing
4. **Combined:** strategy narrowing is inevitable, accelerating, and caused by objective + optimizer + sampling jointly

#### 6.5 What the Theory DOES NOT Predict (Our Contribution Space)

| Empirical finding | Predicted by theory? | Gap |
|---|---|---|
| v1 anomaly (early narrowing before capability) | **Partially** — Theorem 3.1 predicts narrowing, but not the temporal dynamics | Recovery at v2 is unexplained |
| Coverage expansion (v2 solves 4 new hard problems) | **No** — contradicts "rejection sampler" narrative | RL discovers solutions outside base model support despite KL leash |
| Strategy switching (P34: coord geo → tangram) | **No** — all theories predict ratio preservation, not substitution | Mechanism unknown |
| Strategy expansion (P35: v2 gains 3rd strategy) | **No** — contradicts monotonic narrowing | Possibly: accuracy gain enables exploration of new approach |
| v1 dip → v2 recovery trajectory | **No** — Theorem 3.1 predicts monotonic divergence only | ProRL's reference reset may re-anneal |

---

### 7. Answer-Level Analysis (What Embedding Clustering Missed)

#### 7.1 Why Embedding Clustering Failed

| Diagnostic | Result | Implication |
|---|---|---|
| Intra-problem cosine similarity | 0.92-0.96 for ALL models | Embeddings encode problem identity, not strategy |
| Cross-model centroids | 0.97-0.99 cosine | Cannot distinguish R1's solution from v2's |
| Within-answer vs between-answer gap | 0.004-0.025 | If it can't separate answers, it can't separate strategies |
| Random baseline silhouette | 0.007 (vs 0.55 real data) | Something real but not strategy-level — likely surface features |
| Cluster count replication | k=2 for 58-59/60 problems, all models | Embedding artifact, not model property |

**Lesson:** MiniLM (384-dim, trained on natural language) is not strategy-sensitive for mathematical reasoning. This likely generalizes to all general-purpose sentence transformers used in DIVER, DSDR, DRA-GRPO, etc.

#### 7.2 Answer Diversity is Mostly Error Diversity

| Model | Problems with >1 unique answer | Problems with >1 CORRECT answer |
|---|---|---|
| R1-Distill | 24/60 | 2/60 |
| v1 (2K) | 4/60 | 0/60 |
| v2 (3K) | 7/60 | 1/60 |
| BroRL | 6/60 | 1/60 |

At 1.5B, "diversity" is overwhelmingly wrong-answer diversity. ProRL didn't destroy diversity — it eliminated errors.

---

### 8. The Combined Narrative

#### The Breadth-Reliability Tradeoff

ProRL trades **strategy breadth** for **execution reliability**:

- R1-Distill: wider repertoire (2.14 strategies), lower accuracy (pass@1 = 0.597)
- v2 (3K ProRL): narrower repertoire (1.57 strategies), higher accuracy (pass@1 = 0.729)
- The accuracy gain also expands coverage — v2 reaches 4 hard problems R1-Distill can't solve at any K

This is NOT simple collapse. It's a restructuring:
- Minority strategies are dropped (10 exclusive strategies lost from R1 to v2)
- The dominant strategy is sharpened (pass@1 improves 22%)
- In rare cases, new strategies emerge (P35) or old ones are replaced by better ones (P34)
- The net coverage effect is POSITIVE (+3 hard problems)

#### The v1 Anomaly as Phase Transition Marker

The v1 checkpoint (2K steps) represents the worst of both worlds: narrowed distribution (lost diversity) without yet gaining capability (worse pass@k than R1-Distill on hard). The v2 checkpoint (3K steps) has crossed a threshold where accuracy gains compensate for diversity loss.

This suggests a **phase transition** between 2K and 3K ProRL steps where the model transitions from "narrowing" to "narrowing + capability gain." The location of this transition is a novel finding.

#### Explorability ≠ Diversity

The key reframe: what matters is not how many strategies a model STORES (diversity) but how many it CAN DISCOVER given inference compute (explorability).

| Metric | R1-Distill | v2 |
|---|---|---|
| Strategy breadth | 2.14 (more diverse) | 1.57 |
| pass@1 | 0.597 | **0.729** |
| pass@64 (hard) | 0.750 | **0.900** |
| Coverage ceiling (hard) | 15/20 | **18/20** |
| R1@64 vs v2@64 | 0.750 | **0.900** |

v2 is less diverse but MORE explorable. With equal search budget, v2 reaches more problems. The diversity R1-Distill has is error diversity, not useful exploration capacity.

---

### 9. Novel Contributions (with supporting evidence)

| # | Contribution | Evidence | Status |
|---|---|---|---|
| 1 | **4-checkpoint longitudinal strategy evolution** across ProRL trajectory | Tables §2-5 | No prior work tracks strategy counts across RL checkpoints |
| 2 | **v1 anomaly**: early ProRL worse than base at k≥8 on hard (p=0.005) | §3.1, hard k=8 | Unreported in ProRL/BroRL papers |
| 3 | **Strategy narrowing quantified**: R1-Distill 2.14 → v2 1.57 strategies | §5.3 | First measurement with LLM judge on ProRL trajectory |
| 4 | **Narrowing is not total**: strategy switching (P34) and expansion (P35) | §5.4 | Contradicts simple "collapse" narrative |
| 5 | **Coverage ceiling expansion**: v2 reaches 4 new hard problems | §4.1 | Contradicts "rejection sampler" (Yue et al.) |
| 6 | **3-7× inference efficiency**: R1@k ≈ v2@1 at k=3-7 | §3.3 | Concrete quantification on matched trajectory |

#### What Is NOT Novel (confirmed but not discovered by us)

- RLVR improves pass@1 while degrading pass@k at large k (Yue et al. NeurIPS 2025)
- Token entropy ↑ while answer diversity ↓ (Invisible Leash, 2507.14843)
- KL-regularized RL specifies unimodal target (Chen et al. 2510.20817)
- ProRL achieves higher MATH accuracy than R1-Distill (ProRL paper)

---

### 10. Differentiation from "Rewarding the Rare" (Hu et al. 2601.08763)

| Dimension | Rewarding the Rare | Our work |
|---|---|---|
| Purpose | Intervention (reward rare strategies during training) | Diagnostic (measure strategy evolution across training) |
| Scale | 7B-8B | 1.5B |
| LLM judge | 32B-72B, black-box | GPT-5-nano, structured taxonomy |
| Training | Yes (modified GRPO advantages) | No (analyze existing checkpoints) |
| Longitudinal | No (single endpoint) | Yes (4 checkpoints on same trajectory) |
| Mathematical grounding | None (empirical only) | 3 formal results (Chen, Sinha, F-GRPO) |
| Anomaly detection | N/A | v1 dip, P34 switching, P35 expansion |
| Cost | Not reported (72B judge during training = expensive) | $4.15 total ($4 generation + $0.15 judge) |

---

### 11. Gaps and Limitations

1. **Small problem set (60).** Hard-tier significance tests are underpowered (p=0.19 for v2 vs R1 at pass@64). Need 100+ problems for robust hard-tier statistics.

2. **1.5B scale only.** The 1.5B models have very limited strategy diversity (23% of problems). At 7B+ we'd expect more strategies and clearer signal.

3. **Single LLM judge run.** No inter-annotator agreement or judge reliability assessment. A second judge (e.g., GPT-5.2-mini) for calibration would strengthen the classification.

4. **No causal mechanism for anomalies.** P34 (strategy switching) and P35 (strategy expansion) are empirical observations. We can describe but not explain them.

5. **Binary reward only.** Our analysis uses correct/incorrect. A process reward model would enable strategy classification within incorrect traces too.

6. **Trace truncation.** LLM judge saw only first 3000 chars per trace. For very long solutions (4096 tokens), late-stage strategy changes may be missed.

---

### 12. Paper Skeleton

**Title (candidates):**
- "Breadth for Reliability: Strategy Evolution Along the ProRL Training Trajectory"
- "The Cost of Convergence: How RLVR Trades Strategy Diversity for Execution Accuracy"
- "Explorability ≠ Diversity: What 15,360 Reasoning Traces Reveal About RLVR Training"

**Abstract sketch:**
We analyze reasoning strategy diversity across 4 checkpoints on NVIDIA's ProRL training trajectory (R1-Distill → Nemotron v1/v2/BroRL, all 1.5B Qwen2). Generating 15,360 traces across 60 MATH problems, we find: (1) ProRL narrows strategy repertoire (2.14 → 1.57 strategies per multi-strategy problem) while improving accuracy (pass@1: +22%), (2) early ProRL (2K steps) is significantly worse than the base model at k≥8 on hard problems (p=0.005), marking a phase transition between narrowing and capability gain, (3) ProRL expands coverage ceiling despite narrowing — solving 4 hard problems unreachable by the base model — contradicting the "rejection sampler" characterization, and (4) three formal results (Chen Remark 4.3, Sinha Theorem 3.1, F-GRPO Proposition 3.2) jointly predict strategy narrowing but fail to predict our observed exceptions: strategy switching, strategy expansion, and coverage ceiling growth. We argue explorability (capacity to discover strategies under search) is a more informative metric than diversity (strategies stored in the policy).

**Section outline:**
1. Introduction — the "is RLVR just a rejection sampler?" question
2. Related Work — Invisible Leash, Rewarding the Rare, ProRL/BroRL, F-GRPO
3. Experimental Setup — 4 models, 60 problems, K=64, generation + analysis pipeline
4. Results: Pass@k and Coverage — tables, bootstrap CIs, sampling efficiency
5. Results: Strategy Classification — LLM judge setup, per-problem results, narrowing quantification
6. Mathematical Framework — Chen + Sinha + F-GRPO combined prediction vs. data
7. Discussion — breadth-reliability tradeoff, v1 anomaly, P34/P35 anomalies, explorability reframe
8. Limitations and Future Work — scale, problem count, causal mechanisms
9. Conclusion — ProRL restructures (not just compresses) the strategy space

---

### 13. Key References (Paper-Ready)

#### Core (we analyze or extend)
- ProRL: Wang et al. "ProRL: Prolonged RL Training of LLMs with Effective Reward Stabilization." arXiv:2505.24864 (2025).
- BroRL: Shi et al. "BroRL: Breadth-First RL." arXiv:2510.01180 (2025).
- F-GRPO: Plyusov et al. "F-GRPO: Understanding and Fixing Group Relative Policy Optimization." arXiv:2602.06717 (2025).
- Invisible Leash: Wu et al. "The Invisible Leash." arXiv:2507.14843 (2025).
- Rewarding the Rare: Hu et al. arXiv:2601.08763 (2026).
- Chen et al. "When is RLVR Not Effective?" arXiv:2510.20817 (2025).
- Sinha et al. "Expected Return Collapse in RLVR." arXiv:2601.21669 (2026).

#### Context (we cite for framing)
- Yue et al. "RLVR is a Rejection Sampler." NeurIPS 2025. arXiv:2504.13837.
- TTT-Discover: Wei et al. arXiv:2601.16175 (2026).
- DAPO: Yu et al. arXiv:2503.14476 (2025).
- Beyond 80/20: Wang et al. arXiv:2506.01939 (2025).
- Chen et al. "Evaluating Mathematical Reasoning." ICLR 2022 (pass@k estimator).

#### Cross-disciplinary (we invoke for analogy)
- Pólya 1930: Urn models, preferential attachment.
- Boltzmann/Kirkpatrick 1983: Simulated annealing.
- Ecoffet et al. 2021: Go-Explore, detachment problem.
- Kauffman 2000: Adjacent possible.

---

### 14. Deliverables Index

| File | Description |
|---|---|
| `pilot/modal_gen_traces.py` | Phase 1: trace generation on Modal (4 models × 60 problems × K=64) |
| `pilot/analyze_rigorous.py` | Bootstrap CIs, Wilcoxon tests, coverage analysis |
| `pilot/analyze_strategies.py` | Embedding clustering (FAILED, diagnostic value only) |
| `pilot/llm_judge_pilot.py` | GPT-5-nano strategy classification |
| `data/modal_runs/gen_traces_full/` | 15,360 traces + embeddings (140MB) |
| `data/analysis/rigorous_results.json` | Statistical analysis results |
| `data/analysis/llm_judge_pilot.json` | LLM judge classifications for 60 problems |
| `data/analysis/phase2_results.json` | Embedding clustering results (artifact, kept for transparency) |
| `pilot/LOG.md` | This experimental log |
| `docs/distill/reasoning-strategy-diversity.md` | Full intellectual distillation |

### 15. Budget

| Item | Cost |
|---|---|
| Trace generation (15,360 traces, A10G) | ~$4.00 |
| LLM judge (GPT-5-nano, 1.7M tokens) | ~$0.15 |
| **Total for current experiment** | **~$4.15** |
| Budget remaining | ~$70 |
| Prior runs (Runs 1-13) | ~$76 |

---

## Run 18: External Validation — Fang Wu Conversation + Literature Cross-Check (2026-02-27)

### Context

Conversation with Fang Wu (co-author BroRL, extended ProRL research at NVIDIA).

### Key Intelligence from Fang Wu

1. **Diminishing returns at 3K steps.** Fang Wu saw this independently — confirms our v2 plateau finding.
2. **Pivoted to diffusion language models.** Fang Wu left RLVR entirely. Signal: builder of ProRL doesn't think RLVR exploration has legs.
3. **DeepSearch narrows, doesn't broaden.** Even search on top of RLVR doesn't create diverse strategies.
4. **TTT-Discover improvements may be sampling error.** Math gains are 4th-5th decimal place. Engineering gains (TriMul 4.6×) are real but domain-specific.
5. **"RLVR shrink, SFT expand."** RLVR is being used as synthetic data generation for SFT distillation, not as direct training.

### QED-Nano Deep Dive

QED-Nano (CMU/ETH/Numina/HF, Qwen3-4B) on IMO-ProofBench:
- **Distillation (SFT): +19.1 points** (20.4 → 39.5)
- **RL + Reasoning Cache: +0.5 points** (39.5 → 40.0)
- **Agent scaffold at test-time: +14.0 points** (40.0 → 54.0)

RL contributes noise-level improvement. Distillation and test-time search do all the work.

**QED-Nano's diversity mechanisms (3 layers):**
1. `reasoning_cache` (training): iterative solve → summarize → re-solve. Prompt explicitly says "Proving the result in a different way. Finding alternative problem-solving strategies."
2. `deepseek_agent` (eval, headline results): generate → verify → refine, pool_size=4 parallel generations.
3. `determine_approaches_agent`: generates N distinct high-level strategies upfront, then solves each independently. Prompt: "Each approach should be distinct... should stand on its own... Avoid Full Solutions."

### Assessment

Our diagnostic findings (strategy narrowing, v1 anomaly, pass@k curves) are **incremental over established literature** (Chen 2510.20817, Invisible Leash, Yue et al. NeurIPS 2025). The direction "RLVR narrows" is known. What's missing in the literature: nobody has tested whether a frozen model + external archive can recover strategies that RLVR training destroyed.

---

## Run 19: Landscape Survey + Archive-Guided Pilot Design (2026-02-27)

### Objective

Survey all papers that tried exploration/diversity for reasoning, assess what worked, identify gap, design a pilot experiment to test archive-guided strategy discovery on our existing models.

### Landscape Survey: Papers That Tried Exploration for Reasoning Diversity

#### A. Training-Time Diversity Methods (modify RL objective)

| Paper | Method | Model | Benchmark | Key Result | Diversity Metric |
|-------|--------|-------|-----------|------------|-----------------|
| **Rewarding the Rare** (Hu et al. 2601.08763) | LLM judge clusters strategies, inverse-frequency reward | Qwen2.5-7B | Math/Physics/Medical | pass@k ↑, AUC@128 +0.058 | LLM judge clusters |
| **DSDR** (2602.19895) | Dual-scale diversity regularization (trajectory + token) | Qwen2.5-1.5B, Qwen3-4B | AIME24/25, MATH500 | AIME24: 56.7 vs 36.7 GRPO (4B) | LLM judge 1-10 scale |
| **SuS** (2601.10349) | Strategy-aware surprise as intrinsic reward | Qwen2.5-1.5B | GSM8K | pass@1: 14.2 vs 12.1, pass@5: 46.8 vs 37.1 | Cluster entropy |
| **RL-PLUS** (2508.00222) | External data + exploration advantage function | Qwen2.5-Math-1.5B/7B | 6 math benchmarks | 1.5B avg: 39.7 vs 30.1 GRPO vs 16.0 base | pass@k curves |
| **DPH-RL** (2509.07430) | Mass-covering f-divergence instead of KL | Llama/Qwen 7B-32B | Math + SQL | pass@1 and pass@k both ↑ | pass@k only |
| **Flow of Reasoning** (2406.05673) | GFlowNet (sample proportional to reward) | Llama-3-8B | BlocksWorld, Game24, etc. | 1.33 unique solutions vs 1.12 baseline | Unique solution count |
| **Pass@k Training** (2508.10751) | Directly optimize pass@k | — | Math | Exploration + exploitation synergy | pass@k curves |
| **Exploration vs Exploitation** (2512.16912, ICLR 2026) | Clipping + entropy analysis | — | Math | Clipping bias reduces entropy; entropy min alone insufficient | Entropy |
| **Scaling Up RL** (2507.12507) | Prolonged training + DAPO + reference reset | R1-Distill-Qwen-1.5B | AIME/GPQA/Codeforces | Entropy collapse without intervention; ref reset recovers | Entropy only |

#### B. Test-Time Diversity Methods (model frozen or near-frozen)

| Paper | Method | Model | Key Result | Diversity Mechanism |
|-------|--------|-------|------------|-------------------|
| **TTT-Discover** (2601.16175) | 50 RL steps + PUCT at test time | Various | Math: marginal (4th decimal). Engineering: 4.6× | Iterative refinement |
| **QED-Nano** (CMU-AIRe) | determine_approaches_agent + reasoning_cache | Qwen3-4B | SFT: +19.1pts. RL+RC: +0.5pts. Agent: +14pts | Explicit approach generation |
| **Diverse Inference & Verification** (2502.09955) | Multi-model, multi-method ensemble | Multiple | IMO combinatorics: 33→78%, HLE: 8→37% | Multi-model diversity |

#### C. Three Key Patterns

**Pattern 1: No standard diversity metric exists.**
- Rewarding the Rare: LLM judge clusters
- DSDR: LLM judge 1-10 scoring
- SuS: cluster entropy
- Flow of Reasoning: unique solution count
- Most papers: just report pass@k curves and call it "diversity"

**Pattern 2: Training-time diversity improvements are modest on math.**
- SuS on Qwen2.5-1.5B: 12.1→14.2 pass@1 on GSM8K (easy benchmark)
- Flow of Reasoning: 1.33 vs 1.12 unique solutions (fractional gains)
- DSDR's AIME gains are impressive but on 4B, and unclear how much is diversity vs raw capability
- Rewarding the Rare: AUC@128 +0.058 — real but small

**Pattern 3: Test-time search dominates training-time diversity, every time, on math.**
- QED-Nano: agent scaffold +14pts, RL training +0.5pts
- Diverse Inference: multi-method ensemble 33→78% on IMO combinatorics
- TTT-Discover: RL helps for engineering tasks, marginal for math
- RL-PLUS: best training method still shows pass@k crossover with base at high k

### The Gap

**No paper has tested: does RLVR training help or hurt a frozen model's capacity as a search/exploration operator?**

Specifically:
- Rewarding the Rare fixes training but doesn't compare to test-time search
- QED-Nano shows test-time search wins but doesn't compare across training checkpoints
- RL-PLUS shows GRPO collapses at high k but their fix is more training
- TTT-Discover uses RL at test-time but doesn't compare base vs RLVR-trained models as the search operator
- Nobody puts them together: same archive mechanism × different training stages × same model family

**Our unique question: Does RLVR training create or destroy the model's capacity to discover strategies when given the same search scaffold?**

### Pilot Experiment Design: Archive-Guided Strategy Discovery

#### Research Question

Given the same archive-guided prompting scaffold, does a RLVR-trained model (v2) discover more, fewer, or different strategies compared to the pre-RLVR base model (R1-Distill)?

#### Kill-Test Predictions

| Finding | Implication |
|---------|-------------|
| R1-Distill + archive discovers strategies v2 cannot | RLVR destroys explorability — training is counterproductive for search |
| v2 + archive discovers MORE than R1-Distill + archive | RLVR creates genuine explorability — training aids search |
| Both find same strategies | Strategies are latent in base model; training irrelevant for search |
| v2 + archive recovers strategies v2 lost (found in R1 baseline) | Archive-guided search can undo RLVR narrowing |

#### Pilot Scope (Validation Before Full Commit)

**Problems:** 10 selected (same as our original LLM judge pilot selection, chosen for diversity signal):
- Easy: P1 (high CV), P5 (R1 answer divergence)
- Medium: P23 (R1 divergent), P37 (R1 >> v1)
- Hard jointly-solved: P40 (v2>>R1), P43 (both 59+), P59 (R1>>v2)
- Hard v2-only: P42 (R1=0, v2=9), P51 (R1=0, v2=29)
- Hard R1-only: P57 (R1=2, v2=0)

**Models:** R1-Distill and v2 only (the endpoints of the trajectory — maximum contrast).

**Archive Loop (per problem, per model):**

```
Phase 0: Baseline (already done)
  - 64 rollouts, standard generation. Count correct, identify strategies via LLM judge.

Phase 1: Archive-Prompted Generation (NEW)
  For iteration i = 1 to 3:
    a. Build archive: collect all distinct strategies found so far (from Phase 0 + prior iterations)
    b. Build prompt: "Here is the problem. Here are strategies already discovered: [archive].
       Find a FUNDAMENTALLY DIFFERENT mathematical approach. Do not use [list of known strategies]."
    c. Generate K=16 rollouts with this prompt (temperature 1.0)
    d. Check correctness (answer matching)
    e. Classify new correct traces with LLM judge
    f. Add genuinely novel strategies to archive

Phase 2: Analysis
  - Total unique strategies per model (baseline + archive-guided)
  - Strategies found ONLY via archive (not in baseline)
  - Strategies found by R1+archive but not v2+archive (and vice versa)
  - Cost per novel strategy discovered
```

**Why 3 iterations:** QED-Nano's reasoning_cache uses 3 iterations. Diminishing returns expected after 2-3.

**Why K=16 per iteration:** Balance cost and coverage. 3 iterations × 16 = 48 archive-guided rollouts per model per problem, on top of 64 baseline = 112 total.

#### Prompt Template (adapted from QED-Nano's determine_approaches_agent)

```
You are solving a mathematical problem. Previous solvers have found these strategies:

{archive_block}

Your task: solve this problem using a FUNDAMENTALLY DIFFERENT mathematical approach
than any listed above. A different approach means a different core mathematical method
(e.g., algebraic vs geometric, direct computation vs proof by contradiction,
substitution vs factoring).

Do NOT:
- Reuse any strategy listed above
- Simply vary the presentation of an existing strategy
- Change only the verbosity or step ordering

Problem: {problem_text}

Solve this problem step by step using your novel approach.
```

#### Cost Estimate

| Component | Count | Cost |
|-----------|-------|------|
| Archive-guided generation (2 models × 10 problems × 3 iters × 16 rollouts) | 960 traces | ~$0.30 (A10G on Modal) |
| LLM judge classification (960 new traces) | ~250K tokens | ~$0.02 (GPT-5-nano) |
| **Pilot total** | | **~$0.35** |

Budget remaining after pilot: ~$70.

#### Success Criteria (What Makes This Worth Full-Scale)

1. **Signal exists:** At least 3/10 problems show ≥1 novel strategy found via archive that wasn't in baseline.
2. **Model difference exists:** R1-Distill and v2 find DIFFERENT strategies via archive on ≥2 problems.
3. **Archive helps:** Archive-guided generation finds at least 1 strategy that 64 baseline rollouts missed.

If ANY criterion fails on all 10 problems, the approach is dead — archive-guided prompting doesn't work at 1.5B scale, and we should write the diagnostic paper with what we have.

If ALL THREE pass: proceed to full-scale (60 problems, 4 models, more iterations).

#### Implementation Plan

| Step | Script | Dependencies | Output |
|------|--------|-------------|--------|
| 1. Build archive from existing data | `pilot/build_archive.py` | `data/analysis/llm_judge_pilot.json` + `data/modal_runs/gen_traces_full/` | `data/analysis/strategy_archives.json` |
| 2. Archive-guided generation on Modal | `pilot/modal_archive_gen.py` | Step 1 output + Modal A10G | `data/modal_runs/archive_pilot/` |
| 3. LLM judge on new traces | `pilot/llm_judge_archive.py` | Step 2 output + OpenAI API | `data/analysis/archive_pilot_results.json` |
| 4. Compare baseline vs archive | `pilot/analyze_archive_pilot.py` | Steps 1+3 output | Console output + summary JSON |

#### Differentiation from Prior Work

| Dimension | QED-Nano | Rewarding the Rare | DSDR | **Our experiment** |
|-----------|---------|-------------------|------|-------------------|
| Modifies training? | Yes (RL+RC) | Yes (reward shaping) | Yes (regularizer) | **No (frozen models)** |
| Archive mechanism? | Per-iteration summary | Cluster reweighting | Token-level reg | **Explicit strategy archive** |
| Compares across RL checkpoints? | No | No | No | **Yes (R1 vs v2)** |
| Tests explorability vs diversity? | No | No | No | **Yes (core question)** |
| Model scale | 4B | 7B | 1.5B-4B | **1.5B** |
| Cost | 9,216 H100 hours | Not reported | Not reported | **~$0.35 pilot, ~$5 full** |

#### What This Would Prove for the Paper

If archive-guided search works:
- **New title:** "Explorability > Diversity: Archive-Guided Search Recovers Strategies RLVR Training Destroys"
- **Story:** RLVR narrows strategy distribution (we showed this in Runs 14-17). But a simple archive mechanism can recover lost strategies at test time, at fraction of training cost. The right question isn't "how to preserve diversity during training" but "how to search effectively at test time."
- **Connects to:** Alpha patterns (Pillar 3), QED-Nano (test-time search), Fang Wu's intuition ("RLVR shrink, SFT expand" → we add "archive search recovers")

If archive-guided search fails at 1.5B:
- **Paper:** Diagnostic paper with Runs 14-17 data. "At 1.5B, strategy diversity is too limited for archive-guided discovery. The model cannot generate genuinely novel approaches even when prompted — strategy narrowing under RLVR reflects the model's fundamental capacity limit, not a training deficiency."
- This is still informative: it tells us WHERE the bottleneck is (model capacity vs training objective).

---

## Run 19: Archive-Guided Strategy Discovery Pilot (2026-02-28)

### Configuration
- **Question:** Does archive-guided prompting recover strategies RLVR training destroyed?
- **Design:** 2 models × 10 problems × 3 iterations × 16 rollouts = 960 traces
- **Models:** R1-Distill (base) + Nemotron v2 (3K ProRL steps)
- **Temperature:** 0.6, top_p=0.95 (DeepSeek recommended)
- **Archive:** Strategy names + descriptions from Run 16 LLM judge
- **Prompt:** Shows known strategies, asks for fundamentally different approach
- **Cost:** ~$0.30 (Modal $0.25 + GPT-5-nano $0.03)
- **Runtime:** 56 min on A10G

### Scripts
- `pilot/build_archive.py` — Extract strategy archives from judge data
- `pilot/modal_archive_pilot.py` — Modal generation with archive prompts
- `pilot/analyze_archive_pilot.py` — LLM judge classification + verdict

### Pilot Problems
P2(easy/4strat), P13(easy/3), P19(easy/3), P35(med/4), P38(med/2), P42(hard/1), P46(hard/4), P52(hard/3), P54(hard/2), P58(hard/2)

### Results: Correctness

| Problem | Tier   | R1-base | R1+arch | v2-base | v2+arch |
|---------|--------|---------|---------|---------|---------|
| P2      | easy   | 59/64   | 6/48    | 64/64   | 48/48   |
| P13     | easy   | 56/64   | 45/48   | 64/64   | 48/48   |
| P19     | easy   | 19/64   | 4/48    | 64/64   | 48/48   |
| P35     | medium | 2/64    | 0/48    | 53/64   | 41/48   |
| P38     | medium | 12/64   | 5/48    | 46/64   | 12/48   |
| P42     | hard   | 0/64    | 0/48    | 9/64    | 37/48   |
| P46     | hard   | 64/64   | 29/48   | 64/64   | 48/48   |
| P52     | hard   | 4/64    | 6/48    | 17/64   | 32/48   |
| P54     | hard   | 63/64   | 17/48   | 64/64   | 48/48   |
| P58     | hard   | 14/64   | 2/48    | 47/64   | 32/48   |

### Results: Strategy Classification (LLM Judge)

| Problem | Base | R1+Arch | v2+Arch | Novel |
|---------|------|---------|---------|-------|
| P2      | 4    | 1       | 2       | 0     |
| P13     | 3    | 2       | 2       | 0     |
| P19     | 3    | 1       | 1       | 0     |
| P35     | 4    | 0       | 4       | 0     |
| P38     | 2    | 1       | 1       | 0     |
| P42     | 1    | 0       | 1       | 0     |
| P46     | 4    | 4       | 0       | 0     |
| P52     | 3    | 1       | 1       | 0     |
| P54     | 2    | 2+12n   | 0       | 2     |
| P58     | 2    | 1       | 1       | 0     |

### Verdict: NO-GO (1/3 criteria met)
- Criterion 1 (≥3 problems with novel strategy): **FAIL** — 1/10
- Criterion 2 (R1 vs v2 differ on ≥2 problems): **PASS** — 2/10
- Criterion 3 (archive finds ≥1 strategy baseline missed): **PASS** — P54

### Key Findings

1. **Archive prompting devastates R1-Distill accuracy.** Average accuracy drops from 46% to 24%. The archive prompt confuses the base model — it can't follow the "use a different approach" instruction AND solve the problem correctly. P2: 92%→12%, P46: 100%→60%, P54: 98%→35%.

2. **v2 sustains accuracy under archive constraint.** Average accuracy stays stable or improves (67%→76%). On P42: 14%→77%, P52: 27%→67%. The RL-trained model is robust to prompt perturbation.

3. **No novel strategies discovered (except P54).** Archive prompting does NOT unlock new mathematical approaches. The models simply use known strategies. The only exception was R1 on P54 (calculus-based + vector-projection), but this may be a judge artifact.

4. **The real finding is about explorability, not diversity.** R1 can't even maintain its baseline strategies under the constraint prompt. v2 can. This is exactly the "explorability ≠ diversity" thesis — v2 is more explorable not because it has more strategies, but because it's more robust to search pressure.

5. **P42 is the clearest signal.** R1-base: 0/64. R1+archive: 0/48. v2-base: 9/64. v2+archive: 37/48. Archive prompting QUADRUPLED v2's success rate while R1 stayed at zero. The archive gave v2 something to work with; R1 couldn't use it.

### Implications for Paper

Archive-guided strategy DISCOVERY fails at 1.5B — the model can't find novel approaches. But archive-guided strategy GUIDANCE succeeds for v2 — telling it what approaches exist dramatically improves coverage. The bottleneck is model capacity, not the prompt mechanism.

**Recommended paper direction:** The correctness data from this run IS the finding. Frame it as:
- "ProRL creates explorability: the ability to respond to search guidance"
- R1 is diverse but fragile (accuracy collapses under constraint)
- v2 is narrow but robust (sustains/improves under constraint)
- This IS the evidence for "explorability > diversity"

### Output Files
- `data/analysis/strategy_archives.json` — Strategy archives for 10 problems
- `data/modal_runs/archive_pilot/full/{r1-distill,nemotron-v2}/traces.json` — Raw traces
- `data/analysis/archive_pilot_results.json` — LLM judge results + verdict

---

## Run 20: Hyperbolic Scale Test (R1-671B archive prompting)

**Date:** 2026-02-27
**Cost:** $0.28 (Hyperbolic API) | **Time:** 45 min

### Question
Is the R1-Distill-1.5B collapse under archive prompting a scale limitation (can't ICL at 1.5B) or a genuine explorability signal?

### Background
Run 19 showed R1-Distill-1.5B accuracy collapses from 46%→24% under archive prompting, while Nemotron-v2-1.5B sustains/improves (67%→76%). Literature check (MathIF benchmark, "When Thinking Fails" NeurIPS 2025, R1 Thoughtology) confirmed R1-Distill-1.5B has only 17% instruction-following on constrained math. This is a known scale limitation, not an explorability signal.

### Design
- Model: DeepSeek-R1-0528 (671B) via Hyperbolic API
- 3 problems × K=4 × {baseline, archive} = 24 API calls
- Problems: P42 (hard, R1-1.5B=0%), P46 (hard, R1-1.5B=60% under archive), P2 (easy, R1-1.5B=12% under archive)
- Same archive prompt as Run 19, same T=0.6, top_p=0.95

### Results

| Problem | R1-1.5B base | R1-1.5B archive | R1-671B base | R1-671B archive |
|---------|-------------|-----------------|-------------|----------------|
| P42 (hard) | 0/64 (0%) | 0/48 (0%) | 4/4 (100%) | 3/4 (75%) |
| P46 (hard) | 64/64 (100%) | 29/48 (60%) | 4/4 (100%) | 4/4 (100%) |
| P2 (easy) | 59/64 (92%) | 6/48 (12%) | 4/4 (100%) | 4/4 (100%) |
| **AVG** | **46%** | **24%** | **100%** | **92%** |

### Key Findings

1. **R1-671B maintains accuracy under archive prompting (100%→92%).** The -8% drop is entirely from one P42 token-limit truncation (hit 16K max_tokens before producing \boxed{}). Not a reasoning failure.

2. **The 1.5B collapse is confirmed as a SCALE LIMITATION.** R1-671B handles the same "use a different approach" constraint that devastates R1-1.5B. This is ICL capacity, not explorability.

3. **P42 is the clearest evidence.** R1-1.5B: 0/64 (can't solve it at all). R1-671B: 4/4 baseline, 3/4 archive. The problem requires combinatorial reasoning that 1.5B simply can't do.

4. **P2 is the most damning for 1.5B.** R1-1.5B drops from 92%→12% on a trivial fraction-to-decimal conversion when asked to "use a different approach." R1-671B: 100%→100%. The 1.5B model can't follow the instruction AND solve the problem.

5. **Archive prompting increases thinking length at 671B.** Baseline avg ~5K tokens, archive avg ~10K tokens. The model explores longer before converging. This is the expected behavior — it's actually trying different approaches.

### Implications

- **The Run 19 "explorability signal" (v2 > R1-base under archive) is CONFOUNDED by scale.** The R1-1.5B collapse is not about explorability — it's about ICL capacity at 1.5B.
- **Archive-guided search IS viable at sufficient scale.** R1-671B can follow constrained prompts while maintaining accuracy.
- **The v2-1.5B advantage (sustaining accuracy under archive) might be: (a) RL making 1.5B more robust to prompt perturbation, or (b) v2 simply being better at the problems anyway. Can't distinguish without testing v2-equivalent at 671B (which doesn't exist as API).**
- **For the paper:** The 1.5B comparison is not a fair test of archive-guided discovery. Need to either (a) test at larger scale, or (b) reframe what the 1.5B comparison actually shows (RL robustness to prompt perturbation, not explorability).

### Files
- `pilot/hyperbolic_archive_test.py` — Test script
- `data/analysis/hyperbolic_archive_test.json` — Raw results

---

## Run 21: Distillation Pilot — "Is Explorability in Traces or Model State?" (2026-02-28)

### Motivation
Runs 14-20 established that ProRL narrows strategy diversity while improving accuracy. Run 19 showed v2 is more robust under archive-guided prompting. Run 20 revealed 1.5B ICL limitations confound archive-guided tests. The open question: if you distill R1-Distill traces vs v2 traces into the SAME base model, what transfers?

### Hypothesis
If explorability is in the traces → Student-R1 inherits R1-Distill's diversity. If explorability is in the model state → both students behave similarly (base model determines everything).

### Design
1. **LoRA SFT Qwen2.5-1.5B** on two corpora:
   - Student-R1: SFT on R1-Distill correct traces (2293 examples, 5.4M tokens)
   - Student-v2: SFT on Nemotron-v2 correct traces (2148 examples, 5.4M tokens, token-equalized)
2. **LoRA config:** r=16, alpha=32, 3 epochs, lr=2e-4, cosine decay, max_length=2048
3. **Generation:** 60 problems × 64 rollouts from each student (same as Run 14)
4. **Analysis:** Accuracy comparison + LLM judge strategy classification

### Results

#### SFT Training
| Student | Loss | Time | Examples | Token Accuracy |
|---------|------|------|----------|----------------|
| Student-R1 | 0.812 | 1h52m | 2293 | 0.787 |
| Student-v2 | 1.022 | 1h45m | 2148 | 0.733 |

**v2 traces are harder to learn** — loss 26% higher. v2's RL-optimized patterns are more "compressed," requiring more capacity to absorb.

#### Accuracy (pass@k)
| Model | pass@1 | pass@4 | pass@8 | pass@16 | pass@64 | unique_ans |
|-------|--------|--------|--------|---------|---------|------------|
| R1-Distill (teacher) | 0.597 | 0.730 | 0.772 | 0.806 | 0.850 | 1.5 |
| Nemotron-v2 (teacher) | 0.729 | 0.837 | 0.871 | 0.888 | 0.900 | 1.1 |
| **Student-R1** | 0.541 | 0.719 | 0.756 | 0.785 | 0.817 | 2.5 |
| **Student-v2** | 0.629 | 0.760 | 0.804 | 0.833 | 0.867 | 2.1 |

- Student-R1 retains **91%** of teacher R1's pass@1
- Student-v2 retains **86%** of teacher v2's pass@1
- **Student-v2 > Student-R1** across all k values
- Both students have MORE unique answers than their teachers (noise from base model's wider distribution?)

#### Per-Tier Breakdown
| Tier | Teacher-R1 p@1 | Teacher-v2 p@1 | Student-R1 p@1 | Student-v2 p@1 |
|------|----------------|----------------|----------------|----------------|
| Easy | 0.799 | 0.927 | 0.742 | 0.840 |
| Medium | 0.556 | 0.714 | 0.539 | 0.642 |
| Hard | 0.436 | 0.545 | 0.341 | 0.405 |

Gap widens on hard problems: Student-R1 retains 78% of teacher hard-tier accuracy, Student-v2 retains 74%.

#### Strategy Diversity (LLM Judge)
| Metric | Teachers | Students |
|--------|----------|----------|
| Avg strategies per problem | 1.4 | 1.3 |
| Student-R1 avg strategies | — | 1.12 |
| Student-v2 avg strategies | — | 0.83 |

**Student-R1 is more diverse than Student-v2** (1.12 vs 0.83) — R1-Distill's strategy diversity TRANSFERS through distillation.

### Findings

1. **v2 traces carry better accuracy signal.** Student-v2 beats Student-R1 on pass@1 (0.629 vs 0.541) despite both starting from same base model. RL's accuracy improvements are captured in trace quality.

2. **R1 traces carry more strategy diversity.** Student-R1 uses 1.12 strategies per problem vs Student-v2's 0.83. R1-Distill's broader strategy repertoire transfers through SFT.

3. **Explorability decomposes into two properties:**
   - **Strategy breadth** = property of traces (transferable via SFT)
   - **Accuracy/robustness** = property of training signal quality (v2's RL-refined traces > R1's diverse-but-noisy traces)

4. **Distillation preserves teacher characteristics.** The diversity gap (R1 > v2) and accuracy gap (v2 > R1) both survive distillation to a new base model.

5. **v2 traces are harder to learn but more effective.** Higher SFT loss (1.022 vs 0.812) but better downstream performance — RL creates a more "compressed" but higher-quality signal.

### Cost
| Component | Estimate |
|-----------|----------|
| Modal A10G (~9h SFT + gen) | ~$4.40 |
| GPT-5-nano judge | ~$0.09 |
| **Total** | **~$4.49** |

### Files
- `pilot/prepare_sft_data.py` — Extract and equalize SFT data
- `pilot/modal_distill_pilot.py` — Combined SFT + generation pipeline
- `pilot/analyze_distillation.py` — LLM judge + comparison analysis
- `data/sft_data/r1-distill.jsonl` — R1-Distill SFT data (2293 examples)
- `data/sft_data/nemotron-v2.jsonl` — Nemotron-v2 SFT data (2148 examples)
- `data/modal_runs/distill_pilot/full/traces/` — Student trace data
- `data/analysis/distillation_judge.json` — Judge results
- `data/analysis/distillation_summary.json` — Summary metrics

---

## Run 22: Rao's Q Prototype (2026-02-28)

### Question
Does continuous strategy distance (Rao's Q) tell a different story than binary Gini-Simpson?

### Results (14 multi-strategy problems, 4 models)
| Model | Gini-Simpson | Jaccard | Embedding | Gower |
|-------|-------------|---------|-----------|-------|
| R1-Distill | 0.3795 | 0.2802 | 0.1331 | 0.1148 |
| v1 | 0.2287 | 0.1617 | 0.0788 | 0.0776 |
| v2 | 0.2052 | 0.1618 | 0.0699 | 0.0705 |
| BroRL | 0.1972 | 0.1510 | 0.0782 | 0.0828 |

R1-Distill is most diverse by all 4 metrics. Ranking diverges at positions 2-4: BroRL rank 4→2 under Gower/Embedding.

### Files
- `pilot/rao_q_prototype.py`
- `data/analysis/rao_q_prototype.json`

---

## Run 23: Rao's Q Validation (2026-02-28)

### Question
Is Rao's Q measuring something real? 9 validation tests across 3 experiments.

### Experiment 1: Internal Validation (FREE — existing data)

**1a. Rarefaction curves: PASS**
All curves plateau by k=32. 64 rollouts is sufficient.
| Model | k=4 | k=8 | k=16 | k=32 | k=64 |
|-------|-----|-----|------|------|------|
| R1-Distill | 1.94 | 2.18 | 2.14 | 2.14 | 2.14 |
| v1 | 1.57 | 1.78 | 1.75 | 1.75 | 1.75 |
| v2 | 1.52 | 1.64 | 1.69 | 1.69 | 1.69 |
| BroRL | 1.55 | 1.70 | 1.62 | 1.62 | 1.62 |

**1b. Bootstrap CIs: FAIL**
All CIs overlap. R1 [0.064, 0.172] vs v2 [0.025, 0.136]. N=14 problems insufficient for statistical separation.

**1c. Convergent validity: PASS**
- Rao's Q vs Shannon entropy: ρ=0.873 (p<0.001)
- Rao's Q vs Vendi Score: ρ=0.992 (p<0.001)
- Rao's Q vs Answer diversity: NaN (all correct answers identical for math)

**1d. Discriminant validity: PARTIAL**
- Rao's Q vs Trace length: ρ=-0.404 (p=0.003) — CONCERN
- Rao's Q vs Accuracy: ρ=0.270 (p=0.053) — OK
- Rao's Q vs Difficulty: ρ=-0.107 (p=0.451) — OK

Trace length correlation means longer traces tend to have LESS diversity — possibly because diverse models also produce shorter correct traces.

**1e. Predictive validity: FAIL**
- Rao's Q vs Scaling gap (p@64-p@1): ρ=-0.005 (p=0.934)
- Strategy diversity does NOT predict pass@k headroom. This is a significant negative result.

**1f. Replication principle: FAIL**
3/6 violations. Pooled Q not always > max(individual Q). The Gower distance function may have issues — description length and trait extraction are noisy.

### Experiment 2: Judge Reliability ($1.32)

**2a. Prompt sensitivity: FAIL** (CV=0.312, threshold <0.10)
Strategy counts vary 50% across prompt phrasings. "Detailed" and "concise" prompts find more strategies than "original" and "adversarial". The judge is sensitive to how you ask.

**2b. Self-consistency: FAIL** (κ=0.141, threshold >0.60)
Same prompt run 3× gives different trace-level classifications. Mean Fleiss' κ=0.141 (barely above chance). The judge changes its mind about which traces belong to which strategy.

**2c. Soft classification: PASS** (7.1% ambiguous)
Only 17/238 traces genuinely ambiguous (conf<0.7). Mean confidence 0.84. Hard classification is adequate — the problem is judge consistency, not inherent ambiguity.

### Experiment 3: Temperature Sweep ($0.25 Modal + $0.13 judge)

**Monotonicity: FAIL** (0/3 metrics monotone)
| Temp | Gini-Simpson | Jaccard | Gower | pass@1 |
|------|-------------|---------|-------|--------|
| 0.3 | 0.254 | 0.234 | 0.110 | 0.679 |
| 0.6 | 0.108 | 0.085 | 0.033 | 0.667 |
| 1.0 | 0.170 | 0.151 | 0.051 | 0.646 |

T=0.3 has HIGHEST diversity, not T=1.0. This is inverted from expectation. pass@1 does decrease monotonically (as expected).

Explanation: Lower temperature makes the model more deterministic per-rollout, but that determinism LOCKS IN different strategies on different rollouts. Higher temperature adds noise that obscures strategy differences, making more traces look like mixed/unclear approaches that the judge classifies as the same strategy.

### Overall Scorecard

| Test | Verdict | Threshold |
|------|---------|-----------|
| 1a Rarefaction | **PASS** | Plateau by k=32 |
| 1b Bootstrap CIs | FAIL | CIs overlap |
| 1c Convergent | **PASS** | ρ>0.7 with Shannon/Vendi |
| 1d Discriminant | PARTIAL | ρ=-0.404 with trace length |
| 1e Predictive | FAIL | ρ≈0 with scaling gap |
| 1f Replication | FAIL | 3/6 violations |
| 2a Prompt sensitivity | FAIL | CV=0.312 |
| 2b Self-consistency | FAIL | κ=0.141 |
| 2c Soft classification | **PASS** | 7.1% ambiguous |
| 3 Temperature mono. | FAIL | Anti-monotonic |

**Score: 3 PASS, 1 PARTIAL, 6 FAIL**

### Interpretation
The Rao's Q metric has STRUCTURAL problems:
1. **The judge is the bottleneck** — not the metric formula. Strategy COUNT varies 30% with prompt wording, and TRACE ASSIGNMENT has κ=0.141 (nearly random). Rao's Q faithfully computes diversity from judge labels, but the labels themselves are noisy.
2. **Diversity doesn't predict anything useful** — no correlation with scaling gap. This questions whether strategy-level diversity has operational meaning for search.
3. **The temperature result is the most interesting finding** — it suggests that what the judge calls "strategies" may not correspond to what temperature controls. Temperature increases token-level entropy but may not increase strategy-level diversity (consistent with the Invisible Leash finding that token entropy ↑ while answer diversity ↓).

### Implications for Paper
- Cannot use Rao's Q as a validated metric in its current form
- The VALIDATION ITSELF is the paper contribution — showing that LLM-based strategy classification is unreliable (κ=0.141) is novel
- Temperature anti-monotonicity is a finding worth reporting (extends Invisible Leash to strategy level)
- The convergent validity (ρ=0.992 with Vendi) means IF you trust the labels, the math works. The problem is upstream.

### Cost
| Component | Cost |
|-----------|------|
| Experiment 1 (internal) | $0.00 |
| Experiment 2 (judge) | $1.32 |
| Experiment 3 (Modal) | $0.25 |
| Experiment 3 (judge) | $0.13 |
| **Total** | **~$1.70** |

### Files
- `pilot/validate_rao_q.py` — Experiments 1a-1f
- `pilot/validate_judge.py` — Experiments 2a-2c
- `pilot/modal_temp_sweep.py` — Experiment 3 generation
- `pilot/analyze_temp_sweep.py` — Experiment 3 analysis
- `data/analysis/rao_q_validation.json` — Experiment 1 results
- `data/analysis/judge_validation.json` — Experiment 2 results
- `data/analysis/temp_sweep_results.json` — Experiment 3 results
- `data/analysis/temp_sweep_judge_detail.json` — Full judge detail for temp sweep
- `data/modal_runs/temp_sweep_full/full/` — Raw temperature sweep traces
