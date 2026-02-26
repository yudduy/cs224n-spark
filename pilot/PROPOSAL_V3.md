# Spark-MCTS: Unified Logit-Space Trigger and Gradient Routing for Exploration-Preserving RLVR

*Research Proposal v3 — Updated 2026-02-23*

*Changes from v2: Gate ordering fix (Gate 3 before Gate 2 in τ_p computation); V(R) epistemic gate added to backward pass; Lp-Reg coefficient corrected to β=0.1; mass balance prediction updated to match pilot data; new open mathematical questions section; regret bound claim removed.*

---

## The Problem, Stated from First Principles

RLVR trains a language model by generating rollouts, scoring them with a verifier, and using the advantage signal to reinforce successful trajectories and suppress unsuccessful ones. The mechanism fails in two distinct ways that prior work has not simultaneously addressed.

**Failure Mode 1 — Unsampled Coupling** (formalized by BroRL, 2510.01180): At small rollout size N, the heavy-tailed distribution of correct reasoning paths means most rare-but-correct continuations are never sampled. Because probability mass must sum to 1, reinforcing the sampled (frequently mediocre) tokens passively erodes the mass of unvisited tokens. BroRL's Lemma 2 gives the exact term: for a token with probability p in N draws, the expected unsampled second-moment contribution is E[S] = p²(1-p)^N. This decays to zero only as N → ∞. BroRL's solution: scale N to 512 to achieve stochastic coverage of the heavy tail.

**Failure Mode 2 — Failed-Trajectory Suppression**: Tokens that *are* sampled but appear in unsuccessful rollouts receive active negative gradients. Because GRPO's update scales as 1/π, low-probability tokens in failed trajectories receive the hardest suppression — precisely the tokens with the highest exploratory value. This is a distinct problem from unsampled coupling: these tokens were visited, but visiting them in failure destroys them. BroRL at N=512 does not specifically solve this.

Three independent lines of prior work (branching, token identification, gradient protection) have each attacked one piece. No prior method addresses both failure modes with a unified mechanism calibrated to the same token coordinates. That is this proposal.

---

## Chronological Literature Foundation

**2024 — GRPO and the Baseline Problem**
DeepSeek-Math establishes GRPO as the dominant RLVR algorithm: sample G rollouts per prompt, compute group-relative advantages, update the policy. The core weakness: every token in a trajectory inherits the terminal outcome reward uniformly. A formatting space in a successful trajectory receives the same positive gradient as the decisive logical step that solved the problem. The 1/π scaling means this pathology is worst precisely on the lowest-probability tokens — the ones with the most exploratory signal.

**June 2025 — The Forking Token Discovery (Wang et al., arXiv 2506.01939, NeurIPS 2025)**
The empirical foundation for Gate 1. Wang et al. analyze token-level entropy distributions across CoT reasoning and find that roughly 20% of tokens exhibit high entropy, and these determine reasoning direction. Critically: "RLVR primarily adjusts the entropy of high-entropy tokens, while the entropy of low-entropy tokens fluctuates only within a narrow range." Further: "tokens with higher initial entropy in the base model tend to exhibit larger increases in entropy after RLVR" — meaning RLVR pushes high-entropy positions toward more uncertainty, not less. Restricting gradient updates to top-20% entropy tokens matches full-gradient performance on 8B and substantially outperforms it on 14B and 32B (+11.04 AIME'25 points at 32B). The overlap between the base model's top-20% entropy positions and the RLVR-converged model's top-20% positions remains above 86% even at step 1360 — confirming that RLVR does not fundamentally reorganize which positions are uncertain, it amplifies existing uncertainty at those positions. This establishes **Gate 1**: H > τ_h is not arbitrary — it is the empirically validated filter for positions that matter.

**July 2025 — FR3E: Entropy-Triggered Branching (Zheng et al., arXiv 2507.07017)**
FR3E takes the forking token insight into the sampling loop: identify high-entropy positions, halt generation, branch from there. This is the operational ancestor of the forward-pass branching in this proposal. FR3E's limitation, documented by ROSE: raw entropy picks up formatting artifacts and structural punctuation. Two branches from "can" and "need" at the same entropy position generate near-identical downstream reasoning — wasted compute with vanishing advantage differentials.

**August 2025 — CURE: Parallel Re-Concatenation Infrastructure (Li et al., arXiv 2508.11016)**
CURE formalizes the two-stage branching algorithm: identify high-entropy critical tokens via top-K entropy selection, truncate the sequence at that position, re-concatenate the problem prefix, generate N2 parallel branches from this new prompt. The re-concatenation mechanics in verl are clean and open-source. CURE's trigger is entropy-only — same false positive problem as FR3E. CURE's *infrastructure* (parallel prefix re-concatenation, branch batching in vLLM) is what this proposal imports directly. The modification is replacing CURE's top-K entropy trigger with the 3-gate filter.

**September 2025 — Tree-GRPO: Process Supervision from Tree Structure (Ji et al., arXiv 2509.21240)**
Tree-GRPO demonstrates that shared-prefix branching enables step-level credit assignment from outcome-only rewards: "the tree-structured trajectory naturally allows construction of step-wise process supervised signals even using only the outcome reward." Also establishes the initialize-then-expand pattern — multiple chains initialized in parallel, expanded in parallel rounds — that resolves the sequential bottleneck concern and preserves vLLM batching efficiency. Confirms that tree-based GRPO is implementable in the online RL loop without catastrophic throughput loss.

**September 2025 — SIREN: Two-Gate Entropy Masking (Jiang et al., arXiv 2509.25133)**
SIREN identifies that naive entropy regularization increases entropy of low-entropy majority tokens, which are irrelevant to reasoning. SIREN applies a two-step entropy masking: a top-p mask (eliminating extreme tails) and a peak-entropy mask (restricting regularization to high-entropy positions). This is a 2-gate filter applied only to the regularization objective — not to branching, not to asymmetric loss routing. The π condition is absent. SIREN demonstrates +6.6 maj@k on AIME24/25 with Qwen2.5-Math-7B over entropy-only baselines, confirming that selective entropy intervention outperforms global entropy manipulation. Distinguished from this proposal: SIREN does not use π as a condition and applies its filter symmetrically across success/failure.

**October 2025 — BroRL: Rollout Scaling via Unsampled Coupling Analysis (Liu et al., arXiv 2510.01180)**
BroRL provides the formal mathematical framework for small-N RLVR failure. The mass balance equation for correct token probability change decomposes into three terms: a direct reinforcement term, a negative "unsampled coupling" term, and a variance term. BroRL's Lemma 2 proves the unsampled coupling term E[S] = p²(1-p)^N, which decreases monotonically in N. "Increasing N dampens the negative unsampled coupling term in the policy update, ensuring a more reliable learning signal." BroRL scales N from 16 to 512, achieving hardware efficiency (throughput doubles from 36.5 to 72.4 samples/s) alongside learning efficiency.

Critically: BroRL addresses Failure Mode 1 (unsampled coupling) via stochastic coverage. It does not specifically address Failure Mode 2 (failed-trajectory suppression of sampled tokens). Spark-MCTS addresses both modes through orthogonal mechanisms: deterministic visitation for Mode 1, asymmetric gradient routing for Mode 2.

**October 2025 — Lp-Reg: Reasoning Spark Protection (Huang et al., arXiv 2510.03222)**
Lp-Reg defines "reasoning sparks" empirically — tokens like "Wait", "Alternatively", "Perhaps" that initiate diverse reasoning paths and reside in the low-probability tail of the distribution. These sparks satisfy: π < dynamic threshold (ρ-th percentile) AND π > κ·max(π) (above the min-p noise floor). Lp-Reg demonstrates that standard GRPO extinguishes sparks in failed trajectories by suppressing their probability, leading to measurable entropy collapse. The remedy: a KL regularization term protecting qualified tokens from gradient suppression in failed trajectories. This establishes **Gate 3** and the backward-pass protection mechanism. Importantly, Lp-Reg's KL term operates on tokens that *were sampled and appeared in failed branches* — it intercepts active negative gradients. It cannot protect tokens that were never sampled (unsampled coupling). This distinction is mechanistically precise.

**January 2026 — ROSE: Semantic Diversity Branching (Zhao et al., arXiv 2601.05053)**
ROSE documents FR3E's failure mode empirically: entropy-based branching generates semantically homogeneous branches that "share nearly identical reasoning patterns," yielding vanishing advantage differentials. ROSE's fix is semantic clustering of complete rollouts. This is orthogonal to the 3-gate approach: ROSE operates at the semantic level post-hoc across complete rollouts; the 3-gate filter operates at the logit level during generation. Both address the same branching waste problem through different mechanisms. Critically: ROSE's insight that k formatting synonyms are indistinguishable from k logical paths in logit-space is also resolved by the V(R) gate introduced in this proposal — see below.

**February 2026 — STAPO: Spurious Token Masking (Ma et al., arXiv 2602.15620)**
STAPO identifies the complementary pathology to Lp-Reg: tokens simultaneously low-entropy AND low-probability in *successful* trajectories receive pathological positive gradient amplification via the 1/π explosion. STAPO's S2T mask zeros these out. Crucially for this proposal: spurious tokens are defined by low H. Since Gate 1 requires H > τ_h, spurious tokens fail Gate 1 by definition. The 3-gate mask on successful branches simultaneously implements 80/20 gradient focus (Wang et al.) and STAPO spurious suppression (Ma et al.) in a single operation — one mask, two effects, no code redundancy.

**February 2026 — DEEP-GRPO: Dual-Stream Optimization with Prefix Masking (Guo et al., arXiv 2602.14169)**
DEEP-GRPO applies dual-stream optimization with a stop-gradient mask on shared prefixes: "gradient masking to shared prefixes facilitates stable policy learning." DEEP-GRPO's suffix treatment is uniform standard GRPO — no token-level routing within generated branches. The prefix stop-gradient is imported directly into this proposal. DEEP-GRPO's per-sequence pivot identification is not imported — the 3-gate filter replaces it with a logit-space criterion that operates during generation rather than post-hoc on completed trajectories.

---

## The 3-Gate Filter: Formal Definition and Correct Evaluation Order

A token at position t qualifies as a **reasoning spark** if and only if it satisfies all three conditions simultaneously. **The gates must be applied in the order: Gate 3 → Gate 2 → Gate 1.** Applying Gate 2 before Gate 3 produces quantile drag — see below.

**Gate 3 — Noise Floor Exclusion (applied first)** (from Lp-Reg 2510.03222 min-p filter):
π(t) > κ · max_v(π_v), where κ = 0.02.

*Why first:* Positions where the model is confident (max(π) ≈ 0.91) but a rare token was sampled (π ≈ 0.009) — "silent forks" — produce extreme left-tail π values. If Gate 2 is computed before Gate 3, these silent fork positions dominate the left tail and drag τ_p down to ≈0.192, making genuine sparks (π ≈ 0.233) fail Gate 2. Empirically verified: simulation produces 0% spark capture when Gate 2 is applied to the full distribution. Applying Gate 3 first eliminates silent fork positions from the τ_p denominator, anchoring the quantile to the viable semantic candidate distribution.

**Gate 2 — Greedy Departure (applied to Gate-3 survivors)** (from Lp-Reg 2510.03222):
π(t) < τ_p, where τ_p is the bottom-30% probability percentile computed *only over positions that cleared Gate 3*.

```python
# Correct order of operations:
gate3_mask = pi_all_positions > kappa * max_pi_per_position
tau_p = np.percentile(pi_all_positions[gate3_mask], 30)
gate2 = pi_all_positions < tau_p
```

*Why:* Low π proves the model sampled against its own dominant hypothesis. High-π tokens at a high-entropy position represent distributions peaked elsewhere — the sampled token is a formatting variant or minor paraphrase of the greedy choice. Lp-Reg's empirical analysis shows reasoning pivots are structurally low-probability at generation time because the greedy path leads to continuation of the current reasoning direction, not reconsideration of it.

*Empirical note:* At 95th percentile entropy threshold, Gates 1 and 2 are empirically correlated in the 1.5B model — every token clearing Gate 1 also clears Gate 2 (zero Gate 2 failures in the recalibration run). This is a model-scale artifact: at 1.5B, extreme-entropy positions are also extreme-probability-departure positions. At 7B+, the entropy/probability relationship decouples and Gate 2 contributes independent filtering. This will be documented in the paper as a scale limitation and tested in the aspirational 7B experiment.

**Gate 1 — Cognitive Bifurcation (applied last)** (from Wang et al. 2506.01939):
H(t) > τ_h, where τ_h is the top-5% entropy percentile computed per-sequence on the target model.

*Why:* High entropy proves the model is at a genuine decision point. Low-entropy tokens are deterministic continuations. Gate 1 eliminates STAPO's spurious (low-H, low-π) tokens by definition — they cannot satisfy Gate 1. The 5% (95th percentile) threshold for branching, rather than the 80/20 paper's 20% for gradient masking, reflects different use-case requirements: 20% is appropriate for gradient routing across an entire trajectory; 2-3% is the correct selectivity for identifying positions worth halting generation and re-branching from. Our pilot confirmed: 80th percentile produced 13.1% spark fraction (too dense); 95th percentile produced 2.88% (correctly characterizes density of genuine reasoning pivots).

**The conjunction:** A qualifying spark is uncertain (high H), exploratory (low π vs. viable semantic candidates), and viable (above noise floor). The gate ordering ensures that the τ_p quantile is computed over the semantically meaningful subspace rather than the full distribution including noise.

---

## Architecture: Spark-MCTS v3

### Forward Pass — Spark-Triggered Branching (addresses Failure Mode 1)

1. Generate N1=4 scout rollouts in parallel from the problem prompt
2. At each token position, apply the 3-gate filter **in order: Gate 3 → Gate 2 → Gate 1**
3. When a token clears all three gates, record the (position, prefix) pair
4. At the first qualifying spark position: halt N1 rollouts, re-concatenate the problem prefix up to position t, generate N2=4 parallel branches from this truncated prefix
5. Each branch is a complete trajectory from position t to the terminal verifier

**How this addresses Failure Mode 1:** BroRL achieves stochastic coverage by sampling N=512 times, hoping to encounter rare-but-correct continuations. Spark-MCTS achieves *deterministic visitation*: rather than randomly sampling until reaching a heavy-tail position, we navigate analytically to the exact logit-space coordinate where the heavy tail diverges from the greedy head, then branch from there. This is not equivalent to N=512 coverage — it is complementary. Stochastic coverage guarantees eventual visitation with probability; deterministic visitation guarantees visitation at the identified coordinate with certainty, but requires calibration accuracy of the 3-gate filter.

### Backward Pass — Asymmetric Gradient Routing with V(R) Epistemic Gate (addresses Failure Mode 2)

After the verifier scores all N2 branches, compute GRPO advantages within the branch group. Then apply the two-stage routing:

**Stage 1 — V(R) Epistemic Gate:**

Compute the variance of terminal rewards across N2 branches at the trigger position:

```python
# Before applying any asymmetric routing:
if np.var(branch_rewards) > 0:
    # Epistemic fork: branches diverged in outcome
    # → activate asymmetric routing
else:
    # Aleatoric fork: branches yielded identical outcomes
    # → apply uniform GRPO, no Lp-Reg protection
```

*Why V(R) = 0 means aleatoric:* If k formatting synonyms ("Wait" vs "However") are the true choice at a high-entropy position, both continuations pursue identical logic and yield identical terminal rewards. If k genuine logical paths diverge at a position, they will produce different outcomes with nonzero probability. The variance of terminal rewards therefore separates epistemic forks (V(R) > 0) from aleatoric formatting choices (V(R) = 0) without requiring a trained Process Reward Model. This resolves the k ≤ m Gibbs indistinguishability problem — when k logical paths equal m formatting synonyms, logit-space (H, π) cannot distinguish them, but value-space (V(R)) can.

*Caveat:* At N2=4, V(R) is noisy. All 4 branches can succeed or fail by chance even at a genuine epistemic fork, yielding V(R)=0 spuriously. However, this case is already handled by GRPO's advantage normalization: if all branches have identical rewards, all advantages are zero and neither the positive-branch mask nor Lp-Reg would apply meaningful gradient signal regardless. The V(R) gate formalizes what the loss function would do anyway. The noisiness of V(R) at N2=4 is a known limitation; the aspirational experiment at N2=8 would make this estimate substantially more reliable.

**Stage 2 — Asymmetric Routing (applied only when V(R) > 0):**

**Successful branches (A > 0):**
Apply the GRPO loss with the Gate-1 mask: weight = 1 for tokens satisfying H > τ_h, weight = 0 otherwise. Because Gate 1 requires high H, this single operation simultaneously concentrates gradient on reasoning-determining tokens (80/20 result) and zeros gradient on spurious low-H tokens (STAPO). One mask, two effects.

*Note on using Gate 1 only for positive routing:* Successful branches use Gate 1 (not all three gates) because we want to reinforce all high-entropy decisions along a successful path, including those with high π. A choice between two highly probable algebraic steps is still a cognitive fork worth reinforcing. Gate 1 correctly identifies all such positions. The strict 3-gate filter is reserved for the protection case where targeting precision matters most — protecting a false positive from Lp-Reg wastes regularization; missing a true spark from positive routing has lower cost.

**Failed branches (A < 0):**
Apply standard GRPO loss, but for the trigger token (the specific token at position t that triggered the branch), apply Advantage-Weighted NLL protection with α=0.3 and asymmetric clipping.

*Why Advantage-Weighted NLL instead of Lp-Reg KL:* Lp-Reg's KL gradient on the target token scales as β(π_θ - π_ref). For a rare spark token where both π_θ ≈ 0.01 and π_ref ≈ 0.01, the restoring force is ≈ β × 0.00 — nearly zero. The mechanism designed to protect rare tokens provides essentially no gradient to the rarest tokens. The NLL gradient is -α|A|(1-π_θ) ≈ α|A| for small π_θ — rarity-invariant. Protection scales with advantage magnitude, not with how probable the token already is.

*α=0.3 requires verification:* The pre-flight checklist includes a 3-value coefficient sweep (0.1, 0.3, 0.5) on 5 training steps to confirm protection fires (mean > 0.005) while pg_loss dominates (ratio ≥ 3.0).

**Shared prefix (all branches):**
Apply the DEEP-GRPO prefix stop-gradient mask: zero gradient for all tokens before position t. Prevents branches from providing contradictory gradient signals on shared prefix tokens.

---

## Threshold Implementation: Decoupled Intervention and Logging

**Intervention trigger (dynamic, per-sequence):** At each forward pass, apply gates in order: Gate 3 (fixed κ=0.02), then compute τ_p as the bottom-30% percentile over Gate-3-surviving positions, then compute τ_h as the top-5% entropy percentile within the sequence. Per-sequence relative thresholds match Wang et al.'s implementation and guarantee the filter identifies relative cognitive forks regardless of the model's absolute confidence at any training step.

**Metric logging (fixed, absolute):** At every 5-step interval, compute |S_t| using fixed calibrated constants τ_h = 1.2800, τ_p = 0.6237, κ = 0.02. Write to log only — no gradient effect. Fixed ruler enables interpretable time-series comparison across conditions.

---

## Experiments

### Pilot Results — Calibration Validated (Runs 1-3)

**Run 1 (80th percentile):** Spark fraction 13.1%, 62.1 sparks/rollout. Top sparks dominated by stopwords. Branch accuracy 10.0% < initial 12.5%. Verdict: threshold too permissive.

**Run 2 (95th percentile):** Spark fraction 2.88%, 15.1 sparks/rollout. Branch accuracy 11.1% ≥ initial 10.5%. Branch diversity Jaccard 0.058 (near-non-overlapping branches). Entropy-only false positive rate 42.2%. Zero Gate 2 failures (correlated with Gate 1 at 1.5B — scale limitation). Infrastructure validated.

**Run 3 — Backward pass smoke test:** All 5 checks pass across all 3 conditions. Lp-Reg fires every step (mean |loss| = 0.072 at β=1.0 — above pg_loss mean of 0.061, flagged for coefficient reduction). Prefix stop-gradient zeros 89.6% of prefix positions. Grad norm 3.6-16.3, no NaN/Inf.

**Run 4 — 50-step ablation:** All three conditions ran cleanly. Key findings:
- Entropy: A: 0.97→0.27, B: 1.24→0.36, C: 1.27→0.50. C retains most entropy.
- Mass balance drift: A: 7.17%, B: 4.33%, C: 4.33%. B and C are *identical* — branching alone explains the mass balance win. The 3-gate filter and Lp-Reg do not provide additional mass balance benefit beyond branching. They contribute only to entropy preservation.
- Lp-Reg loss (0.074) > pg_loss (0.061). Regularizer is dominant — causal direction of entropy preservation ambiguous.
- No accuracy numbers collected. Entropy preservation without accuracy curves cannot distinguish healthy exploration from suppressed learning.

**Pre-flight fixes required before full run:**
1. Gate ordering fix: compute τ_p after Gate 3 filter
2. V(R) gate: add one-line epistemic check before applying asymmetric routing
3. Replace Lp-Reg KL with Advantage-Weighted NLL (α sweep at 0.1, 0.3, 0.5)
4. Add pass@1 accuracy logging every 10 steps
5. Restrict protection to trigger token only

### Pre-Commitment Viability Run (~$15-20)

Before the full $80-120 run: 100 steps on Conditions A and C only, with all pre-flight fixes applied, accuracy logged every 10 steps. Success criterion: Condition C pass@1 ≥ Condition A pass@1 at step 100. This answers the load-bearing question — does entropy preservation translate to accuracy — without committing to the full run.

### Main Experiment — 3-Condition Ablation (~$80-100)

**Model:** Qwen2.5-Math-1.5B (full fine-tuning, no LoRA)
**Dataset:** DAPO-Math-17K, filtered for MATH Level 3-4 (target 30-40% baseline accuracy)
**Platform:** Modal, 1×A100-80GB
**Duration:** 300 steps per condition (not 50 — 50 steps is not enough for convergence signal)
**Batch:** 16 problems per step, N1=4 scout rollouts, N2=4 branches per spark

**Condition A — Vanilla GRPO:**
Standard GRPO, no branching, no routing. 16 problems × 4 rollouts = 64 rollouts per step. All tokens treated equally.

**Condition B — Entropy-Only Branching (FR3E/CURE):**
Same branching infrastructure; entropy-only trigger (top-5% per sequence). Uniform GRPO on all suffix tokens. Prefix stop-gradient applied. Isolates branching benefit without routing.

**Condition C — Spark-MCTS v3 (full proposal):**
3-gate trigger (Gate 3 → Gate 2 → Gate 1 order), V(R) epistemic gate, asymmetric backward routing (Gate-1 mask on successful suffixes, Advantage-Weighted NLL at α=0.3 on trigger token in failed branches), prefix stop-gradient.

### Metrics

**Primary — Three-Flow Spark Decomposition (logged every 5 steps, fixed thresholds):**

- **Extinction rate:** Fraction of step-t sparks whose π falls below Gate 3 at step t+1. Protection target failure mode. Prediction: high in A, reduced in B, near-zero in C.
- **Assimilation rate:** Fraction of step-t sparks whose π rises above τ_p at step t+1. Expected lower than intuition suggests — Quadrant II dynamics (low-π + positive advantage → entropy increases, distribution spreads) rarely cleanly push a single token above τ_p.
- **Renewal rate:** Count of tokens entering the 3-gate set at step t that were not present at step t-1. A healthy system replaces assimilated sparks with new ones.

**Secondary — Mass Balance Metric, Σπ_t (logged every 10 steps):**

Before training, run forward passes on 50 held-out MATH Level 3-4 problems. Record spark token IDs and their probabilities π_0. At steps 0, 10, 20, ..., 300: run inference on same problems and measure Σπ_t for that exact token set.

*Updated prediction, consistent with pilot data:* Branching (Conditions B and C) is expected to produce better Σπ_t stability than A, primarily because branching generates more diverse advantage signals that stabilize mass balance. The B vs. C divergence on Σπ_t is the test of whether the V(R) gate + asymmetric routing adds value beyond branching alone. If B=C on Σπ_t at 300 steps (as at 50 steps), the protection mechanism contributes only to entropy, not mass balance. This is a narrower claim and should be stated as such.

**Primary accuracy metric (logged every 10 steps):**
Pass@1 on a fixed 100-problem MATH Level 3-4 holdout. This is the load-bearing evaluation. All other metrics are mechanistic — pass@1 is the bottom line.

**Additional secondary:**
- Sequence-level policy entropy (tracks |S_t| with a lag)
- Gradient magnitude ratio: pg_loss / loss_protect (verify α=0.3 gives ratio ≥ 3.0)
- V(R) gate statistics: fraction of branching events with V(R)>0 vs. V(R)=0 per step (measures the epistemic fork rate)
- Branch trigger precision: fraction of CURE/FR3E entropy-only triggers that qualify as 3-gate sparks

**Kill criteria:**
- B and C show no three-flow divergence and no pass@1 divergence after 150 steps → effect below detection at 1.5B; report as scale limitation
- Condition B shows no advantage over A in any metric → branching implementation broken
- pg_loss / loss_protect ratio < 1.5 at any step → α too high; interrupt and re-run with α=0.15

### Aspirational Full Experiment (for external researchers / follow-up)

**Extended ablation at 7B:** Reproduce Conditions A-C on Qwen2.5-Math-7B at 300 steps. At 7B: Gate 2 is expected to decouple from Gate 1 (the 1.5B correlation artifact resolves), enabling a test of the full 3-gate conjunction's independent contribution. The V(R) gate is also more reliable at 7B where the model has sufficient capability to produce genuinely divergent branch outcomes.

**Condition D — Routing without branching:** Apply the asymmetric backward routing to vanilla GRPO rollouts (no branching). Tests whether routing mechanism has value independent of branching structure.

**Condition E — BroRL + 3-gate routing:** Apply Spark-MCTS's backward routing on top of BroRL's N=512 rollouts. BroRL addresses Mode 1; Condition C addresses Mode 2. If orthogonal, combination should outperform either alone — direct test of the two-mode decomposition thesis.

**Condition F — ROSE + 3-gate routing:** Combine ROSE's semantic diversity branching trigger with the 3-gate backward routing. If orthogonal (ROSE fixes branching semantic diversity, 3-gate fixes gradient routing), combination should outperform either alone.

**N2 scaling:** Test N2=8 vs N2=4 for the V(R) estimate reliability. At N2=4, V(R)=0 can occur spuriously. N2=8 substantially reduces this noisiness.

---

## Open Mathematical Questions

These are genuine unresolved issues, not claims:

**1. Advantage normalization at N2=4.** GRPO computes A_i = (r_i - mean(r)) / std(r) within the group. At N2=4, std(r) is computed over 4 samples — statistically noisy. When V(R)=0 (all same outcome), std=0 and division is undefined; the V(R) gate handles this case by bypassing routing. But even at V(R)>0, a 4-sample group advantage is a weak signal. The full experiment should log the within-group reward variance at each branch event to characterize how often the advantage estimate is well-conditioned.

**2. NLL direction.** This proposal uses -α|A|·log π_θ restricted to the trigger token (Advantage-Weighted NLL). The gradient is -α|A|(1-π_θ), which is rarity-invariant. However, the interaction between this term and verl's GRPO log-ratio clipping needs verification: if GRPO already clips the log-ratio for the trigger token, the NLL term adds a second gradient signal on the same parameter. Verify that the combined gradient doesn't produce instability — the coefficient sweep addresses this empirically.

**3. Prefix stop-gradient coverage.** The smoke test showed 89.6% of prefix positions zeroed (10.4% residual attributed to padding). This should be verified against expected prefix length distribution — if average prefix is 80% of sequence length, 89.6% coverage is approximately correct; if prefix is shorter, the 10.4% residual may include non-padding positions receiving unintended gradients.

**4. The k ≤ m boundary condition.** The Gibbs indistinguishability simulation (k=3, m=3) proves the filter cannot separate epistemic from aleatoric uncertainty in logit-space when formatting synonyms equal logical paths. The V(R) gate solves this post-hoc. What the simulation does not test: the k >> m regime (many logical paths, few synonyms) where the filter should work correctly without V(R), and the k < m regime (few logical paths, many synonyms) where even V(R) at N2=4 may be insufficient. The full experiment should measure the V(R)>0 rate and interpret it as an estimate of the epistemic fork fraction in the training distribution.

---

## The Novel Claim, Stated Precisely

RLVR at small rollout size (N=16) suffers from two mechanistically distinct failure modes: (1) unsampled coupling (BroRL, 2510.01180) and (2) failed-trajectory suppression (Lp-Reg, 2510.03222). BroRL addresses Mode 1 via stochastic coverage at N=512 but does not specifically address Mode 2. No prior method addresses both modes with a unified mechanism.

Spark-MCTS addresses Mode 1 via *deterministic visitation*: the 3-gate filter (applied in Gate 3 → Gate 2 → Gate 1 order to avoid quantile drag from silent forks) navigates to the heavy-tail logit-space coordinate where rare-but-correct continuations diverge from the greedy head, then forces branching from that position. Spark-MCTS addresses Mode 2 via *V(R)-gated asymmetric gradient routing*: the V(R) epistemic gate distinguishes genuine logical forks (V(R)>0) from formatting synonyms (V(R)=0) using Monte Carlo outcome variance, then applies Advantage-Weighted NLL protection only to confirmed epistemic pivots — preventing active suppression of the trigger token without wasting regularization budget on aleatoric formatting choices.

The pilot data updates the claim's scope: branching alone (Condition B) explains the mass balance improvement over vanilla GRPO. The Spark-MCTS-specific contribution (Condition C minus B) is entropy preservation, not mass balance. The load-bearing test is whether this entropy difference translates to better pass@1 at convergence.

**The aporia (stated honestly):** Analytic shielding introduces calibration fragility. If Gate 3's κ=0.02 admits any hallucination tokens, the protection mechanism permanently embeds them — the NLL term prevents the optimizer from washing them out. BroRL's unbiased stochastic sampling organically washes out garbage via repeated negative feedback. The V(R) gate mitigates this: a hallucinated token that somehow cleared all three gates but consistently appears in branches with V(R)=0 will never receive protection, because its branches always yield identical (wrong) outcomes. But a hallucinated token that appears in a branch where other branches succeed (V(R)>0 by coincidence) will receive spurious protection. The 2% min-p floor is empirically calibrated, not theoretically derived. This is the fundamental cost of the analytic approach and cannot be engineered away without reducing to stochastic coverage.

---

## Calibrated Thresholds

Fixed thresholds for metric logging (not for the dynamic intervention trigger):
- τ_h = 1.2800 (95th percentile entropy, Qwen2.5-Math-1.5B on MATH Level 3-4)
- τ_p = 0.6237 (30th percentile probability over Gate-3-surviving positions)
- κ = 0.02 (min-p noise floor, from Lp-Reg paper)

The intervention trigger uses per-sequence dynamic thresholds computed fresh at each forward pass, with Gate 3 applied before τ_p computation.

---

## References (Chronological)

1. DeepSeek-Math / GRPO (2024)
2. Wang et al., "Beyond the 80/20 Rule," arXiv 2506.01939, NeurIPS 2025
3. Zheng et al., FR3E, arXiv 2507.07017 (July 2025)
4. Li et al., CURE, arXiv 2508.11016 (August 2025)
5. Ji et al., Tree-GRPO, arXiv 2509.21240 (September 2025)
6. Jiang et al., SIREN, arXiv 2509.25133 (September 2025)
7. Liu et al., BroRL, arXiv 2510.01180 (October 2025)
8. Huang et al., Lp-Reg, arXiv 2510.03222 (October 2025)
9. Zhao et al., ROSE, arXiv 2601.05053 (January 2026)
10. Yuksekgonul et al., TTT-Discover, arXiv 2601.16175 (January 2026)
11. Ma et al., STAPO, arXiv 2602.15620 (February 2026)
12. Guo et al., DEEP-GRPO, arXiv 2602.14169 (February 2026)
