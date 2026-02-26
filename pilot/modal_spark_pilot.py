"""
Spark-MCTS Pilot Calibration
=============================
Validates the 3-gate filter for identifying reasoning sparks.
Inference-only diagnostic — no training. ~$5 on Modal.

The 3-Gate Filter (from Spark-MCTS proposal):
  Gate 1 — Cognitive Bifurcation:  H(t) > τ_h
    Source: Wang et al. 2506.01939 — top-20% entropy tokens carry reasoning signal
  Gate 2 — Greedy Departure:       π(t) < τ_p
    Source: Huang et al. 2510.03222 — low-prob tokens are exploratory departures
  Gate 3 — Noise Floor Exclusion:  π(t) > κ · max_v(π_v)
    Source: Huang et al. 2510.03222 — min-p filter separates sparks from noise

A token clearing all 3 gates is: uncertain (high H), exploratory (low π),
and viable (above noise floor) — an abductive fork.

Success criteria:
  - Spark fraction: 2-8% of tokens
  - Sparks are semantically meaningful reasoning pivots
  - Re-concatenation from spark positions produces valid branches

Usage: modal run pilot/modal_spark_pilot.py
GPU: 1× A100-80GB (~1-2 hours, ~$5)
"""

import modal
import json
import os
import math

# ============================================================
# MODAL SETUP
# ============================================================

app = modal.App("spark-mcts-pilot")

# Lessons learned (from modal-build.md):
# - vllm==0.8.4 pins torch==2.6.0; install vllm first
# - Python 3.11 for pilot (no flash-attn needed for inference-only)
# - transformers <5.0.0 for vllm 0.8.4 compatibility
pilot_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.4")  # Installs torch==2.6.0 as dependency
    .pip_install(
        "transformers>=4.45.0,<5.0.0",
        "datasets",
        "numpy",
        "tqdm",
        "huggingface_hub",
    )
)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("spark-pilot-results", create_if_missing=True)

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Model: 1.5B — same we'll train, for accurate calibration
    "model_id": "Qwen/Qwen2.5-Math-1.5B",
    "max_model_len": 4096,
    "dtype": "bfloat16",
    # gpu_memory_utilization=0.85: KV cache needs ~1.75GB for inference batches;
    # leave headroom per gpu-optimization.md
    "gpu_memory_utilization": 0.85,
    # Generation
    "num_problems": 50,
    "n1": 4,  # rollouts per problem
    "n2": 3,  # branches per spark (re-concat test only)
    "temperature": 1.0,
    "max_new_tokens": 2048,
    "branch_max_new_tokens": 1024,
    "vllm_logprobs": 20,  # top-K logprobs per token (vLLM 0.8.4 max=20)
    # 3-Gate threshold targets (calibrated from data)
    # τ_h: Wang et al. 2506.01939 validated 80th percentile for 80/20 result
    "entropy_percentile": 80,
    # τ_p: 30th percentile is a forward-pass branching threshold, NOT Lp-Reg's 1%.
    # Lp-Reg's 1% selects tokens for backward-pass KL in training.
    # 30% identifies the "greedy departure" region during generation.
    # Pilot will verify this produces 2-8% sparks; adjust if not.
    "prob_percentile": 30,
    # κ: min-p noise floor from Lp-Reg (verified from CarlanLark/Lp-Reg-dev)
    "kappa": 0.02,
    # CURE comparison (for false-positive analysis)
    "cure_top_k": 20,
    # Re-concatenation test
    "reconcat_problems": 10,
    # Kill criteria
    "kill_spark_fraction_low": 0.01,  # <1% → gates too strict
    "kill_spark_fraction_high": 0.15,  # >15% → gates too loose
}

# ============================================================
# REASONING TOKEN CLASSIFICATION
# Empirically validated set from modal_pilot.py
# These are tokens CURE should branch at and Lp-Reg should protect.
# ============================================================

REASONING_TOKENS = {
    "wait",
    "however",
    "but",
    "actually",
    "alternatively",
    "instead",
    "let",
    "check",
    "verify",
    "reconsider",
    "note",
    "recall",
    "suppose",
    "assume",
    "consider",
    "hmm",
    "think",
    "perhaps",
    "try",
    "maybe",
    "otherwise",
    "yet",
    "still",
    "therefore",
    "hence",
    "thus",
    "because",
    "since",
    "although",
    "hold",
    "whereas",
    "while",
    "unless",
    "rather",
    "moreover",
    "oops",
    "careful",
    "mistake",
    "error",
    "wrong",
    "correct",
    "revisit",
    "wait,",
    "however,",
    "but,",
    "actually,",
    "so,",
    "now,",
}


def classify_token(token_text):
    """Classify token as reasoning / noise / ambiguous."""
    text = token_text.strip().lower()
    if text in REASONING_TOKENS:
        return "reasoning"
    for pat in REASONING_TOKENS:
        if (text.startswith(pat) or pat.startswith(text)) and len(text) >= 3:
            return "reasoning"
    if len(text) <= 1:
        return "noise"
    if text.replace(".", "").replace(",", "").replace("-", "").isdigit():
        return "noise"
    if text in {"\\n", "\\t", "\\\\", "$$", "\\[", "\\]", "\\(", "\\)", "```", "---"}:
        return "noise"
    return "ambiguous"


# ============================================================
# ANSWER VERIFICATION (from modal_pilot.py)
# ============================================================


def extract_boxed_answer(text):
    """Extract answer from \\boxed{...}. Handles nested braces."""
    import re

    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


def verify_answer(predicted, ground_truth):
    """Verify answer correctness."""
    if predicted is None:
        return False
    try:
        pred = predicted.strip().replace(" ", "").replace(",", "")
        gt = ground_truth.strip().replace(" ", "").replace(",", "")
        if pred == gt:
            return True
        try:
            return abs(float(pred) - float(gt)) < 1e-6
        except ValueError:
            pass
        return False
    except Exception:
        return False


# ============================================================
# DATA LOADING
# ============================================================


def load_math_level5(num_problems):
    """Load MATH Level 5 problems for calibration."""
    from datasets import load_dataset

    # Primary: MATH-500 (Parquet-based, no trust_remote_code needed)
    # level column is int64: 1-5 (not string "Level 5")
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        problems = []
        for row in ds:
            if row.get("level", 0) == 5:
                problems.append(
                    {
                        "problem": row["problem"],
                        "answer": row["answer"],
                        "type": row.get("subject", "unknown"),
                    }
                )
        if len(problems) >= num_problems:
            print(
                f"  Loaded {num_problems} MATH Level 5 problems from MATH-500 (of {len(problems)} available)"
            )
            return problems[:num_problems]
        print(
            f"  MATH-500 Level 5: only {len(problems)}, need {num_problems}. Including Level 4..."
        )
        for row in ds:
            if row.get("level", 0) == 4:
                problems.append(
                    {
                        "problem": row["problem"],
                        "answer": row["answer"],
                        "type": row.get("subject", "unknown"),
                    }
                )
        if len(problems) >= num_problems:
            print(f"  Loaded {num_problems} MATH Level 4-5 problems from MATH-500")
            return problems[:num_problems]
    except Exception as e:
        print(f"  MATH-500 failed: {e}")

    # Fallback: MATH-Hard (has "solution" not "answer" — parse \boxed{})
    try:
        ds = load_dataset("lighteval/MATH-Hard", split="test")
        problems = []
        for r in ds:
            answer = extract_boxed_answer(r.get("solution", ""))
            if answer:
                problems.append(
                    {
                        "problem": r["problem"],
                        "answer": answer,
                        "type": r.get("type", "hard"),
                    }
                )
        if len(problems) >= num_problems:
            print(f"  Loaded {num_problems} problems from MATH-Hard")
            return problems[:num_problems]
    except Exception as e:
        print(f"  MATH-Hard failed: {e}")

    raise RuntimeError("Could not load enough MATH Level 5 problems")


# ============================================================
# CORE: PER-TOKEN STATISTICS
# ============================================================


def compute_token_stats(token_ids, pos_logprobs_list, tokenizer):
    """
    Compute per-token statistics for 3-gate analysis.

    For each token position t, computes:
      H(t)     — Shannon entropy from top-K logprobs (approximate lower bound)
      π(t)     — probability of the sampled token
      max_π(t) — max probability in the distribution (exact: true max always in top-K)

    NOTE: H(t) is an underestimate for diffuse distributions since we only see top-20
    logprobs. This means τ_h will be set conservatively (lower than true 80th %ile).
    Direction is safe: if sparks pass with approximate H, they definitely pass with true H.
    """
    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    stats = []

    for t, (tid, pos_lps) in enumerate(zip(token_ids, pos_logprobs_list)):
        if pos_lps is None or len(pos_lps) == 0:
            stats.append(None)
            continue

        token_id = tid if isinstance(tid, int) else int(tid)

        # Entropy from top-K logprobs
        log_probs_vals = list(pos_lps.values())
        probs = [math.exp(lp) for lp in log_probs_vals]
        total = sum(probs)
        if total > 0:
            H = -sum((p / total) * math.log(p / total + 1e-10) for p in probs if p > 0)
        else:
            H = 0.0

        # Sampled token probability
        # vLLM 0.8.4: logprobs dict always includes the sampled token even if not in top-K
        if token_id in pos_lps:
            pi = math.exp(pos_lps[token_id])
        else:
            pi = 0.0  # Extremely rare — not even in the logprobs dict

        # Max probability (exact: the true argmax is always in top-K)
        max_pi = math.exp(max(log_probs_vals))

        token_text = tokenizer.decode(token_id)
        is_special = int(token_id) in special_ids
        is_short = len(token_text.strip()) <= 1

        stats.append(
            {
                "position": t,
                "token_id": token_id,
                "token_text": token_text,
                "H": H,
                "pi": pi,
                "max_pi": max_pi,
                "is_special": is_special,
                "is_short": is_short,
            }
        )

    return stats


# ============================================================
# CORE: THRESHOLD CALIBRATION
# ============================================================


def calibrate_thresholds(all_stats_flat, entropy_pct=80, prob_pct=30):
    """
    Compute gate thresholds from corpus-wide token statistics.

    τ_h = entropy_pct-th percentile of H (Wang et al. use per-batch; we use corpus-wide)
    τ_p = prob_pct-th percentile of π (forward-pass branching threshold)

    Returns (tau_h, tau_p, diagnostics_dict).
    """
    import numpy as np

    # Filter: non-special, non-short tokens (matching CURE's token filtering)
    valid = [
        s
        for s in all_stats_flat
        if s is not None and not s["is_special"] and not s["is_short"]
    ]
    entropies = [s["H"] for s in valid]
    probs = [s["pi"] for s in valid if s["pi"] > 0]

    tau_h = float(np.percentile(entropies, entropy_pct))
    tau_p = float(np.percentile(probs, prob_pct))

    diagnostics = {
        "n_total_tokens": len(all_stats_flat),
        "n_valid_tokens": len(valid),
        "n_with_prob": len(probs),
        "tau_h": tau_h,
        "tau_p": tau_p,
        "entropy_mean": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies)),
        "entropy_p10": float(np.percentile(entropies, 10)),
        "entropy_p50": float(np.percentile(entropies, 50)),
        "entropy_p80": float(np.percentile(entropies, 80)),
        "entropy_p90": float(np.percentile(entropies, 90)),
        "entropy_p95": float(np.percentile(entropies, 95)),
        "prob_mean": float(np.mean(probs)),
        "prob_p05": float(np.percentile(probs, 5)),
        "prob_p10": float(np.percentile(probs, 10)),
        "prob_p30": float(np.percentile(probs, 30)),
        "prob_p50": float(np.percentile(probs, 50)),
        "prob_p70": float(np.percentile(probs, 70)),
    }
    return tau_h, tau_p, diagnostics


# ============================================================
# CORE: 3-GATE FILTER
# ============================================================


def apply_3gate_filter(stats, tau_h, tau_p, kappa=0.02):
    """
    Apply the 3-gate conjunction to token statistics.

    A token at position t is a SPARK iff:
      Gate 1: H(t) > τ_h         (cognitive bifurcation — high entropy)
      Gate 2: π(t) < τ_p         (greedy departure — low probability)
      Gate 3: π(t) > κ·max_π(t)  (noise exclusion — above min-p floor)

    Also applies CURE's pre-filters: skip position 0, special tokens, single-char.

    Returns list of result dicts with gate status per position.
    """
    results = []
    for s in stats:
        if s is None:
            results.append(None)
            continue

        # CURE pre-filters (from get_critic_key_token_mask)
        if s["is_special"] or s["is_short"] or s["position"] == 0:
            results.append(
                {
                    **s,
                    "is_spark": False,
                    "g1": False,
                    "g2": False,
                    "g3": False,
                    "reason": "prefiltered",
                }
            )
            continue

        g1 = s["H"] > tau_h
        # Gate 2: require pi > 0 (pi=0 means token wasn't even in top-K logprobs)
        g2 = 0 < s["pi"] < tau_p
        g3 = s["pi"] > kappa * s["max_pi"]

        is_spark = g1 and g2 and g3

        results.append(
            {
                **s,
                "is_spark": is_spark,
                "g1": g1,
                "g2": g2,
                "g3": g3,
                "classification": classify_token(s["token_text"]) if is_spark else None,
            }
        )

    return results


# ============================================================
# COMPARISON: CURE ENTROPY-ONLY CANDIDATES
# ============================================================


def cure_entropy_candidates(stats, top_k=20):
    """
    CURE's entropy-only branch candidate selection.
    Source: bytedance/CURE get_critic_key_token_mask

    Returns the top-K valid high-entropy tokens (sorted by H descending),
    matching CURE's filtering: skip pos 0, skip special, skip len(text)<=1.
    """
    candidates = []
    for s in stats:
        if s is None or s["is_special"] or s["is_short"]:
            continue
        if s["position"] == 0:
            continue
        candidates.append(s)

    candidates.sort(key=lambda x: x["H"], reverse=True)
    return candidates[:top_k]


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================


def run_diagnostic(config):
    """
    Spark-MCTS pilot calibration diagnostic.

    Steps:
      1. Load MATH Level 5 problems
      2. Generate n1=4 rollouts per problem via vLLM
      3. Compute per-token stats (H, π, max_π)
      4. Calibrate thresholds τ_h, τ_p
      5. Apply 3-gate filter, measure spark fraction
      6. Compare with entropy-only (CURE baseline)
      7. Test re-concatenation from spark positions
      8. Evaluate kill criteria
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import numpy as np
    import random
    from tqdm import tqdm

    random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("  Spark-MCTS Pilot Calibration")
    print("  3-Gate Filter: H > τ_h  ∧  π < τ_p  ∧  π > κ·max(π)")
    print("=" * 70)

    # --- 1. Load data ---
    print("\n[1/7] Loading MATH Level 5 problems...")
    problems = load_math_level5(config["num_problems"])

    # --- 2. Load model + generate ---
    print(f"\n[2/7] Loading {config['model_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    llm = LLM(
        model=config["model_id"],
        tensor_parallel_size=1,
        gpu_memory_utilization=config["gpu_memory_utilization"],
        dtype=config["dtype"],
        max_model_len=config["max_model_len"],
        trust_remote_code=True,
    )

    gen_params = SamplingParams(
        temperature=config["temperature"],
        max_tokens=config["max_new_tokens"],
        logprobs=config["vllm_logprobs"],
        top_p=1.0,
    )

    sys_prompt = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    prompts = []
    prompt_texts = []
    for p in problems:
        chat = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": p["problem"]},
        ]
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        for _ in range(config["n1"]):
            prompts.append(text)
            prompt_texts.append(text)

    print(
        f"  Generating {len(prompts)} rollouts ({config['num_problems']} × {config['n1']})..."
    )
    outputs = llm.generate(prompts, gen_params)
    print(f"  Generated {len(outputs)} rollouts")

    # --- 3. Compute per-token stats ---
    print("\n[3/7] Computing per-token stats (H, π, max_π)...")
    all_stats_flat = []
    rollout_data = []

    for prob_idx, problem in enumerate(tqdm(problems, desc="  Token stats")):
        prob_rollouts = []
        for r_idx in range(config["n1"]):
            out_idx = prob_idx * config["n1"] + r_idx
            output = outputs[out_idx]
            completion = output.outputs[0]

            text = completion.text
            tids = list(completion.token_ids)

            # Extract per-position logprobs
            pos_lps = []
            for lp_obj in completion.logprobs or []:
                if lp_obj is None:
                    pos_lps.append(None)
                else:
                    pos_lps.append({tid: info.logprob for tid, info in lp_obj.items()})

            # Compute token stats
            stats = compute_token_stats(tids, pos_lps, tokenizer)
            all_stats_flat.extend([s for s in stats if s is not None])

            # Score
            predicted = extract_boxed_answer(text)
            correct = verify_answer(predicted, problem["answer"])

            prob_rollouts.append(
                {
                    "r_idx": r_idx,
                    "text": text,
                    "token_ids": tids,
                    "pos_logprobs": pos_lps,
                    "stats": stats,
                    "is_correct": correct,
                    "predicted": predicted,
                    "n_tokens": len(tids),
                }
            )

        rollout_data.append(
            {
                "problem_idx": prob_idx,
                "problem_text": problem["problem"][:200],
                "ground_truth": problem["answer"],
                "type": problem.get("type", "unknown"),
                "rollouts": prob_rollouts,
            }
        )

    print(f"  Total tokens analyzed: {len(all_stats_flat)}")

    # --- 4. Calibrate thresholds ---
    print("\n[4/7] Calibrating thresholds...")
    tau_h, tau_p, cal_diag = calibrate_thresholds(
        all_stats_flat,
        entropy_pct=config["entropy_percentile"],
        prob_pct=config["prob_percentile"],
    )
    print(f"  τ_h (Gate 1, {config['entropy_percentile']}th %ile of H) = {tau_h:.4f}")
    print(f"  τ_p (Gate 2, {config['prob_percentile']}th %ile of π) = {tau_p:.6f}")
    print(f"  κ   (Gate 3, min-p floor)                   = {config['kappa']}")
    print(
        f"  Entropy distribution: mean={cal_diag['entropy_mean']:.4f}  "
        f"p50={cal_diag['entropy_p50']:.4f}  p80={cal_diag['entropy_p80']:.4f}  "
        f"p95={cal_diag['entropy_p95']:.4f}"
    )
    print(
        f"  Prob distribution:    mean={cal_diag['prob_mean']:.6f}  "
        f"p10={cal_diag['prob_p10']:.6f}  p30={cal_diag['prob_p30']:.6f}  "
        f"p50={cal_diag['prob_p50']:.6f}"
    )

    # --- 5. Apply 3-gate filter ---
    print("\n[5/7] Applying 3-gate filter...")
    total_valid = 0
    total_sparks = 0
    total_g1_pass = 0
    total_g2_pass = 0
    total_g3_pass = 0
    total_g1_only = 0  # pass G1, fail G2 or G3 (entropy-only false positives)
    total_g1_fail_g2 = 0  # pass G1, fail G2 (high-H but also high-π → formatting)
    total_g1_fail_g3 = (
        0  # pass G1+G2, fail G3 (high-H, low-π, below noise → hallucination)
    )
    all_sparks = []
    per_rollout_spark_counts = []

    # CURE comparison accumulators
    cure_total_candidates = 0
    cure_also_spark = 0

    for pd in rollout_data:
        for rollout in pd["rollouts"]:
            gate_results = apply_3gate_filter(
                rollout["stats"], tau_h, tau_p, config["kappa"]
            )
            # Store gate results on rollout for later use
            rollout["gate_results"] = gate_results

            n_sparks = 0
            for gr in gate_results:
                if gr is None or gr.get("reason") == "prefiltered":
                    continue
                total_valid += 1

                if gr["g1"]:
                    total_g1_pass += 1
                if gr["g2"]:
                    total_g2_pass += 1
                if gr["g3"]:
                    total_g3_pass += 1

                # Entropy-only false positive breakdown
                if gr["g1"] and not gr["is_spark"]:
                    total_g1_only += 1
                    if not gr["g2"]:
                        total_g1_fail_g2 += 1  # High-H, high-π → formatting variant
                    elif not gr["g3"]:
                        total_g1_fail_g3 += 1  # High-H, low-π, below noise floor

                if gr["is_spark"]:
                    total_sparks += 1
                    n_sparks += 1
                    all_sparks.append(
                        {
                            "problem_idx": pd["problem_idx"],
                            "problem_type": pd["type"],
                            "r_idx": rollout["r_idx"],
                            "is_correct": rollout["is_correct"],
                            **{
                                k: gr[k]
                                for k in [
                                    "position",
                                    "token_id",
                                    "token_text",
                                    "H",
                                    "pi",
                                    "max_pi",
                                    "classification",
                                ]
                            },
                        }
                    )

            per_rollout_spark_counts.append(n_sparks)

            # CURE comparison: how many of CURE's entropy-only top-K candidates
            # also pass the full 3-gate filter?
            cure_candidates = cure_entropy_candidates(
                rollout["stats"], config["cure_top_k"]
            )
            cure_total_candidates += len(cure_candidates)
            for cc in cure_candidates:
                g2 = 0 < cc["pi"] < tau_p
                g3 = cc["pi"] > config["kappa"] * cc["max_pi"]
                if g2 and g3:
                    cure_also_spark += 1

    spark_fraction = total_sparks / max(total_valid, 1)
    g1_fraction = total_g1_pass / max(total_valid, 1)
    g1_false_positive_rate = total_g1_only / max(total_g1_pass, 1)
    cure_precision = cure_also_spark / max(cure_total_candidates, 1)

    print(f"\n  Gate pass rates:")
    print(f"    Valid tokens (non-special, non-short): {total_valid}")
    print(f"    Gate 1 pass (H > τ_h):    {total_g1_pass:>7} ({g1_fraction:.1%})")
    print(
        f"    Gate 2 pass (π < τ_p):    {total_g2_pass:>7} ({total_g2_pass/max(total_valid,1):.1%})"
    )
    print(
        f"    Gate 3 pass (π > κ·max):  {total_g3_pass:>7} ({total_g3_pass/max(total_valid,1):.1%})"
    )
    print(f"    3-Gate SPARKS:            {total_sparks:>7} ({spark_fraction:.2%})")
    print(
        f"    Sparks per rollout:       {np.mean(per_rollout_spark_counts):.1f} ± {np.std(per_rollout_spark_counts):.1f}"
    )

    # Classification breakdown
    n_reasoning = sum(1 for s in all_sparks if s["classification"] == "reasoning")
    n_noise = sum(1 for s in all_sparks if s["classification"] == "noise")
    n_ambiguous = sum(1 for s in all_sparks if s["classification"] == "ambiguous")

    print(f"\n  Spark classification:")
    print(f"    Reasoning: {n_reasoning:>5} ({n_reasoning/max(total_sparks,1):.1%})")
    print(f"    Noise:     {n_noise:>5} ({n_noise/max(total_sparks,1):.1%})")
    print(f"    Ambiguous: {n_ambiguous:>5} ({n_ambiguous/max(total_sparks,1):.1%})")

    # --- 6. CURE comparison ---
    print(f"\n[6/7] CURE (entropy-only) comparison...")
    print(
        f"    CURE top-{config['cure_top_k']} candidates total:   {cure_total_candidates}"
    )
    print(
        f"    Of those, also 3-gate sparks:    {cure_also_spark} ({cure_precision:.1%})"
    )
    print(
        f"    G1 pass but NOT 3-gate spark:    {total_g1_only} ({g1_false_positive_rate:.1%} of G1 passes)"
    )
    print(f"      → fail G2 (high-π, formatting): {total_g1_fail_g2}")
    print(f"      → fail G3 (below noise floor):  {total_g1_fail_g3}")
    print(f"    Entropy-only false positive rate: {g1_false_positive_rate:.1%}")

    # Spark vocabulary analysis
    spark_vocab = {}
    for s in all_sparks:
        text = s["token_text"].strip().lower()
        spark_vocab[text] = spark_vocab.get(text, 0) + 1
    top_sparks_vocab = sorted(spark_vocab.items(), key=lambda x: x[1], reverse=True)[
        :30
    ]

    print(f"\n  Top 30 spark tokens:")
    print(f"  {'Token':>20}  {'Count':>6}  {'Class':>12}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*12}")
    for token, count in top_sparks_vocab:
        cls = classify_token(token)
        print(f"  {repr(token):>20}  {count:>6}  {cls:>12}")

    # Manual inspection: sparks with context
    print(f"\n  Manual inspection (first 30 sparks with context):")
    print(
        f"  {'P#':>3}  {'R':>1}  {'Pos':>5}  {'Correct':>7}  {'Class':>10}  "
        f"{'Token':>15}  {'H':>7}  {'π':>9}  {'Context'}"
    )
    print(
        f"  {'-'*3}  {'-'*1}  {'-'*5}  {'-'*7}  {'-'*10}  "
        f"{'-'*15}  {'-'*7}  {'-'*9}  {'-'*30}"
    )
    shown = 0
    for pd in rollout_data:
        if shown >= 30:
            break
        for rollout in pd["rollouts"]:
            if shown >= 30:
                break
            for gr in rollout.get("gate_results", []):
                if gr is None or not gr.get("is_spark"):
                    continue
                if shown >= 30:
                    break
                pos = gr["position"]
                ctx_start = max(0, pos - 10)
                ctx_tids = rollout["token_ids"][ctx_start:pos]
                ctx_text = "".join(
                    tokenizer.decode(t if isinstance(t, int) else int(t))
                    for t in ctx_tids
                )
                ctx_text = ctx_text[-40:]
                cls = classify_token(gr["token_text"])
                correct_str = "Y" if rollout["is_correct"] else "N"
                print(
                    f"  {pd['problem_idx']:>3}  {rollout['r_idx']:>1}  "
                    f"{pos:>5}  {correct_str:>7}  {cls:>10}  "
                    f"{repr(gr['token_text']):>15}  {gr['H']:.4f}  "
                    f"{gr['pi']:.2e}  ...{repr(ctx_text)}"
                )
                shown += 1

    # --- 7. Re-concatenation test ---
    print(f"\n[7/7] Re-concatenation test ({config['reconcat_problems']} problems)...")

    branch_gen_params = SamplingParams(
        temperature=config["temperature"],
        max_tokens=config["branch_max_new_tokens"],
        logprobs=config["vllm_logprobs"],
        top_p=1.0,
    )

    branch_prompts = []
    branch_meta = []

    for pd in rollout_data[: config["reconcat_problems"]]:
        rollout = pd["rollouts"][0]  # First rollout only
        gate_results = rollout.get("gate_results", [])

        # Find first spark position past position 5
        # (need enough prefix for meaningful branching)
        spark_pos = None
        spark_token_text = "?"
        for gr in gate_results:
            if gr is not None and gr.get("is_spark") and gr["position"] > 5:
                spark_pos = gr["position"]
                spark_token_text = gr["token_text"]
                break

        if spark_pos is None:
            continue

        # CURE re-concatenation (from dapo_ray_trainer.py:215-240):
        # branched_prompt = original_prompt + response[:spark_pos]
        prefix_tids = rollout["token_ids"][:spark_pos]
        prefix_text = tokenizer.decode(prefix_tids, skip_special_tokens=False)
        out_idx = pd["problem_idx"] * config["n1"]
        branched_prompt = prompt_texts[out_idx] + prefix_text

        for b_idx in range(config["n2"]):
            branch_prompts.append(branched_prompt)
            branch_meta.append(
                {
                    "problem_idx": pd["problem_idx"],
                    "spark_pos": spark_pos,
                    "spark_token": spark_token_text,
                    "b_idx": b_idx,
                }
            )

    reconcat_results = []
    branch_correct = 0
    jaccard_sims = []

    if branch_prompts:
        n_problems_branched = len(set(m["problem_idx"] for m in branch_meta))
        print(
            f"  Generating {len(branch_prompts)} branches from {n_problems_branched} problems..."
        )
        branch_outputs = llm.generate(branch_prompts, branch_gen_params)

        for meta, b_out in zip(branch_meta, branch_outputs):
            comp = b_out.outputs[0]
            predicted = extract_boxed_answer(comp.text)
            gt = rollout_data[meta["problem_idx"]]["ground_truth"]
            correct = verify_answer(predicted, gt)

            reconcat_results.append(
                {
                    "problem_idx": meta["problem_idx"],
                    "spark_pos": meta["spark_pos"],
                    "spark_token": meta["spark_token"],
                    "b_idx": meta["b_idx"],
                    "is_correct": correct,
                    "n_tokens": len(comp.token_ids),
                    "text_preview": comp.text[:200],
                }
            )

        branch_correct = sum(1 for r in reconcat_results if r["is_correct"])

        # Branch diversity: Jaccard similarity between siblings
        from itertools import combinations

        for pid in set(m["problem_idx"] for m in branch_meta):
            siblings = [r for r in reconcat_results if r["problem_idx"] == pid]
            for a, b in combinations(siblings, 2):
                wa = set(a["text_preview"].split())
                wb = set(b["text_preview"].split())
                union = wa | wb
                if union:
                    jaccard_sims.append(len(wa & wb) / len(union))

        print(
            f"  Branch accuracy: {branch_correct}/{len(reconcat_results)} "
            f"({branch_correct/max(len(reconcat_results),1):.1%})"
        )
        if jaccard_sims:
            print(
                f"  Branch diversity (Jaccard): {np.mean(jaccard_sims):.3f} ± {np.std(jaccard_sims):.3f}"
            )
            print(f"    (lower = more diverse; <0.5 is good)")

        print(f"\n  Sample branches:")
        for r in reconcat_results[:6]:
            status = "correct" if r["is_correct"] else "wrong"
            print(
                f"    P{r['problem_idx']:02d} B{r['b_idx']} spark@{r['spark_pos']}"
                f"[{r['spark_token']!r}] {status:>7}  {r['text_preview'][:60]}..."
            )
    else:
        print("  WARNING: No spark positions found for re-concatenation test!")

    # Free GPU
    del llm
    import torch

    torch.cuda.empty_cache()
    print("  Released GPU memory")

    # ============================================================
    # RESULTS
    # ============================================================

    initial_correct = sum(
        1 for pd in rollout_data for r in pd["rollouts"] if r["is_correct"]
    )
    initial_total = len(outputs)

    results = {
        "config": config,
        "calibration": {
            "tau_h": tau_h,
            "tau_p": tau_p,
            "kappa": config["kappa"],
            **cal_diag,
        },
        "spark_analysis": {
            "total_valid_tokens": total_valid,
            "total_sparks": total_sparks,
            "spark_fraction": spark_fraction,
            "sparks_per_rollout_mean": float(np.mean(per_rollout_spark_counts)),
            "sparks_per_rollout_std": float(np.std(per_rollout_spark_counts)),
            "gate1_pass": total_g1_pass,
            "gate1_fraction": g1_fraction,
            "gate2_pass": total_g2_pass,
            "gate3_pass": total_g3_pass,
            "classification": {
                "reasoning": n_reasoning,
                "noise": n_noise,
                "ambiguous": n_ambiguous,
                "reasoning_fraction": n_reasoning / max(total_sparks, 1),
            },
        },
        "cure_comparison": {
            "cure_total_candidates": cure_total_candidates,
            "cure_also_3gate": cure_also_spark,
            "cure_precision": cure_precision,
            "entropy_only_false_positive_rate": g1_false_positive_rate,
            "g1_fail_g2_formatting": total_g1_fail_g2,
            "g1_fail_g3_noise": total_g1_fail_g3,
        },
        "reconcat_test": {
            "n_branches": len(reconcat_results),
            "n_problems_tested": (
                len(set(m["problem_idx"] for m in branch_meta)) if branch_meta else 0
            ),
            "branch_accuracy": branch_correct / max(len(reconcat_results), 1),
            "branch_diversity_jaccard": (
                float(np.mean(jaccard_sims)) if jaccard_sims else 0
            ),
            "branch_diversity_std": float(np.std(jaccard_sims)) if jaccard_sims else 0,
        },
        "accuracy": {
            "initial": initial_correct / max(initial_total, 1),
            "initial_correct": initial_correct,
            "initial_total": initial_total,
        },
        "spark_vocabulary": dict(top_sparks_vocab),
        "sample_sparks": all_sparks[:100],
        "reconcat_branches": reconcat_results[:20],
    }

    # ============================================================
    # KILL CRITERIA
    # ============================================================

    print("\n" + "=" * 70)
    print("  KILL CRITERIA")
    print("=" * 70)
    kills = []

    # 1. Spark fraction in range
    if spark_fraction < config["kill_spark_fraction_low"]:
        print(
            f"  FAIL  spark fraction {spark_fraction:.2%} < {config['kill_spark_fraction_low']:.0%}"
        )
        print(
            f"        → Gates too strict. Relax: τ_h to 70th %ile or τ_p to 40th %ile."
        )
        kills.append("spark_fraction_too_low")
    elif spark_fraction > config["kill_spark_fraction_high"]:
        print(
            f"  FAIL  spark fraction {spark_fraction:.2%} > {config['kill_spark_fraction_high']:.0%}"
        )
        print(f"        → Gates too loose. Tighten: τ_p to 20th %ile.")
        kills.append("spark_fraction_too_high")
    else:
        print(
            f"  PASS  spark fraction {spark_fraction:.2%} in "
            f"[{config['kill_spark_fraction_low']:.0%}, {config['kill_spark_fraction_high']:.0%}]"
        )

    # 2. Semantic quality: sparks should include reasoning tokens
    reasoning_fraction = n_reasoning / max(total_sparks, 1)
    if reasoning_fraction < 0.10:
        print(f"  WARN  reasoning fraction {reasoning_fraction:.1%} < 10%")
        print(f"        → Sparks are mostly not recognized reasoning tokens.")
        print(
            f"        → Check if REASONING_TOKENS set needs expansion for this model."
        )
        kills.append("low_reasoning_fraction")
    else:
        print(f"  PASS  reasoning fraction {reasoning_fraction:.1%} >= 10%")

    # 3. Re-concatenation validity
    if reconcat_results and all(not r["is_correct"] for r in reconcat_results):
        print(f"  WARN  all {len(reconcat_results)} branches incorrect")
        print(f"        → Re-concatenation may produce degenerate continuations.")
        kills.append("reconcat_all_wrong")
    elif not reconcat_results:
        print(f"  WARN  no re-concatenation test run (no sparks found)")
        kills.append("no_reconcat_test")
    else:
        print(
            f"  PASS  re-concatenation producing valid branches "
            f"({branch_correct}/{len(reconcat_results)} correct)"
        )

    # 4. Branch diversity
    if jaccard_sims and np.mean(jaccard_sims) > 0.90:
        print(
            f"  WARN  branch diversity too low (Jaccard={np.mean(jaccard_sims):.3f} > 0.90)"
        )
        print(f"        → Branches from spark positions are near-identical.")
        kills.append("low_branch_diversity")
    elif jaccard_sims:
        print(f"  PASS  branch diversity OK (Jaccard={np.mean(jaccard_sims):.3f})")

    # 5. Entropy-only false positive rate should be substantial
    # (if it's near 0%, 3-gate adds nothing over entropy-only)
    if g1_false_positive_rate < 0.10:
        print(f"  NOTE  entropy-only FP rate only {g1_false_positive_rate:.1%}")
        print(
            f"        → 3-gate filter adds little over entropy-only at this model/threshold."
        )
    else:
        print(
            f"  PASS  entropy-only FP rate {g1_false_positive_rate:.1%} → 3-gate filter adds value"
        )

    if kills:
        print(f"\n  VERDICT: {len(kills)} issue(s): {kills}")
        print(f"           Adjust thresholds before proceeding to training.")
    else:
        print(
            f"\n  VERDICT: All criteria pass. Ready for Spark-MCTS training experiment."
        )

    results["kill_criteria"] = kills

    # Print summary table for professor
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"  {'Metric':<40} {'Value':>12} {'Status':>10}")
    print(f"  {'-'*40} {'-'*12} {'-'*10}")
    print(f"  {'Model':40} {config['model_id'].split('/')[-1]:>12}")
    n_prob = config["num_problems"]
    n_roll = config["n1"]
    print(f"  {'Problems x Rollouts':40} {str(n_prob) + 'x' + str(n_roll):>12}")
    print(f"  {'Total tokens analyzed':40} {total_valid:>12,}")
    print(f"  {'τ_h (entropy threshold)':40} {tau_h:>12.4f}")
    print(f"  {'τ_p (probability threshold)':40} {tau_p:>12.6f}")
    print(f"  {'κ (min-p noise floor)':40} {config['kappa']:>12.3f}")
    print(
        f"  {'Spark fraction':40} {spark_fraction:>12.2%} {'PASS' if config['kill_spark_fraction_low'] <= spark_fraction <= config['kill_spark_fraction_high'] else 'FAIL':>10}"
    )
    print(f"  {'Sparks per rollout':40} {np.mean(per_rollout_spark_counts):>12.1f}")
    print(f"  {'Reasoning spark fraction':40} {reasoning_fraction:>12.1%}")
    print(f"  {'Entropy-only FP rate':40} {g1_false_positive_rate:>12.1%}")
    print(f"  {'  → fail G2 (formatting)':40} {total_g1_fail_g2:>12,}")
    print(f"  {'  → fail G3 (noise floor)':40} {total_g1_fail_g3:>12,}")
    print(
        f"  {'CURE top-{0} → 3-gate precision'.format(config['cure_top_k']):40} {cure_precision:>12.1%}"
    )
    print(f"  {'Initial accuracy':40} {initial_correct/max(initial_total,1):>12.1%}")
    if reconcat_results:
        print(
            f"  {'Branch accuracy':40} {branch_correct/max(len(reconcat_results),1):>12.1%}"
        )
        if jaccard_sims:
            print(f"  {'Branch diversity (Jaccard)':40} {np.mean(jaccard_sims):>12.3f}")
    print(f"  {'Kill criteria triggered':40} {len(kills):>12}")

    return results


# ============================================================
# MODAL ENTRY POINTS
# ============================================================


@app.function(
    image=pilot_image,
    gpu="A100-80GB",
    timeout=7200,  # 2 hours
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
)
def pilot_spark_calibration():
    """Run Spark-MCTS pilot calibration on Modal."""
    from datetime import datetime

    results = run_diagnostic(CONFIG)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/results/spark_pilot_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Full JSON results
    json_path = f"{out_dir}/results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 2. Spark tokens TSV (for spreadsheet analysis)
    tsv_path = f"{out_dir}/sparks.tsv"
    with open(tsv_path, "w") as f:
        headers = [
            "problem_idx",
            "type",
            "r_idx",
            "position",
            "token_text",
            "classification",
            "H",
            "pi",
            "max_pi",
            "is_correct",
        ]
        f.write("\t".join(headers) + "\n")
        for s in results.get("sample_sparks", []):
            row = [
                str(s.get("problem_idx", "")),
                s.get("problem_type", ""),
                str(s.get("r_idx", "")),
                str(s.get("position", "")),
                s.get("token_text", "").replace("\t", " ").replace("\n", " "),
                s.get("classification", ""),
                f"{s.get('H', 0):.4f}",
                f"{s.get('pi', 0):.6e}",
                f"{s.get('max_pi', 0):.4f}",
                str(s.get("is_correct", "")),
            ]
            f.write("\t".join(row) + "\n")

    # 3. Branch test TSV
    branch_path = f"{out_dir}/branches.tsv"
    with open(branch_path, "w") as f:
        headers = [
            "problem_idx",
            "b_idx",
            "spark_pos",
            "spark_token",
            "is_correct",
            "n_tokens",
            "text_preview",
        ]
        f.write("\t".join(headers) + "\n")
        for b in results.get("reconcat_branches", []):
            row = [
                str(b.get("problem_idx", "")),
                str(b.get("b_idx", "")),
                str(b.get("spark_pos", "")),
                b.get("spark_token", "").replace("\t", " "),
                str(b.get("is_correct", "")),
                str(b.get("n_tokens", "")),
                b.get("text_preview", "")[:100].replace("\t", " ").replace("\n", " "),
            ]
            f.write("\t".join(row) + "\n")

    # 4. Summary (one-liner for quick grep)
    summary_path = f"{out_dir}/summary.txt"
    sa = results["spark_analysis"]
    cc = results["cure_comparison"]
    rt = results["reconcat_test"]
    with open(summary_path, "w") as f:
        f.write(f"timestamp={ts}\n")
        f.write(f"model={CONFIG['model_id']}\n")
        f.write(f"tau_h={results['calibration']['tau_h']:.6f}\n")
        f.write(f"tau_p={results['calibration']['tau_p']:.8f}\n")
        f.write(f"kappa={CONFIG['kappa']}\n")
        f.write(f"spark_fraction={sa['spark_fraction']:.6f}\n")
        f.write(f"total_sparks={sa['total_sparks']}\n")
        f.write(f"sparks_per_rollout={sa['sparks_per_rollout_mean']:.1f}\n")
        f.write(
            f"reasoning_fraction={sa['classification']['reasoning_fraction']:.4f}\n"
        )
        f.write(f"entropy_only_fp_rate={cc['entropy_only_false_positive_rate']:.4f}\n")
        f.write(f"cure_precision={cc['cure_precision']:.4f}\n")
        f.write(f"branch_accuracy={rt['branch_accuracy']:.4f}\n")
        f.write(f"branch_diversity={rt['branch_diversity_jaccard']:.4f}\n")
        f.write(f"kill_criteria={','.join(results['kill_criteria']) or 'none'}\n")

    results_vol.commit()

    print(f"\nResults saved to Modal volume 'spark-pilot-results':")
    print(f"  {json_path}")
    print(f"  {tsv_path}")
    print(f"  {branch_path}")
    print(f"  {summary_path}")
    print(f"\nDownload all:")
    print(f"  modal volume get spark-pilot-results spark_pilot_{ts}/")

    return results["spark_analysis"]


# ============================================================
# SMOKE TEST: Minimal pipeline validation (~5 min, ~$1)
# ============================================================

SMOKE_CONFIG = {
    **CONFIG,
    "num_problems": 5,
    "n1": 2,
    "n2": 2,
    "max_new_tokens": 512,
    "branch_max_new_tokens": 256,
    "reconcat_problems": 3,
    # Relax kill criteria for smoke (small sample = noisy)
    "kill_spark_fraction_low": 0.001,
    "kill_spark_fraction_high": 0.50,
}

# Recalibration: tighten tau_h from 80th -> 95th percentile for branching use case
# Run 1 showed 13.13% sparks dominated by stopwords at 80th percentile.
# 95th targets 3-6% and should shift top sparks toward discourse markers.
RECAL_CONFIG = {
    **CONFIG,
    "entropy_percentile": 95,
    # Tighter kill criteria now that we expect fewer sparks
    "kill_spark_fraction_low": 0.01,
    "kill_spark_fraction_high": 0.10,
}


@app.function(
    image=pilot_image,
    gpu="A100-80GB",
    timeout=1800,  # 30 min max
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
)
def smoke_test():
    """Smoke test: minimal pipeline validation before full pilot."""
    import time

    start = time.time()
    print("=" * 60)
    print("SMOKE TEST: Spark-MCTS Pilot Pipeline Validation")
    print("=" * 60)
    print(
        f"  Config: {SMOKE_CONFIG['num_problems']} problems, "
        f"{SMOKE_CONFIG['n1']} rollouts, "
        f"{SMOKE_CONFIG['max_new_tokens']} max_tokens"
    )

    results = run_diagnostic(SMOKE_CONFIG)

    elapsed = time.time() - start

    # Validate outputs
    checks = []
    sa = results["spark_analysis"]

    if sa["total_valid_tokens"] > 0:
        checks.append(("tokens_analyzed", "PASS", f"{sa['total_valid_tokens']} tokens"))
    else:
        checks.append(("tokens_analyzed", "FAIL", "0 tokens"))

    if results["calibration"]["tau_h"] > 0:
        checks.append(
            ("tau_h_calibrated", "PASS", f"{results['calibration']['tau_h']:.4f}")
        )
    else:
        checks.append(("tau_h_calibrated", "FAIL", "tau_h=0"))

    if results["calibration"]["tau_p"] > 0:
        checks.append(
            ("tau_p_calibrated", "PASS", f"{results['calibration']['tau_p']:.6f}")
        )
    else:
        checks.append(("tau_p_calibrated", "FAIL", "tau_p=0"))

    if sa["total_sparks"] > 0:
        checks.append(
            (
                "sparks_found",
                "PASS",
                f"{sa['total_sparks']} sparks ({sa['spark_fraction']:.2%})",
            )
        )
    else:
        checks.append(("sparks_found", "WARN", "0 sparks (may be OK with 5 problems)"))

    if results["cure_comparison"]["cure_total_candidates"] > 0:
        checks.append(
            (
                "cure_comparison",
                "PASS",
                f"{results['cure_comparison']['cure_total_candidates']} candidates",
            )
        )
    else:
        checks.append(("cure_comparison", "FAIL", "0 CURE candidates"))

    rt = results["reconcat_test"]
    if rt["n_branches"] > 0:
        checks.append(
            ("reconcat_test", "PASS", f"{rt['n_branches']} branches generated")
        )
    else:
        checks.append(("reconcat_test", "WARN", "no branches (may be OK if no sparks)"))

    # Save smoke results
    os.makedirs("/results/smoke", exist_ok=True)
    with open("/results/smoke/smoke_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    results_vol.commit()

    print(f"\n{'=' * 60}")
    print(f"SMOKE TEST RESULTS ({elapsed:.0f}s = {elapsed/60:.1f} min)")
    print(f"{'=' * 60}")
    all_pass = True
    for name, status, detail in checks:
        marker = "[OK]" if status == "PASS" else "[!!]" if status == "FAIL" else "[??]"
        print(f"  {marker} {name}: {detail}")
        if status == "FAIL":
            all_pass = False

    if all_pass:
        print(f"\nSMOKE TEST PASSED — pipeline validated, ready for full pilot")
    else:
        print(f"\nSMOKE TEST FAILED — debug before running full pilot")

    return {
        "status": "pass" if all_pass else "fail",
        "elapsed_seconds": elapsed,
        "checks": {name: status for name, status, _ in checks},
        "spark_fraction": sa["spark_fraction"],
        "total_sparks": sa["total_sparks"],
    }


@app.function(
    image=pilot_image,
    gpu="A100-80GB",
    timeout=7200,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
)
def recalibrate():
    """Rerun pilot with tightened tau_h (95th percentile)."""
    from datetime import datetime

    results = run_diagnostic(RECAL_CONFIG)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/results/recal_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    sa = results["spark_analysis"]
    cal = results["calibration"]
    summary_lines = [
        f"timestamp={ts}",
        f"mode=recalibrate",
        f"entropy_percentile=95",
        f"model={RECAL_CONFIG['model_id']}",
        f"tau_h={cal['tau_h']:.6f}",
        f"tau_p={cal['tau_p']:.8f}",
        f"kappa={cal['kappa']}",
        f"spark_fraction={sa['spark_fraction']:.6f}",
        f"total_sparks={sa['total_sparks']}",
        f"sparks_per_rollout={sa['sparks_per_rollout_mean']:.1f}",
        f"reasoning_fraction={sa.get('reasoning_fraction', 0):.4f}",
        f"entropy_only_fp_rate={results.get('cure_comparison', {}).get('false_positive_rate', 0):.4f}",
        f"cure_precision={results.get('cure_comparison', {}).get('precision', 0):.4f}",
        f"kill_criteria={','.join(results.get('kill_criteria', [])) or 'none'}",
    ]
    with open(f"{out_dir}/summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    # Save sparks TSV
    if "per_problem" in results:
        with open(f"{out_dir}/sparks.tsv", "w") as f:
            f.write(
                "problem\trollout\tposition\ttoken\tH\tpi\tmax_pi\tg1\tg2\tg3\tis_spark\tclass\n"
            )
            for pd in results["per_problem"]:
                for ri, rollout in enumerate(pd.get("rollouts", [])):
                    for s in rollout.get("spark_details", []):
                        if s and s.get("is_spark"):
                            f.write(
                                f"{pd['problem_idx']}\t{ri}\t{s['position']}\t"
                                f"{s['token_text']}\t{s['H']:.4f}\t{s['pi']:.6e}\t"
                                f"{s['max_pi']:.6e}\t{s['g1']}\t{s['g2']}\t{s['g3']}\t"
                                f"{s['is_spark']}\t{s.get('classification', '')}\n"
                            )

    results_vol.commit()
    print(f"\nResults saved to Modal volume 'spark-pilot-results':")
    print(f"  {out_dir}/results.json")
    print(f"  {out_dir}/sparks.tsv")
    print(f"  {out_dir}/summary.txt")
    print(f"\nDownload all:")
    print(f"  modal volume get spark-pilot-results {out_dir.lstrip('/')}/")

    return {
        "tau_h": cal["tau_h"],
        "tau_p": cal["tau_p"],
        "spark_fraction": sa["spark_fraction"],
        "total_sparks": sa["total_sparks"],
        "sparks_per_rollout": sa["sparks_per_rollout_mean"],
        "reasoning_fraction": sa.get("reasoning_fraction", 0),
        "kill_criteria": results.get("kill_criteria", []),
    }


@app.local_entrypoint()
def main(mode: str = "full"):
    """
    Launch Spark-MCTS pilot.

    Args:
        mode: "smoke" | "full" | "recal"
    """
    if mode == "smoke":
        print("Launching SMOKE TEST on Modal (1x A100-80GB)...")
        result = smoke_test.remote()
        print(f"\nSmoke test: {result['status'].upper()}")
        print(f"  Elapsed: {result['elapsed_seconds']:.0f}s")
        print(f"  Sparks: {result['total_sparks']} ({result['spark_fraction']:.2%})")
        for check, status in result["checks"].items():
            print(f"  {check}: {status}")
    elif mode == "recal":
        print("Launching RECALIBRATION on Modal (1x A100-80GB)...")
        print("  entropy_percentile: 80 -> 95")
        print("  Expected: spark fraction 3-6%, discourse markers as top sparks")
        result = recalibrate.remote()
        print("\n" + "=" * 70)
        print("  RECALIBRATION COMPLETE")
        print("=" * 70)
        print(f"  tau_h: {result['tau_h']:.4f}")
        print(f"  tau_p: {result['tau_p']:.6f}")
        print(f"  Spark fraction: {result['spark_fraction']:.2%}")
        print(f"  Sparks/rollout: {result['sparks_per_rollout']:.1f}")
        print(f"  Reasoning fraction: {result['reasoning_fraction']:.2%}")
        kills = result.get("kill_criteria", [])
        if kills:
            print(f"  Kill criteria: {kills}")
        else:
            print(f"  Kill criteria: NONE (all pass)")
        print(f"\nDownload: modal volume get spark-pilot-results")
    elif mode == "full":
        print("Launching Spark-MCTS pilot calibration on Modal (1× A100-80GB)...")
        summary = pilot_spark_calibration.remote()
        print("\n" + "=" * 70)
        print("  PILOT COMPLETE")
        print("=" * 70)
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    if isinstance(vv, float):
                        print(f"    {kk}: {vv:.4f}")
                    else:
                        print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")
        print("\nDownload full results:")
        print("  modal volume get spark-pilot-results")
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'smoke' or 'full'.")
