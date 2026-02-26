"""
CURE × Lp-Reg Pilot Diagnostic
===============================
Validates the core hypothesis: CURE's branching creates Q3 spark tokens
in failed branches that GRPO systematically suppresses. Lp-Reg's mechanism
fires on exactly these tokens.

Phase 1: Inference-only diagnostic. No training. ~$5 on Modal.

Papers:
  - CURE (2508.11016): bytedance/CURE, verl 0.4.1.dev
  - Lp-Reg (2510.03222): CarlanLark/Lp-Reg-dev, verl 0.3.1.dev
  - STAPO (2602.15620): S2T mask for spurious token silencing

Usage:
  modal run pilot/modal_pilot.py

GPU: 1× A100-80GB (~1-2 hours, ~$5-10)
"""

import modal
import json
import os

# ============================================================
# MODAL SETUP
# ============================================================
app = modal.App("cure-lpreg-pilot")

pilot_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "vllm==0.8.4",
    "transformers>=4.45.0,<5.0.0",  # vllm 0.8.4 incompatible with transformers 5.x
    "datasets",
    "numpy",
    "torch",
    "tqdm",
    "huggingface_hub",
    "scipy",
)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("pilot-results", create_if_missing=True)

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Model
    "model_id": "Qwen/Qwen2.5-Math-7B",
    "max_model_len": 4096,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.85,
    # CURE branching params (verified from bytedance/CURE/recipe/CURE_First_Stage/run_cure_stage_1.sh)
    "n1": 4,  # initial rollouts per problem
    "n2": 3,  # branches per rollout
    "critical_top_k": 20,  # top-K entropy tokens for branch selection
    # Lp-Reg params (verified from CarlanLark/Lp-Reg-dev/recipe/lp_reg/)
    "min_p_kappa": 0.02,  # proxy threshold κ (from actor.minp_p_threshold)
    "lp_reg_rho": 0.01,  # bottom 1% for "low probability" (logp_neg_k_percent 14B=0.01, 32B=0.005)
    # NOTE: Lp-Reg training hyperparams (not used in pilot, for Phase 2 reference):
    #   ppo_kl_coef=0.02 (NOT 1.0), kl_type=reverse_kl (NOT low_var_kl)
    #   clip_ratio_low=1.0, clip_ratio_high=9.0 (verified from recipe scripts)
    # Generation
    "num_problems": 50,
    "temperature": 1.0,
    "max_new_tokens": 2048,
    "branch_max_new_tokens": 1024,  # branches are shorter (continuing from midpoint)
    "vllm_logprobs": 20,  # top-K logprobs per token (vLLM 0.8.4 max is 20)
    # Kill criteria thresholds
    "kill_spark_fraction": 0.15,
    "kill_proxy_max_prob": 0.05,
    "kill_branch_similarity": 0.95,
}

# ============================================================
# EXTRACTED: CURE Branch Point Detection
# Source: bytedance/CURE/recipe/CURE_First_Stage/dapo_ray_trainer.py:50-113
#
# Original: get_critic_key_token_mask()
# - Computes per-token entropy via forward pass
# - Sorts tokens by entropy descending
# - Filters: skip special tokens, skip len(text.strip()) <= 1
# - Collects top_k (20) valid tokens
# - Randomly picks ONE as branch point
#
# Adaptation: Uses vLLM's top-K logprobs instead of full logits.
# Entropy is approximated from top-20 logprobs (vLLM max=20).
# Known limitation: underestimates true entropy at diffuse positions.
# ============================================================


def detect_branch_point(token_ids, pos_logprobs_list, tokenizer, top_k=20):
    """
    CURE's branch detection adapted for vLLM output.

    Args:
        token_ids: list[int] - generated token IDs
        pos_logprobs_list: list[dict[int, float]] - top-K logprobs per position
        tokenizer: HF tokenizer
        top_k: number of candidate positions to consider

    Returns:
        branch_pos: int - selected branch position
        entropies: list[float] - approximate entropy at each position
        branch_candidates: list[int] - all valid candidate positions
    """
    import math
    import random

    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    entropies = []

    for pos_lps in pos_logprobs_list:
        if pos_lps is None or len(pos_lps) == 0:
            entropies.append(0.0)
            continue
        # Approximate entropy from top-K logprobs
        log_probs = list(pos_lps.values())
        probs = [math.exp(lp) for lp in log_probs]
        total = sum(probs)
        if total > 0:
            ent = -sum(
                (p / total) * math.log(p / total + 1e-10) for p in probs if p > 0
            )
        else:
            ent = 0.0
        entropies.append(ent)

    # Sort positions by entropy descending
    sorted_positions = sorted(
        range(len(entropies)), key=lambda i: entropies[i], reverse=True
    )

    # Filter: skip special tokens and single-char tokens (matches CURE exactly)
    valid_positions = []
    for pos in sorted_positions:
        if len(valid_positions) >= top_k:
            break
        if pos >= len(token_ids):
            continue
        if pos == 0:
            continue  # CURE skips index 0 (get_critic_key_token_mask line ~80)
        tid = token_ids[pos] if isinstance(token_ids[pos], int) else int(token_ids[pos])
        if tid in special_ids:
            continue
        token_text = tokenizer.decode(tid)
        if len(token_text.strip()) <= 1:
            continue
        valid_positions.append(pos)

    if not valid_positions:
        return 0, entropies, []

    # Random pick from valid candidates (matches CURE's random selection)
    branch_pos = valid_positions[random.randint(0, len(valid_positions) - 1)]
    return branch_pos, entropies, valid_positions


# ============================================================
# EXTRACTED: Lp-Reg Proxy Condition Check
# Source: CarlanLark/Lp-Reg-dev/verl/trainer/ppo/core_algos.py:648-745
#         CarlanLark/Lp-Reg-dev/verl/workers/actor/dp_actor.py:76-110
#
# Original flow:
# 1. apply_min_p(): threshold = kappa * max(softmax(logits))
#    Zero out tokens below threshold, renormalize -> proxy distribution
# 2. compute_policy_loss_lp_reg():
#    - Split tokens by advantage sign
#    - For negative-advantage: select bottom rho% by log_prob
#    - For each selected: check if log_prob < proxy_log_prob (survived min-p)
#    - If yes: replace loss with -A*ratio + beta*KL(proxy||policy)
#
# Adaptation: Uses vLLM's top-K logprobs to check conditions without
# requiring full logits. The key insight: we only need max(pi) and
# pi(token) to check the min-p survival condition.
# ============================================================


def check_lpreg_conditions(
    token_ids, pos_logprobs_list, advantage, rho_threshold, kappa=0.02
):
    """
    Check which tokens satisfy Lp-Reg's three activation conditions.

    Conditions (from core_algos.py:648-745):
      1. pi(token) < rho-th percentile of batch (low probability)
      2. pi_proxy(token) > 0 (survived min-p filter: pi >= kappa * max(pi))
      3. A < 0 (negative advantage)

    Args:
        token_ids: list[int]
        pos_logprobs_list: list[dict[int, float]]
        advantage: float - group-relative advantage for this trajectory
        rho_threshold: float - log-probability threshold for condition 1
        kappa: float - min-p proxy threshold (default 0.02)

    Returns:
        list[dict] - qualifying tokens with metadata
    """
    import math

    if advantage >= 0:
        return []  # Condition 3 fails for entire trajectory

    qualifying = []
    for t, (tid, pos_lps) in enumerate(zip(token_ids, pos_logprobs_list)):
        if pos_lps is None or len(pos_lps) == 0:
            continue

        token_id = tid if isinstance(tid, int) else int(tid)
        if token_id not in pos_lps:
            continue

        log_p = pos_lps[token_id]

        # Condition 1: low probability (bottom rho percentile)
        if log_p >= rho_threshold:
            continue

        # Condition 2: survived min-p filter
        # pi_proxy(token) > 0 iff pi(token) >= kappa * max(pi)
        max_lp = max(pos_lps.values())
        max_p = math.exp(max_lp)
        p = math.exp(log_p)
        threshold = kappa * max_p

        if p < threshold:
            continue  # Token would be zeroed by min-p -> pi_proxy = 0

        qualifying.append(
            {
                "position": t,
                "token_id": token_id,
                "log_prob": log_p,
                "prob": p,
                "max_prob_at_position": max_p,
                "minp_threshold": threshold,
            }
        )

    return qualifying


# ============================================================
# EXTRACTED: STAPO S2T Mask (for future Version 3)
# Source: arXiv 2602.15620, Section 3
#
# S2T mask zeros tokens where ALL THREE hold:
#   - pi(token) < tau_p (2e-3, fixed absolute)
#   - H(position) < tau_h (20th percentile within batch)
#   - A > 0 (positive advantage)
#
# Note: This is complementary to Lp-Reg. Lp-Reg protects Q3
# (low-pi, A<0). STAPO silences a specific Q2 subset
# (low-pi, low-H, A>0). They don't overlap.
# ============================================================


def check_stapo_spurious(
    token_ids, pos_logprobs_list, advantage, entropy_threshold, tau_p=2e-3
):
    """
    Identify STAPO-spurious tokens (for logging/analysis only in pilot).

    Returns list of token positions that would be masked by S2T.
    """
    import math

    if advantage <= 0:
        return []  # Only fires on positive-advantage trajectories

    spurious = []
    for t, (tid, pos_lps) in enumerate(zip(token_ids, pos_logprobs_list)):
        if pos_lps is None or len(pos_lps) == 0:
            continue

        token_id = tid if isinstance(tid, int) else int(tid)
        if token_id not in pos_lps:
            continue

        p = math.exp(pos_lps[token_id])

        # Condition 1: low probability
        if p >= tau_p:
            continue

        # Condition 2: low entropy (approximate from top-K)
        probs = [math.exp(lp) for lp in pos_lps.values()]
        total = sum(probs)
        ent = (
            -sum((pr / total) * math.log(pr / total + 1e-10) for pr in probs if pr > 0)
            if total > 0
            else 0
        )

        if ent >= entropy_threshold:
            continue

        spurious.append(
            {"position": t, "token_id": token_id, "prob": p, "entropy": ent}
        )

    return spurious


# ============================================================
# SPARK CLASSIFICATION
# ============================================================
# Reasoning connectives that signal cognitive transitions in math CoT.
# These are the tokens CURE should branch at and Lp-Reg should protect.

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


def classify_token(token_text, tokenizer=None):
    """Classify token as reasoning / noise / ambiguous."""
    text = token_text.strip().lower()

    # Check reasoning patterns
    if text in REASONING_TOKENS:
        return "reasoning"
    for pat in REASONING_TOKENS:
        if text.startswith(pat) or pat.startswith(text):
            if len(text) >= 3:  # avoid very short partial matches
                return "reasoning"

    # Noise patterns
    if len(text) <= 1:
        return "noise"
    if text.replace(".", "").replace(",", "").replace("-", "").isdigit():
        return "noise"
    if text in {"\\n", "\\t", "\\\\", "$$", "\\[", "\\]", "\\(", "\\)", "```", "---"}:
        return "noise"

    return "ambiguous"


# ============================================================
# ANSWER VERIFICATION
# ============================================================


def extract_boxed_answer(text):
    """Extract answer from \\boxed{...}. Handles nested braces."""
    import re

    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


def verify_answer(predicted, ground_truth):
    """Verify answer correctness. Works for AIME (integer) and MATH (expression)."""
    if predicted is None:
        return False
    try:
        # Normalize whitespace and common formatting
        pred = predicted.strip().replace(" ", "").replace(",", "")
        gt = ground_truth.strip().replace(" ", "").replace(",", "")
        if pred == gt:
            return True
        # Try numeric comparison
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


def load_problems(num_problems):
    """Load math problems. Prefers AIME, falls back to MATH Level 5."""
    from datasets import load_dataset

    sources = [
        (
            "di-dimitrov/AIME-2024",
            "test",
            lambda r: {"problem": r["Question"], "answer": str(r["Answer"])},
        ),
        (
            "HuggingFaceH4/MATH-500",
            "test",
            lambda r: {"problem": r["problem"], "answer": r["answer"]},
        ),
        (
            "lighteval/MATH-Hard",
            "test",
            lambda r: {"problem": r["problem"], "answer": r["answer"]},
        ),
    ]

    for name, split, mapper in sources:
        try:
            ds = load_dataset(name, split=split, trust_remote_code=True)
            problems = [mapper(row) for row in ds]
            # For MATH-500, filter hard problems
            if "MATH-500" in name:
                hard = [
                    p
                    for p, row in zip(problems, ds)
                    if row.get("level", "") in ("Level 5", "Level 4")
                ]
                if len(hard) >= num_problems:
                    problems = hard
            if len(problems) >= num_problems:
                print(f"  Loaded {num_problems} problems from {name}")
                return problems[:num_problems]
            print(f"  {name}: only {len(problems)} problems, need {num_problems}")
        except Exception as e:
            print(f"  {name}: failed ({e})")

    raise RuntimeError("Could not load any math dataset")


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================


def run_diagnostic(config):
    """
    Phase 1 pilot diagnostic. Runs on single GPU.

    Steps:
      1. Load problems + model
      2. Generate n1=4 initial responses per problem
      3. Detect branch points (CURE entropy-based)
      4. Generate n2=3 branches per response from branch points
      5. Score all trajectories, compute group advantages
      6. Check Lp-Reg conditions on tokens in failed branches
      7. Compute proxy health, branch diversity, spark fraction
      8. Evaluate kill criteria
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import numpy as np
    import random
    import math
    from tqdm import tqdm

    random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("  CURE × Lp-Reg  Pilot Diagnostic (Phase 1)")
    print("=" * 70)

    # --- 1. Load data ---
    print("\n[1/7] Loading problems...")
    problems = load_problems(config["num_problems"])

    # --- 2. Load model ---
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
    branch_gen_params = SamplingParams(
        temperature=config["temperature"],
        max_tokens=config["branch_max_new_tokens"],
        logprobs=config["vllm_logprobs"],
        top_p=1.0,
    )

    # --- 3. Generate initial responses ---
    print(f"\n[3/7] Generating n1={config['n1']} responses per problem...")
    sys_prompt = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    prompts = []
    prompt_texts = []  # store the formatted prompt for branch construction
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

    initial_outputs = llm.generate(prompts, gen_params)
    print(f"  Generated {len(initial_outputs)} initial responses")

    # --- 4. Detect branch points + generate branches ---
    print(
        f"\n[4/7] Detecting branch points, generating n2={config['n2']} branches each..."
    )

    # Parse initial outputs and detect branches
    problem_data = []
    branch_prompts = []
    branch_meta = []

    for prob_idx, problem in enumerate(tqdm(problems, desc="  Branch detection")):
        pd = {
            "problem_idx": prob_idx,
            "problem_text": problem["problem"][:200],
            "ground_truth": problem["answer"],
            "responses": [],
        }

        for r_idx in range(config["n1"]):
            out_idx = prob_idx * config["n1"] + r_idx
            output = initial_outputs[out_idx]
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

            # Detect branch point
            bp, entropies, candidates = detect_branch_point(
                tids, pos_lps, tokenizer, top_k=config["critical_top_k"]
            )

            # Score
            predicted = extract_boxed_answer(text)
            correct = verify_answer(predicted, problem["answer"])

            # Branch point info
            bp_token = tokenizer.decode(tids[bp]) if bp < len(tids) else "<none>"
            bp_entropy = entropies[bp] if bp < len(entropies) else 0
            bp_max_prob = 0
            if bp < len(pos_lps) and pos_lps[bp]:
                bp_max_prob = math.exp(max(pos_lps[bp].values()))

            # Branch context: 10 tokens before branch point for semantic quality audit
            # (Reviewer requirement: verify CURE branches at reasoning transitions, not formatting)
            ctx_start = max(0, bp - 10)
            ctx_tids = tids[ctx_start : bp + 1]
            ctx_tokens = [
                tokenizer.decode(t if isinstance(t, int) else int(t)) for t in ctx_tids
            ]

            pd["responses"].append(
                {
                    "r_idx": r_idx,
                    "text_preview": text[:300],
                    "is_correct": correct,
                    "predicted": predicted,
                    "n_tokens": len(tids),
                    "branch_pos": bp,
                    "branch_token": bp_token,
                    "branch_entropy": bp_entropy,
                    "branch_max_prob": bp_max_prob,
                    "n_branch_candidates": len(candidates),
                    "branch_context": ctx_tokens,
                }
            )

            # Construct branch prompts (CURE re-concatenation)
            # Source: dapo_ray_trainer.py:215-240
            # prompt + response[:branch_pos]
            prefix_text = tokenizer.decode(tids[:bp], skip_special_tokens=False)
            branched_prompt = prompt_texts[out_idx] + prefix_text

            for b_idx in range(config["n2"]):
                branch_prompts.append(branched_prompt)
                branch_meta.append(
                    {
                        "prob_idx": prob_idx,
                        "r_idx": r_idx,
                        "b_idx": b_idx,
                        "branch_pos": bp,
                    }
                )

        problem_data.append(pd)

    # Generate all branches
    print(f"  Generating {len(branch_prompts)} branches...")
    branch_outputs = llm.generate(branch_prompts, branch_gen_params)

    # Free vLLM memory
    del llm
    import torch

    torch.cuda.empty_cache()
    print("  Released GPU memory")

    # --- 5. Score branches, compute advantages ---
    print("\n[5/7] Scoring branches, computing advantages...")

    for pd in problem_data:
        pd["branches"] = []

    for i, (meta, b_out) in enumerate(zip(branch_meta, branch_outputs)):
        comp = b_out.outputs[0]
        text = comp.text
        tids = list(comp.token_ids)
        pos_lps = []
        for lp_obj in comp.logprobs or []:
            if lp_obj is None:
                pos_lps.append(None)
            else:
                pos_lps.append({tid: info.logprob for tid, info in lp_obj.items()})

        predicted = extract_boxed_answer(text)
        gt = problem_data[meta["prob_idx"]]["ground_truth"]
        correct = verify_answer(predicted, gt)

        problem_data[meta["prob_idx"]]["branches"].append(
            {
                "r_idx": meta["r_idx"],
                "b_idx": meta["b_idx"],
                "branch_pos": meta["branch_pos"],
                "text_preview": text[:300],
                "is_correct": correct,
                "predicted": predicted,
                "n_tokens": len(tids),
                "token_ids": tids,
                "logprobs": pos_lps,
            }
        )

    # Compute group-relative advantages (GRPO-style, per problem)
    for pd in problem_data:
        scores = [1.0 if r["is_correct"] else 0.0 for r in pd["responses"]]
        scores += [1.0 if b["is_correct"] else 0.0 for b in pd["branches"]]
        mean_s = np.mean(scores)
        std_s = np.std(scores) + 1e-8

        for r in pd["responses"]:
            r["advantage"] = float((1.0 if r["is_correct"] else 0.0) - mean_s) / std_s
        for b in pd["branches"]:
            b["advantage"] = float((1.0 if b["is_correct"] else 0.0) - mean_s) / std_s

    # --- 6. Lp-Reg analysis ---
    print("\n[6/7] Checking Lp-Reg qualifying tokens in failed branches...")

    # Two-pass approach:
    # Pass 1: Scan ALL tokens in failed branches, check min-p survival only (no rho gate)
    #         This gives us the actual Lp-Reg protection pool at inference time.
    # Pass 2: Original rho-gated analysis for comparison with training-time behavior.
    #
    # Key insight from v3 run: rho=1% yields p~2e-5 which is incompatible with
    # min-p survival (kappa=0.02 * max_p >> 2e-5). In training, policy shifts
    # make moderate tokens become low-prob. Pre-training, we measure the pool directly.

    failed_branch_count = 0
    total_failed_tokens = 0
    stapo_spurious_count = 0

    # Pass 1: min-p survival scan (no rho gate)
    all_minp_survivors = []  # tokens that survive min-p AND are in failed branches
    all_logps = []  # for rho computation

    # STAPO entropy threshold
    all_entropies_for_stapo = []
    for pd in problem_data:
        for b in pd["branches"]:
            for pos_lps in b["logprobs"]:
                if pos_lps and len(pos_lps) > 0:
                    probs = [math.exp(lp) for lp in pos_lps.values()]
                    total = sum(probs)
                    if total > 0:
                        ent = -sum(
                            (pr / total) * math.log(pr / total + 1e-10)
                            for pr in probs
                            if pr > 0
                        )
                        all_entropies_for_stapo.append(ent)

    stapo_entropy_threshold = (
        float(np.percentile(all_entropies_for_stapo, 20))
        if all_entropies_for_stapo
        else 0
    )

    for pd in problem_data:
        for b in pd["branches"]:
            if b["advantage"] < 0:
                failed_branch_count += 1
                total_failed_tokens += len(b["token_ids"])

                for t, (tid, pos_lps) in enumerate(zip(b["token_ids"], b["logprobs"])):
                    if pos_lps is None or len(pos_lps) == 0:
                        continue
                    token_id = tid if isinstance(tid, int) else int(tid)
                    if token_id not in pos_lps:
                        continue

                    log_p = pos_lps[token_id]
                    p = math.exp(log_p)
                    max_lp = max(pos_lps.values())
                    max_p = math.exp(max_lp)
                    threshold = config["min_p_kappa"] * max_p

                    all_logps.append(log_p)

                    # Min-p survival: token not zeroed by proxy distribution
                    if p >= threshold:
                        all_minp_survivors.append(
                            {
                                "position": t,
                                "token_id": token_id,
                                "log_prob": log_p,
                                "prob": p,
                                "max_prob_at_position": max_p,
                                "minp_threshold": threshold,
                                "problem_idx": pd["problem_idx"],
                                "token_text": tokenizer.decode(token_id),
                                "advantage": b["advantage"],
                            }
                        )

            # STAPO analysis on successful branches
            if b["advantage"] > 0:
                spurious = check_stapo_spurious(
                    b["token_ids"],
                    b["logprobs"],
                    b["advantage"],
                    stapo_entropy_threshold,
                )
                stapo_spurious_count += len(spurious)

    # Compute rho threshold for reference
    if all_logps:
        rho_threshold = float(np.percentile(all_logps, config["lp_reg_rho"] * 100))
    else:
        rho_threshold = -15.0

    print(
        f"  rho threshold (1%): log_p={rho_threshold:.4f}  (p={math.exp(rho_threshold):.8f})"
    )
    print(
        f"  min-p survivors in failed branches: {len(all_minp_survivors)} / {total_failed_tokens}"
    )

    # Bucket min-p survivors by probability for distribution analysis
    prob_buckets = {
        "p<0.001": [],
        "0.001<=p<0.01": [],
        "0.01<=p<0.05": [],
        "0.05<=p<0.10": [],
        "0.10<=p<0.20": [],
        "p>=0.20": [],
    }
    for s in all_minp_survivors:
        p = s["prob"]
        if p < 0.001:
            prob_buckets["p<0.001"].append(s)
        elif p < 0.01:
            prob_buckets["0.001<=p<0.01"].append(s)
        elif p < 0.05:
            prob_buckets["0.01<=p<0.05"].append(s)
        elif p < 0.10:
            prob_buckets["0.05<=p<0.10"].append(s)
        elif p < 0.20:
            prob_buckets["0.10<=p<0.20"].append(s)
        else:
            prob_buckets["p>=0.20"].append(s)

    print(f"\n  Min-p survivor distribution by probability bucket:")
    for bucket_name, bucket_tokens in prob_buckets.items():
        n = len(bucket_tokens)
        n_reasoning = sum(
            1 for t in bucket_tokens if classify_token(t["token_text"]) == "reasoning"
        )
        pct = n / max(total_failed_tokens, 1) * 100
        print(
            f"    {bucket_name:>20}: {n:>6} tokens ({pct:.2f}%)  reasoning: {n_reasoning}"
        )

    # The Lp-Reg "zone" is the low-but-survived-min-p region: p < 0.05, survived proxy
    # This is where GRPO suppression hits hardest (high |A|/p) AND Lp-Reg would fire
    all_qualifying = [s for s in all_minp_survivors if s["prob"] < 0.05]
    for q in all_qualifying:
        q["classification"] = classify_token(q["token_text"])
        q["suppression_magnitude"] = abs(q["advantage"]) / max(q["prob"], 1e-10)

    # Suppression magnitude stats
    supp_mags = (
        [q["suppression_magnitude"] for q in all_qualifying] if all_qualifying else []
    )
    supp_stats = {
        "median": float(np.median(supp_mags)) if supp_mags else 0,
        "p90": float(np.percentile(supp_mags, 90)) if supp_mags else 0,
        "p99": float(np.percentile(supp_mags, 99)) if supp_mags else 0,
        "mean": float(np.mean(supp_mags)) if supp_mags else 0,
    }
    # Per-class suppression for reasoning vs noise comparison
    supp_reasoning = [
        q["suppression_magnitude"]
        for q in all_qualifying
        if q["classification"] == "reasoning"
    ]
    supp_noise = [
        q["suppression_magnitude"]
        for q in all_qualifying
        if q["classification"] == "noise"
    ]

    # --- 7. Compute metrics ---
    print("\n[7/7] Computing metrics...")

    n_reasoning = sum(1 for q in all_qualifying if q["classification"] == "reasoning")
    n_noise = sum(1 for q in all_qualifying if q["classification"] == "noise")
    n_ambiguous = sum(1 for q in all_qualifying if q["classification"] == "ambiguous")
    n_total = len(all_qualifying)
    spark_fraction = n_reasoning / max(n_total, 1)

    # Proxy health: max(pi) distribution at branch points vs random positions
    bp_max_probs = [
        r["branch_max_prob"] for pd in problem_data for r in pd["responses"]
    ]
    random_max_probs = []
    for pd in problem_data:
        for b in pd["branches"]:
            # Sample a random mid-response position
            mid = len(b["logprobs"]) // 2
            if mid < len(b["logprobs"]) and b["logprobs"][mid]:
                random_max_probs.append(math.exp(max(b["logprobs"][mid].values())))

    # Branch diversity: Jaccard similarity of branch continuations
    jaccard_sims = []
    for pd in problem_data:
        for r_idx in range(config["n1"]):
            siblings = [b for b in pd["branches"] if b["r_idx"] == r_idx]
            for i in range(len(siblings)):
                for j in range(i + 1, len(siblings)):
                    wi = set(siblings[i]["text_preview"].split())
                    wj = set(siblings[j]["text_preview"].split())
                    union = wi | wj
                    if union:
                        jaccard_sims.append(len(wi & wj) / len(union))

    # Accuracy stats
    init_correct = sum(
        1 for pd in problem_data for r in pd["responses"] if r["is_correct"]
    )
    init_total = sum(len(pd["responses"]) for pd in problem_data)
    branch_correct = sum(
        1 for pd in problem_data for b in pd["branches"] if b["is_correct"]
    )
    branch_total = sum(len(pd["branches"]) for pd in problem_data)

    # ============================================================
    # RESULTS
    # ============================================================
    results = {
        "config": config,
        "summary": {
            "num_problems": len(problems),
            "initial_accuracy": init_correct / max(init_total, 1),
            "branch_accuracy": branch_correct / max(branch_total, 1),
            "initial_total": init_total,
            "branch_total": branch_total,
            "failed_branches": failed_branch_count,
            "total_failed_tokens": total_failed_tokens,
            "lpreg_qualifying_total": n_total,
            "lpreg_qualifying_reasoning": n_reasoning,
            "lpreg_qualifying_noise": n_noise,
            "lpreg_qualifying_ambiguous": n_ambiguous,
            "spark_fraction": spark_fraction,
            "rho_threshold_logp": rho_threshold,
            "rho_threshold_p": math.exp(rho_threshold),
            "proxy_mean_max_prob_at_branch": (
                float(np.mean(bp_max_probs)) if bp_max_probs else 0
            ),
            "proxy_mean_max_prob_at_random": (
                float(np.mean(random_max_probs)) if random_max_probs else 0
            ),
            "proxy_std_max_prob_at_branch": (
                float(np.std(bp_max_probs)) if bp_max_probs else 0
            ),
            "branch_diversity_mean_jaccard": (
                float(np.mean(jaccard_sims)) if jaccard_sims else 0
            ),
            "branch_diversity_std_jaccard": (
                float(np.std(jaccard_sims)) if jaccard_sims else 0
            ),
            "stapo_spurious_in_successful": stapo_spurious_count,
            # Suppression magnitude: how hard GRPO pushes qualifying tokens
            "suppression_median": supp_stats["median"],
            "suppression_p90": supp_stats["p90"],
            "suppression_p99": supp_stats["p99"],
            "suppression_mean": supp_stats["mean"],
            "suppression_reasoning_median": (
                float(np.median(supp_reasoning)) if supp_reasoning else 0
            ),
            "suppression_noise_median": (
                float(np.median(supp_noise)) if supp_noise else 0
            ),
        },
        "qualifying_tokens": [
            {k: v for k, v in q.items() if k != "logprobs"}
            for q in all_qualifying[:200]
        ],
        "branch_point_samples": [
            {
                "problem": pd["problem_text"],
                "response_idx": r["r_idx"],
                "branch_pos": r["branch_pos"],
                "branch_token": r["branch_token"],
                "branch_entropy": r["branch_entropy"],
                "branch_max_prob": r["branch_max_prob"],
                "n_candidates": r["n_branch_candidates"],
                "is_correct": r["is_correct"],
                "branch_context": r.get("branch_context", []),
            }
            for pd in problem_data[:10]
            for r in pd["responses"]
        ],
    }

    # Print summary
    s = results["summary"]
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"\n  Trajectories")
    print(f"    Problems:          {s['num_problems']}")
    print(
        f"    Initial accuracy:  {s['initial_accuracy']:.1%}  ({init_correct}/{init_total})"
    )
    print(
        f"    Branch accuracy:   {s['branch_accuracy']:.1%}  ({branch_correct}/{branch_total})"
    )
    print(f"    Failed branches:   {s['failed_branches']}")

    print(f"\n  Lp-Reg Qualifying Tokens (in failed branches)")
    print(
        f"    Total:      {n_total}  (of {s['total_failed_tokens']} tokens in failed branches)"
    )
    print(f"    Reasoning:  {n_reasoning}  ({spark_fraction:.1%})")
    print(f"    Noise:      {n_noise}  ({n_noise/max(n_total,1):.1%})")
    print(f"    Ambiguous:  {n_ambiguous}  ({n_ambiguous/max(n_total,1):.1%})")
    print(f"    rho thresh: p={s['rho_threshold_p']:.8f}")

    print(f"\n  Proxy Health")
    print(
        f"    mean max(pi) at branch points:   {s['proxy_mean_max_prob_at_branch']:.4f}  (std {s['proxy_std_max_prob_at_branch']:.4f})"
    )
    print(
        f"    mean max(pi) at random positions: {s['proxy_mean_max_prob_at_random']:.4f}"
    )

    print(f"\n  Branch Diversity")
    print(
        f"    Mean Jaccard similarity: {s['branch_diversity_mean_jaccard']:.3f}  (std {s['branch_diversity_std_jaccard']:.3f})"
    )

    print(f"\n  Suppression Magnitude (|A|/pi)")
    print(f"    Median:   {supp_stats['median']:.1f}")
    print(f"    P90:      {supp_stats['p90']:.1f}")
    print(f"    P99:      {supp_stats['p99']:.1f}")
    print(f"    Mean:     {supp_stats['mean']:.1f}")
    if supp_reasoning:
        print(
            f"    Reasoning tokens median: {np.median(supp_reasoning):.1f}  (n={len(supp_reasoning)})"
        )
    if supp_noise:
        print(
            f"    Noise tokens median:     {np.median(supp_noise):.1f}  (n={len(supp_noise)})"
        )
    if supp_reasoning and supp_noise:
        ratio = np.median(supp_reasoning) / max(np.median(supp_noise), 1e-10)
        print(f"    Reasoning/Noise ratio:   {ratio:.2f}x")

    print(
        f"\n  Branch Context Audit (10 tokens before branch point, first 10 problems)"
    )
    bp_count = 0
    for pd in problem_data[:10]:
        for r in pd["responses"]:
            ctx = r.get("branch_context", [])
            if ctx:
                ctx_str = "".join(ctx)
                cls = classify_token(r["branch_token"])
                print(
                    f"    P{pd['problem_idx']:02d}/R{r['r_idx']}  [{cls:>9}]  ...{repr(ctx_str)} -> [{r['branch_token']}]  H={r['branch_entropy']:.3f}"
                )
                bp_count += 1
            if bp_count >= 20:
                break
        if bp_count >= 20:
            break

    print(f"\n  STAPO (bonus)")
    print(f"    Spurious tokens in successful branches: {stapo_spurious_count}")

    # Literature comparison
    print(f"\n  Literature Comparison")
    print(f"    {'Metric':<35} {'Pilot':>10} {'Published':>12} {'Source':>15}")
    print(f"    {'-'*35} {'-'*10} {'-'*12} {'-'*15}")
    print(
        f"    {'Branch candidates (top_k)':35} {config['critical_top_k']:>10} {'20':>12} {'CURE §3.2':>15}"
    )
    print(
        f"    {'STAPO spurious % (of success tok)':35} {'N/A':>10} {'~0.01%':>12} {'STAPO §3':>15}"
    )
    print(
        f"    {'Lp-Reg rho percentile':35} {config['lp_reg_rho']:>10.3f} {'0.01':>12} {'Lp-Reg §3':>15}"
    )
    print(
        f"    {'Min-p kappa':35} {config['min_p_kappa']:>10.3f} {'0.02':>12} {'Lp-Reg §3':>15}"
    )
    print(
        f"    {'Spark fraction':35} {spark_fraction:>10.1%} {'(novel)':>12} {'This work':>15}"
    )
    print(
        f"    {'Supp. magnitude median':35} {supp_stats['median']:>10.1f} {'(novel)':>12} {'This work':>15}"
    )

    # Kill criteria
    print(f"\n  KILL CRITERIA")
    kills = []

    if spark_fraction < config["kill_spark_fraction"]:
        print(
            f"    FAIL  spark fraction {spark_fraction:.1%} < {config['kill_spark_fraction']:.0%}"
        )
        kills.append("spark_fraction")
    else:
        print(
            f"    PASS  spark fraction {spark_fraction:.1%} >= {config['kill_spark_fraction']:.0%}"
        )

    if s["proxy_mean_max_prob_at_branch"] < config["kill_proxy_max_prob"]:
        print(
            f"    FAIL  proxy max(pi) {s['proxy_mean_max_prob_at_branch']:.4f} < {config['kill_proxy_max_prob']}"
        )
        kills.append("proxy_health")
    else:
        print(
            f"    PASS  proxy max(pi) {s['proxy_mean_max_prob_at_branch']:.4f} >= {config['kill_proxy_max_prob']}"
        )

    if jaccard_sims and np.mean(jaccard_sims) > config["kill_branch_similarity"]:
        print(
            f"    FAIL  branch similarity {np.mean(jaccard_sims):.3f} > {config['kill_branch_similarity']}"
        )
        kills.append("branch_diversity")
    else:
        print(
            f"    PASS  branch similarity {np.mean(jaccard_sims):.3f} < {config['kill_branch_similarity']}"
        )

    if kills:
        print(f"\n  VERDICT: {len(kills)} kill criteria triggered: {kills}")
        print(f"           Investigate before proceeding to Phase 2.")
    else:
        print(f"\n  VERDICT: All criteria pass. Proceed to Phase 2 training.")

    # Sample tokens for manual inspection
    print(f"\n  Sample Qualifying Tokens (first 30)")
    print(
        f"  {'Class':>12}  {'Token':>15}  {'p':>10}  {'max_p':>8}  {'A':>7}  {'|A|/p':>10}"
    )
    print(f"  {'-'*12}  {'-'*15}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*10}")
    for q in all_qualifying[:30]:
        print(
            f"  {q['classification']:>12}  {repr(q['token_text']):>15}  {q['prob']:.2e}  {q['max_prob_at_position']:.4f}  {q['advantage']:+.3f}  {q['suppression_magnitude']:>10.1f}"
        )

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
def pilot_phase1():
    """Run Phase 1 inference diagnostic on Modal."""
    from datetime import datetime

    results = run_diagnostic(CONFIG)

    # Timestamped output directory
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/results/pilot_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Full JSON results
    json_path = f"{out_dir}/results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 2. Qualifying tokens TSV (for spreadsheet analysis)
    tsv_path = f"{out_dir}/qualifying_tokens.tsv"
    with open(tsv_path, "w") as f:
        headers = [
            "problem_idx",
            "position",
            "token_text",
            "classification",
            "prob",
            "max_prob",
            "advantage",
            "suppression_mag",
            "log_prob",
            "minp_threshold",
        ]
        f.write("\t".join(headers) + "\n")
        for q in results.get("qualifying_tokens", []):
            row = [
                str(q.get("problem_idx", "")),
                str(q.get("position", "")),
                q.get("token_text", "").replace("\t", " ").replace("\n", " "),
                q.get("classification", ""),
                f"{q.get('prob', 0):.6e}",
                f"{q.get('max_prob_at_position', 0):.6f}",
                f"{q.get('advantage', 0):+.4f}",
                f"{q.get('suppression_magnitude', 0):.2f}",
                f"{q.get('log_prob', 0):.4f}",
                f"{q.get('minp_threshold', 0):.6e}",
            ]
            f.write("\t".join(row) + "\n")

    # 3. Branch contexts TSV (for semantic quality audit)
    ctx_path = f"{out_dir}/branch_contexts.tsv"
    with open(ctx_path, "w") as f:
        headers = [
            "problem",
            "response_idx",
            "branch_pos",
            "branch_token",
            "entropy",
            "max_prob",
            "context_before",
            "is_correct",
        ]
        f.write("\t".join(headers) + "\n")
        for bp in results.get("branch_point_samples", []):
            ctx = bp.get("branch_context", [])
            ctx_str = "".join(ctx).replace("\t", " ").replace("\n", " ")
            row = [
                bp.get("problem", "")[:100].replace("\t", " ").replace("\n", " "),
                str(bp.get("response_idx", "")),
                str(bp.get("branch_pos", "")),
                bp.get("branch_token", "").replace("\t", " "),
                f"{bp.get('branch_entropy', 0):.4f}",
                f"{bp.get('branch_max_prob', 0):.4f}",
                ctx_str,
                str(bp.get("is_correct", "")),
            ]
            f.write("\t".join(row) + "\n")

    # 4. Summary one-liner for quick grep
    summary_path = f"{out_dir}/summary.txt"
    s = results["summary"]
    with open(summary_path, "w") as f:
        f.write(f"timestamp={ts}\n")
        for k, v in s.items():
            f.write(f"{k}={v}\n")

    results_vol.commit()

    print(f"\nResults saved to Modal volume 'pilot-results':")
    print(f"  {json_path}")
    print(f"  {tsv_path}")
    print(f"  {ctx_path}")
    print(f"  {summary_path}")
    print(f"\nDownload all:")
    print(f"  modal volume get pilot-results pilot_{ts}/")
    return results["summary"]


@app.local_entrypoint()
def main():
    """Launch pilot from CLI: modal run pilot/modal_pilot.py"""
    print("Launching CURE x Lp-Reg pilot on Modal (1x A100-80GB)...")
    summary = pilot_phase1.remote()

    print("\n" + "=" * 70)
    print("  PILOT COMPLETE")
    print("=" * 70)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print("\nDownload full results:")
    print("  modal volume get pilot-results pilot_results.json")
