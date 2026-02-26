"""
Spark-MCTS Backward Pass Smoke Test
=====================================
Validates the Spark-MCTS training pipeline before spending full compute (~$30).

Four conditions × 5 steps each on GSM8K (dataset-independent for the backward
pass; MATH Level 3-4 switch happens at the full training run stage):
  A: Vanilla GRPO (baseline)
  B: CURE (entropy-only branching, vanilla loss)
  C: Spark-MCTS (3-gate branching, Lp-Reg loss, prefix stop-gradient)
  D: Gradient routing (flat GRPO + entropy-based gradient mask, NLL protection)

Verifies:
  1. Loss computes without NaN/Inf across all 3 conditions
  2. Lp-Reg KL protection fires with non-zero magnitude on 3-gate spark tokens
     in failed branches (condition C)
  3. Prefix gradient is zero — response_mask correctly zeros prefix positions
     for branched responses (condition C)

Usage:
  modal run pilot/modal_spark_smoke.py
"""

import modal
import json
import os

# ============================================================
# MODAL SETUP
# ============================================================

app = modal.App("spark-mcts-smoke")

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("spark-pilot-results", create_if_missing=True)

# ============================================================
# PATCH CODE — copied exactly from modal_ablation.py
# ============================================================

CORE_ALGOS_APPEND = r'''

# === Advantage-Weighted NLL spark protection ===
# Replaces Lp-Reg KL (which has near-zero gradient on rare tokens: beta*(pi_theta - pi_ref) ≈ 0)
# with NLL whose gradient is -alpha*|A|*(1-pi_theta) ≈ alpha*|A| for small pi_theta.
# Rarity-invariant: a spark at pi=0.01 gets the same protective force as pi=0.1.

def compute_policy_loss_spark_nll(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode="token-mean",
    logp_neg_k_percent=0.01,
    nll_alpha=0.3,
):
    """Policy loss with Advantage-Weighted NLL protection on spark tokens in failed branches."""
    import torch

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)

    with torch.no_grad():
        ppo_kl_mean = verl_F.masked_mean(
            kl_penalty(log_prob, old_log_prob, "low_var_kl"), response_mask
        )

    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    ratio_clipped = ratio.clamp(1 - cliprange_low, 1 + cliprange_high)
    pg_losses2 = -advantages * ratio_clipped
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    with torch.no_grad():
        pg_clipfrac = verl_F.masked_mean(
            (pg_losses2 > pg_losses1).float(), response_mask
        )

    # Find valid indices
    all_valid_mask = response_mask > 0
    all_valid_flat_idx = all_valid_mask.reshape(-1).nonzero(as_tuple=True)[0]

    if len(all_valid_flat_idx) == 0:
        pg_loss = agg_loss(
            loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
        )
        return pg_loss, pg_clipfrac, ppo_kl_mean, torch.tensor(0.0)

    _, seq_len = advantages.shape
    advantages_flat = advantages.reshape(-1)
    log_prob_flat = log_prob.reshape(-1)

    neg_indices = all_valid_flat_idx[advantages_flat[all_valid_flat_idx] < 0]

    # Negative-advantage tokens: apply Advantage-Weighted NLL protection
    # on the lowest-probability tokens (the most vulnerable sparks).
    # NLL gradient: -alpha*|A|*(1-pi_theta) ≈ alpha*|A| for rare tokens.
    nll_protect_mean_val = 0.0
    n_protected = 0
    if len(neg_indices) > 0 and logp_neg_k_percent > 0:
        neg_logp_k = max(1, int(len(neg_indices) * logp_neg_k_percent))
        if neg_logp_k > 1:
            with torch.no_grad():
                neg_logp_values = log_prob_flat[neg_indices]
                _, neg_logp_indices = neg_logp_values.topk(k=neg_logp_k, largest=False)
                neg_target_idx = neg_indices[neg_logp_indices]

            batch_idx = neg_target_idx // seq_len
            seq_idx = neg_target_idx % seq_len

            if len(batch_idx) > 0:
                # Advantage-Weighted NLL: -alpha * |A| * log(pi_theta)
                # log_prob < 0, so -log_prob > 0 → protection term is positive.
                # Gradient pushes pi_theta UP, counteracting GRPO suppression.
                nll_protect = -nll_alpha * torch.abs(advantages[batch_idx, seq_idx]) * log_prob[batch_idx, seq_idx]
                pg_losses[batch_idx, seq_idx] = (
                    -advantages[batch_idx, seq_idx] * ratio[batch_idx, seq_idx]
                    + nll_protect
                )
                with torch.no_grad():
                    nll_protect_mean_val = nll_protect.mean().item()
                    n_protected = len(batch_idx)

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    # Pack protection stats into a tensor that flows through actor/pg_clipfrac_lower
    # Format: nll_protect_mean (readable by trainer diagnostics)
    nll_protect_tensor = torch.tensor(nll_protect_mean_val)

    return pg_loss, pg_clipfrac, ppo_kl_mean, nll_protect_tensor
'''

# Patch 1: Add min_p parameter to _forward_micro_batch
# We modify the non-remove_padding path (else branch) to support min-p filtering
DP_ACTOR_FORWARD_OLD = """\
                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)"""

DP_ACTOR_FORWARD_NEW = """\
                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                    # Min-p filtering for Lp-Reg target distribution
                    if min_p is not None and min_p > 0:
                        import torch.nn.functional as F
                        probs = F.softmax(logits.detach(), dim=-1)
                        max_probs = probs.max(dim=-1, keepdim=True).values
                        threshold = min_p * max_probs
                        invalid_mask = probs < threshold
                        logits_filtered = logits.clone()
                        logits_filtered[invalid_mask] = -1e4
                        log_probs = logprobs_from_logits(logits_filtered, micro_batch["responses"])
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])"""

# Patch 2: Add min_p to _forward_micro_batch signature
DP_ACTOR_SIG_OLD = (
    "    def _forward_micro_batch(\n"
    "        self, micro_batch, temperature, calculate_entropy=False\n"
    "    ) -> Tuple[torch.Tensor, torch.Tensor]:"
)
DP_ACTOR_SIG_NEW = (
    "    def _forward_micro_batch(\n"
    "        self, micro_batch, temperature, calculate_entropy=False, min_p=None\n"
    "    ) -> Tuple[torch.Tensor, torch.Tensor]:"
)

# Patch 3: Add Lp-Reg branch in update_policy
# Insert between "vanilla" branch and "else" (registry) branch
DP_ACTOR_LOSS_OLD = """\
                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                    else:"""

DP_ACTOR_LOSS_NEW = """\
                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                    elif self.config.policy_loss.loss_mode == "spark_nll":
                        # Advantage-Weighted NLL: rarity-invariant spark protection
                        from verl.trainer.ppo.core_algos import compute_policy_loss_spark_nll
                        pg_loss, pg_clipfrac, ppo_kl, nll_protect_mag = compute_policy_loss_spark_nll(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            loss_agg_mode=loss_agg_mode,
                            logp_neg_k_percent=self.config.policy_loss.get("logp_neg_k_percent", 0.01),
                            nll_alpha=self.config.policy_loss.get("nll_alpha", 0.3),
                        )
                        # Flow NLL protection magnitude through pg_clipfrac_lower → actor/pg_clipfrac_lower
                        pg_clipfrac_lower = nll_protect_mag
                    else:"""

# Patch 4: Min-p filtering in the remove_padding path of _forward_micro_batch
DP_ACTOR_RMPAD_OLD = """\
                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )"""

DP_ACTOR_RMPAD_NEW = """\
                    # Min-p filtering for Lp-Reg (remove_padding path)
                    if min_p is not None and min_p > 0:
                        import torch.nn.functional as F
                        probs = F.softmax(logits_rmpad.detach(), dim=-1)
                        max_probs = probs.max(dim=-1, keepdim=True).values
                        threshold = min_p * max_probs
                        invalid_mask = probs < threshold
                        logits_filtered = logits_rmpad.clone()
                        logits_filtered[invalid_mask] = -1e4
                        log_probs = logprobs_from_logits(
                            logits=logits_filtered,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=False,
                        )
                    else:
                        inplace_backward = True
                        if calculate_entropy:
                            inplace_backward = False
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )"""

# Patch 5: Import for compute_policy_loss_spark_nll in dp_actor.py
DP_ACTOR_IMPORT_OLD = (
    "from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, "
    "get_policy_loss_fn, kl_penalty"
)
DP_ACTOR_IMPORT_NEW = (
    "from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, "
    "get_policy_loss_fn, kl_penalty, compute_policy_loss_spark_nll"
)

# Patch 6: Make flash_attn import safe (fallback if not installed)
DP_ACTOR_FLASH_OLD = """\
if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input"""

DP_ACTOR_FLASH_NEW = """\
if is_cuda_available:
    try:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
    except ImportError:
        import warnings
        warnings.warn("flash_attn not installed; use_remove_padding=True will fail")"""

# ============================================================
# DATA PREP — GSM8K (same as ablation; dataset-independent for backward pass)
# MATH Level 3-4 swap happens at the full training run stage.
# ============================================================

DATA_PREP_SCRIPT = r'''
"""Prepare GSM8K dataset in verl-compatible parquet format."""
import os
import re
import datasets

def extract_solution(solution_str):
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0).split("#### ")[1].replace(",", "")
    return final_solution

def prepare_gsm8k(output_dir="/root/data/gsm8k"):
    os.makedirs(output_dir, exist_ok=True)
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction = 'Let\'s think step by step and output the final answer after "####".'

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw + " " + instruction
            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            return {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split, "index": idx,
                    "answer": answer_raw, "question": question_raw,
                },
            }
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(output_dir, "test.parquet"))
    print(f"GSM8K prepared: {len(train_dataset)} train, {len(test_dataset)} test")
    return output_dir

if __name__ == "__main__":
    prepare_gsm8k()
'''

# ============================================================
# DATA PREP — MATH Level 3-4
# ============================================================

MATH_DATA_PREP_SCRIPT = r'''
"""Prepare MATH Level 3-4 dataset in verl-compatible parquet format."""
import os
import re
import random
import datasets

def extract_boxed_answer(solution_str):
    """Extract answer from \\boxed{...} — handles nested braces."""
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution_str)
    return matches[-1].strip() if matches else None

def prepare_math_l34(output_dir="/root/data/math_l34"):
    os.makedirs(output_dir, exist_ok=True)

    # MATH-500 has 500 problems with level (int64), problem, solution, answer
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    level34 = [row for row in ds if row.get("level", 0) in (3, 4)]
    print(f"MATH-500: {len(level34)} problems at Level 3-4 (of {len(ds)} total)")

    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(level34)
    split_idx = int(0.8 * len(level34))
    train_data = level34[:split_idx]
    val_data = level34[split_idx:]
    if len(val_data) % 2 != 0:
        val_data = val_data[:-1]  # ensure even count for multi-GPU chunking

    instruction = "Let's think step by step and put your final answer within \\boxed{}."

    def process_example(example, idx, split):
        answer = example.get("answer") or extract_boxed_answer(example.get("solution", ""))
        if not answer:
            return None
        return {
            "data_source": "HuggingFaceH4/MATH-500",
            "prompt": [{"role": "user", "content": example["problem"] + " " + instruction}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split, "index": idx,
                "level": example.get("level", 0),
                "subject": example.get("type", example.get("subject", "unknown")),
            },
        }

    train_processed = [process_example(r, i, "train") for i, r in enumerate(train_data)]
    train_processed = [r for r in train_processed if r is not None]
    val_processed = [process_example(r, i, "test") for i, r in enumerate(val_data)]
    val_processed = [r for r in val_processed if r is not None]

    from datasets import Dataset
    train_ds = Dataset.from_dict({k: [r[k] for r in train_processed] for k in train_processed[0].keys()})
    val_ds = Dataset.from_dict({k: [r[k] for r in val_processed] for k in val_processed[0].keys()})

    train_ds.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_ds.to_parquet(os.path.join(output_dir, "test.parquet"))
    print(f"MATH L3-4 prepared: {len(train_ds)} train, {len(val_ds)} val")
    return output_dir

if __name__ == "__main__":
    prepare_math_l34()
'''

# ============================================================
# SPARK-MCTS TRAINER
# ============================================================

SPARK_TRAINER_CODE = r'''
"""Spark-MCTS trainer with 3-gate filter and prefix stop-gradient."""
import uuid
import json
import os
import math
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.debug import marked_timer


class SparkMCTSTrainer(RayPPOTrainer):

    def compute_st_metric(self, batch):
        """
        Compute |S_t| using FIXED thresholds from pilot calibration.

        This is the ruler — it never changes during training. Measures how many
        tokens in each sequence pass the fixed 3-gate filter calibrated at step 0.
        Tracks absolute pool health over time.

        The intervention trigger uses dynamic per-sequence thresholds (in
        get_spark_branch_mask). This metric is independent of the intervention.
        """
        fixed_tau_h = self.config.trainer.get("fixed_tau_h", 2.2697)
        fixed_tau_p = self.config.trainer.get("fixed_tau_p", 0.7005)
        fixed_kappa = self.config.trainer.get("fixed_kappa", 0.02)

        log_prob_and_entropy = self.actor_rollout_wg.compute_log_prob(batch)
        entropys = log_prob_and_entropy.batch["entropys"]
        response_ids = batch.batch["responses"]
        log_probs = log_prob_and_entropy.batch["old_log_probs"]
        pi = torch.exp(log_probs)

        batch_size = response_ids.shape[0]
        special_ids = set(int(x) for x in self.tokenizer.all_special_ids)

        spark_counts = []    # |S_t| per sequence — the primary metric
        valid_counts = []
        entropy_values = []

        spark_pos_fracs = []  # position / seq_len for each spark token

        for i in range(batch_size):
            H_vals, pi_vals, positions = self._get_valid_tokens(
                response_ids[i], entropys[i], pi[i], special_ids
            )
            seq_len = response_ids[i].shape[0]
            valid_counts.append(len(H_vals))
            sparks = 0
            for k in range(len(H_vals)):
                g1 = H_vals[k] > fixed_tau_h
                g2 = 0 < pi_vals[k] < fixed_tau_p
                g3 = pi_vals[k] > fixed_kappa
                if g1 and g2 and g3:
                    sparks += 1
                    spark_pos_fracs.append(positions[k] / max(seq_len, 1))
            spark_counts.append(sparks)

            # Mean entropy for this sequence (all valid tokens)
            if H_vals:
                entropy_values.append(float(np.mean(H_vals)))

        total_sparks = sum(spark_counts)
        total_valid = sum(valid_counts)
        spark_frac = total_sparks / max(total_valid, 1)

        st_metrics = {
            # Primary metric: |S_t| per sequence (fixed ruler)
            "st/mean_candidates": float(np.mean(spark_counts)),
            "st/std_candidates": float(np.std(spark_counts)),
            "st/min_candidates": float(np.min(spark_counts)),
            "st/max_candidates": float(np.max(spark_counts)),
            # Pool fraction (fixed ruler)
            "st/spark_fraction_fixed": spark_frac,
            "st/total_sparks_fixed": total_sparks,
            "st/total_valid": total_valid,
        }
        if entropy_values:
            st_metrics["st/mean_entropy"] = float(np.mean(entropy_values))
        # Spark position diagnostic: where in the sequence do sparks fire?
        if spark_pos_fracs:
            pos_arr = np.array(spark_pos_fracs)
            st_metrics["st/spark_pos_frac_mean"] = float(np.mean(pos_arr))
            st_metrics["st/spark_pos_frac_median"] = float(np.median(pos_arr))
            st_metrics["st/spark_pos_frac_std"] = float(np.std(pos_arr))
            st_metrics["st/spark_pos_frac_q25"] = float(np.percentile(pos_arr, 25))
            st_metrics["st/spark_pos_frac_q75"] = float(np.percentile(pos_arr, 75))
            # Fraction of sparks in last 30% of sequence
            st_metrics["st/spark_pos_late_frac"] = float(np.mean(pos_arr > 0.7))

        return st_metrics

    # ================================================================
    # MASS BALANCE METRIC (BroRL Σπ_t)
    # ================================================================

    def _calibrate_mass_balance(self, n_problems=50):
        """
        Calibrate mass balance reference set before training starts.

        Generates responses for n_problems held-out problems, identifies
        3-gate sparks using fixed thresholds, and records their π_0 values.
        The (prompt, response) pairs are saved for teacher-forced re-evaluation
        at training checkpoints.
        """
        # Collect held-out prompts from val_dataloader
        collected = []
        total = 0
        for batch_dict in self.val_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            collected.append(batch)
            total += len(batch.batch["input_ids"])
            if total >= n_problems:
                break

        if not collected:
            print("WARN: no val data for mass balance calibration")
            return False

        mb_batch = DataProto.concat(collected)
        if len(mb_batch.batch["input_ids"]) > n_problems:
            mb_batch = mb_batch[:n_problems]
        actual_n = len(mb_batch.batch["input_ids"])

        # Prepare gen_batch (same format as training loop)
        if "multi_modal_data" in mb_batch.non_tensor_batch.keys():
            gen_batch = mb_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
        else:
            gen_batch = mb_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )

        # Generate 1 response per prompt
        gen_batch.meta_info["n"] = 1
        gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)

        # Compute log_probs and entropies at step 0
        lp_output = self.actor_rollout_wg.compute_log_prob(gen_output)
        entropys = lp_output.batch["entropys"]
        log_probs = lp_output.batch["old_log_probs"]
        pi = torch.exp(log_probs)
        response_ids = gen_output.batch["responses"]

        # Identify 3-gate sparks using FIXED thresholds (the ruler)
        fixed_tau_h = self.config.trainer.get("fixed_tau_h", 2.2697)
        fixed_tau_p = self.config.trainer.get("fixed_tau_p", 0.7005)
        fixed_kappa = self.config.trainer.get("fixed_kappa", 0.02)
        special_ids = set(int(x) for x in self.tokenizer.all_special_ids)

        spark_refs = []
        batch_size = response_ids.shape[0]
        for i in range(batch_size):
            H_vals, pi_vals, positions = self._get_valid_tokens(
                response_ids[i], entropys[i], pi[i], special_ids
            )
            for k, j in enumerate(positions):
                g1 = H_vals[k] > fixed_tau_h
                g2 = 0 < pi_vals[k] < fixed_tau_p
                g3 = pi_vals[k] > fixed_kappa
                if g1 and g2 and g3:
                    spark_refs.append({
                        "seq_idx": i,
                        "position": j,
                        "token_id": int(response_ids[i, j].item()),
                        "token_text": self.tokenizer.decode(response_ids[i, j].item()),
                        "pi_0": float(pi_vals[k]),
                        "entropy_0": float(H_vals[k]),
                    })

        sum_pi_0 = sum(s["pi_0"] for s in spark_refs)

        # Store for checkpoint measurement
        self._mb_gen_output = gen_output
        self._mb_spark_refs = spark_refs
        self._mb_log = [{
            "step": 0,
            "sum_pi": sum_pi_0,
            "n_sparks": len(spark_refs),
            "n_sequences": batch_size,
            "mean_pi": sum_pi_0 / max(len(spark_refs), 1),
            "ratio": 1.0,
        }]

        print(f"Mass balance: {len(spark_refs)} sparks across {actual_n} held-out sequences")
        print(f"  Σπ_0 = {sum_pi_0:.4f}, mean π_0 = {sum_pi_0 / max(len(spark_refs), 1):.6f}")
        return True

    def _measure_mass_balance(self, step):
        """
        Re-evaluate Σπ_t at spark positions using current model weights.

        Teacher-forces the step-0 responses through the current model and
        reads off π_t at each recorded spark position. No gradient effect.
        """
        if not hasattr(self, '_mb_gen_output') or not self._mb_spark_refs:
            return

        # Re-compute log_probs on the fixed (prompt, response) batch
        lp_output = self.actor_rollout_wg.compute_log_prob(self._mb_gen_output)
        log_probs = lp_output.batch["old_log_probs"]
        pi = torch.exp(log_probs)

        sum_pi_t = 0.0
        pi_deltas = []
        for ref in self._mb_spark_refs:
            i, j = ref["seq_idx"], ref["position"]
            pi_t = pi[i, j].item()
            sum_pi_t += pi_t
            pi_deltas.append(pi_t - ref["pi_0"])

        entry = {
            "step": step,
            "sum_pi": sum_pi_t,
            "n_sparks": len(self._mb_spark_refs),
            "mean_pi": sum_pi_t / max(len(self._mb_spark_refs), 1),
            "ratio": sum_pi_t / max(self._mb_log[0]["sum_pi"], 1e-10),
            "mean_delta_pi": float(np.mean(pi_deltas)) if pi_deltas else 0.0,
        }
        self._mb_log.append(entry)

        print(f"Mass balance @ step {step}: Σπ_t = {sum_pi_t:.4f} "
              f"(ratio={entry['ratio']:.4f}, Δmean_π={entry['mean_delta_pi']:.6f})")

    def get_critic_key_token_mask(self, batch, top_k, mode='entropy'):
        """CURE's entropy-only branch selection — used for condition B."""
        token_list = []
        if mode == 'entropy':
            log_prob_and_entropy = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = log_prob_and_entropy.batch["entropys"]
            response_ids = batch.batch["responses"]
            sorted_indices = torch.argsort(entropys, dim=1, descending=True)
        elif mode == 'random':
            response_ids = batch.batch["responses"]
            sorted_indices = torch.stack([
                torch.randperm(response_ids.shape[1], device=response_ids.device)
                for _ in range(response_ids.shape[0])
            ])

        batch_size = sorted_indices.shape[0]
        result_idx = torch.zeros(batch_size, dtype=torch.long)
        special_ids = set([int(x) for x in self.tokenizer.all_special_ids])

        for i in range(batch_size):
            valid_indices = []
            count = 0
            for idx in sorted_indices[i]:
                if idx == 0:
                    continue
                if count >= top_k:
                    break
                if idx < len(response_ids[i]):
                    token_id = response_ids[i, idx].item()
                    if int(token_id) not in special_ids:
                        token_text = self.tokenizer.decode(token_id)
                        if len(token_text.strip()) > 1:
                            valid_indices.append(idx)
                            count += 1

            if valid_indices:
                random_idx = torch.randint(0, len(valid_indices), (1,))
                result_idx[i] = valid_indices[random_idx]
            else:
                result_idx[i] = 0

        key_token_mask = torch.zeros_like(response_ids)
        for i in range(batch_size):
            key_token_mask[i, result_idx[i]] = 1

        for i in range(batch_size):
            token_list.append(self.tokenizer.decode(response_ids[i, result_idx[i]]))

        if batch_size > 0:
            print(f"Branch tokens: {token_list[:min(10, batch_size)]}")
            print(f"Branch positions: {result_idx[:min(10, batch_size)]}")

        return key_token_mask, result_idx

    def _get_valid_tokens(self, response_ids_seq, entropys_seq, pi_seq, special_ids):
        """Extract valid (non-special, non-trivial) token stats from one sequence."""
        seq_len = response_ids_seq.shape[0]
        H_vals, pi_vals, positions = [], [], []
        for j in range(1, seq_len):
            token_id = response_ids_seq[j].item()
            if int(token_id) in special_ids:
                continue
            token_text = self.tokenizer.decode(token_id)
            if len(token_text.strip()) <= 1:
                continue
            h_val = entropys_seq[j].item()
            p_val = pi_seq[j].item()
            if h_val > 0 and p_val > 0:
                H_vals.append(h_val)
                pi_vals.append(p_val)
                positions.append(j)
        return H_vals, pi_vals, positions

    def get_spark_branch_mask(self, batch):
        """
        3-gate filter branch selection for Spark-MCTS (condition C).

        Uses PER-SEQUENCE dynamic percentile calibration for the intervention
        trigger (matches 80/20 paper's validated approach). Each sequence gets
        its own tau_h/tau_p based on its own entropy/probability distribution.

        Gate 1: H(t) > tau_h  (per-sequence 95th-pct entropy)
        Gate 2: pi(t) < tau_p (per-sequence 30th-pct probability)
        Gate 3: pi(t) > kappa (noise floor — conservative lower bound)

        Returns: (key_token_mask, result_idx, spark_diagnostics)
        """
        kappa = self.config.trainer.get("spark_kappa", 0.02)
        entropy_pct = self.config.trainer.get("spark_entropy_percentile", 95)
        prob_pct = self.config.trainer.get("spark_prob_percentile", 30)

        # Forward pass to get entropies and log probs
        log_prob_and_entropy = self.actor_rollout_wg.compute_log_prob(batch)
        entropys = log_prob_and_entropy.batch["entropys"]   # (batch, seq_len)
        response_ids = batch.batch["responses"]              # (batch, seq_len)
        log_probs = log_prob_and_entropy.batch["old_log_probs"]  # (batch, seq_len)
        pi = torch.exp(log_probs)                            # chosen-token probability

        batch_size, seq_len = response_ids.shape
        special_ids = set(int(x) for x in self.tokenizer.all_special_ids)

        # Apply 3-gate filter per sequence with per-sequence thresholds
        result_idx = torch.zeros(batch_size, dtype=torch.long)
        token_list = []
        total_sparks = 0
        total_valid = 0
        per_seq_tau_h = []
        per_seq_tau_p = []
        sequences_with_sparks = 0

        for i in range(batch_size):
            H_vals, pi_vals, positions = self._get_valid_tokens(
                response_ids[i], entropys[i], pi[i], special_ids
            )
            total_valid += len(H_vals)

            if len(H_vals) < 5:
                # Too few valid tokens in this sequence — fall back to max entropy
                sorted_idx = torch.argsort(entropys[i], descending=True)
                for idx in sorted_idx:
                    if idx > 0 and idx < seq_len:
                        token_id = response_ids[i, idx].item()
                        if int(token_id) not in special_ids:
                            result_idx[i] = idx.item()
                            break
                token_list.append(self.tokenizer.decode(response_ids[i, result_idx[i]]))
                continue

            # Per-sequence dynamic thresholds
            H_t = torch.tensor(H_vals)
            pi_t = torch.tensor(pi_vals)
            tau_h_i = torch.quantile(H_t, entropy_pct / 100.0).item()

            # Fix: Gate 3 before Gate 2 — exclude noise-floor tokens before
            # computing tau_p. Silent forks (confident model, rare sampled token
            # with pi << kappa) sit in the left tail and drag tau_p down,
            # causing genuine sparks to fail Gate 2. Filtering by kappa first
            # anchors tau_p to the viable semantic candidate distribution.
            pi_above_kappa = pi_t[pi_t > kappa]
            if len(pi_above_kappa) >= 2:
                tau_p_i = torch.quantile(pi_above_kappa, prob_pct / 100.0).item()
            else:
                # Fallback: not enough tokens above noise floor
                tau_p_i = torch.quantile(pi_t, prob_pct / 100.0).item()

            per_seq_tau_h.append(tau_h_i)
            per_seq_tau_p.append(tau_p_i)

            # 3-gate filter within this sequence
            spark_positions = []
            for k, j in enumerate(positions):
                h_val = H_vals[k]
                p_val = pi_vals[k]
                g1 = h_val > tau_h_i
                g2 = 0 < p_val < tau_p_i
                g3 = p_val > kappa
                if g1 and g2 and g3:
                    spark_positions.append(j)
                    total_sparks += 1

            if spark_positions:
                chosen = spark_positions[torch.randint(0, len(spark_positions), (1,)).item()]
                result_idx[i] = chosen
                sequences_with_sparks += 1
            else:
                # No sparks in this sequence — fall back to highest-entropy valid position
                sorted_idx = torch.argsort(entropys[i], descending=True)
                for idx in sorted_idx:
                    if idx > 0 and idx < seq_len:
                        token_id = response_ids[i, idx].item()
                        if int(token_id) not in special_ids:
                            result_idx[i] = idx.item()
                            break

            token_list.append(self.tokenizer.decode(response_ids[i, result_idx[i]]))

        key_token_mask = torch.zeros_like(response_ids)
        for i in range(batch_size):
            key_token_mask[i, result_idx[i]] = 1

        spark_frac = total_sparks / max(total_valid, 1)
        mean_tau_h = float(np.mean(per_seq_tau_h)) if per_seq_tau_h else 0.0
        mean_tau_p = float(np.mean(per_seq_tau_p)) if per_seq_tau_p else 0.0
        diag = {
            "fallback": False,
            "tau_h_mean": mean_tau_h,
            "tau_h_std": float(np.std(per_seq_tau_h)) if per_seq_tau_h else 0.0,
            "tau_p_mean": mean_tau_p,
            "kappa": kappa,
            "total_valid": total_valid,
            "total_sparks": total_sparks,
            "spark_fraction": spark_frac,
            "sequences_with_sparks": sequences_with_sparks,
            "batch_size": batch_size,
        }

        print(f"3-Gate filter (per-seq): tau_h={mean_tau_h:.4f}±{diag['tau_h_std']:.4f}, "
              f"tau_p={mean_tau_p:.6f}, kappa={kappa}")
        print(f"  Sparks: {total_sparks}/{total_valid} ({spark_frac:.1%}), "
              f"{sequences_with_sparks}/{batch_size} seqs with sparks")
        print(f"  Branch tokens: {token_list[:min(5, batch_size)]}")
        print(f"  Branch positions: {result_idx[:min(5, batch_size)]}")

        return key_token_mask, result_idx, diag

    def fit(self):
        """
        Training loop for Spark-MCTS smoke test.

        Four modes controlled by config:
          Condition A: Vanilla GRPO (enable_branching=False)
          Condition B: CURE (enable_branching=True, use_3gate=False, vanilla loss)
          Condition C: Spark-MCTS (enable_branching=True, use_3gate=True,
                                   loss_mode=spark_nll, use_prefix_stopgrad=True)
          Condition D: Gradient routing (enable_branching=False,
                                   use_gradient_routing=True, loss_mode=spark_nll)

        Diagnostic checks per step:
          1. NaN/Inf detection on loss
          2. KL magnitude tracking (condition C)
          3. Prefix mask verification (condition C)
          4. Gradient routing verification (condition D)
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        enable_branching = self.config.trainer.get("enable_branching", True)
        use_3gate = self.config.trainer.get("use_3gate", False)
        use_prefix_stopgrad = self.config.trainer.get("use_prefix_stopgrad", False)
        use_gradient_routing = self.config.trainer.get("use_gradient_routing", False)
        routing_tau_h = self.config.trainer.get("routing_tau_h", 1.2800)
        st_log_freq = self.config.trainer.get("st_log_freq", 1)
        condition_name = self.config.trainer.get("condition_name", "unknown")

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc=f"[{condition_name}] Training",
        )

        self.global_steps += 1
        last_val_metrics = None
        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        # Diagnostic accumulators
        diagnostics = {
            "condition": condition_name,
            "steps": [],
            "nan_detected": False,
            "kl_magnitudes": [],
            "prefix_mask_checks": [],
            "spark_diagnostics": [],
        }

        st_trajectory = []

        # ===== MASS BALANCE CALIBRATION =====
        # Identify spark reference set on held-out problems before any weight updates
        mb_enabled = self._calibrate_mass_balance(n_problems=50)
        total_steps = self.total_training_steps
        if total_steps <= 10:
            mb_measure_steps = {total_steps}  # step 0 done in calibration
        else:
            mb_measure_steps = set(range(10, total_steps + 1, 10))

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                new_batch = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):

                    # ===== GENERATION =====
                    branch_positions = None  # saved for prefix stop-gradient
                    with marked_timer("gen", timing_raw, "red"):
                        if enable_branching:
                            n1 = self.config.trainer.n1
                            n2 = self.config.trainer.n2

                            # Stage 1: generate n1 full responses
                            gen_batch.meta_info["n"] = n1
                            gen_batch_output_1 = self.actor_rollout_wg.generate_sequences(gen_batch)
                            timing_raw.update(gen_batch_output_1.meta_info.get("timing", {}))
                            gen_batch_output_1.meta_info.pop("timing", None)

                            # Branch point selection
                            if use_3gate:
                                key_token_mask, result_idx, spark_diag = self.get_spark_branch_mask(
                                    gen_batch_output_1
                                )
                                diagnostics["spark_diagnostics"].append(spark_diag)
                            else:
                                key_token_mask, result_idx = self.get_critic_key_token_mask(
                                    gen_batch_output_1,
                                    top_k=self.config.trainer.critical_top_k,
                                    mode=self.config.trainer.critical_token_type,
                                )

                            # Save branch positions for prefix stop-gradient
                            branch_positions = result_idx.clone()

                            # Stage 2: re-concatenate up to branch point and generate n2 branches
                            gen_batch_output_1_2 = deepcopy(gen_batch_output_1)
                            if 'raw_prompt_ids' in gen_batch_output_1_2.non_tensor_batch.keys():
                                gen_batch_output_1_2.non_tensor_batch.pop('raw_prompt_ids')

                            pad_token_id = self.tokenizer.pad_token_id
                            batch_size = gen_batch_output_1_2.batch['input_ids'].shape[0]
                            gen_batch_output_1_2.batch['input_ids'] = torch.zeros_like(
                                gen_batch_output_1_2.batch['prompts']
                            )
                            gen_batch_output_1_2.batch['attention_mask'] = torch.zeros_like(
                                gen_batch_output_1_2.batch['prompts']
                            )
                            gen_batch_output_1_2.batch['position_ids'] = torch.zeros_like(
                                gen_batch_output_1_2.batch['prompts']
                            )

                            for i in range(batch_size):
                                non_pad_index = torch.nonzero(
                                    gen_batch_output_1_2.batch['prompts'][i] != pad_token_id,
                                    as_tuple=False,
                                )[0][0]
                                temp_input_ids = torch.cat([
                                    gen_batch_output_1_2.batch['prompts'][i][non_pad_index:],
                                    gen_batch_output_1_2.batch['responses'][i][:result_idx[i]],
                                ], dim=-1)
                                prompt_len = len(gen_batch_output_1_2.batch['prompts'][i])
                                copy_len = min(prompt_len, len(temp_input_ids))
                                gen_batch_output_1_2.batch['prompts'][i][-copy_len:] = \
                                    temp_input_ids[:copy_len]
                                gen_batch_output_1_2.batch['input_ids'][i] = \
                                    gen_batch_output_1_2.batch['prompts'][i]

                                non_pad_index = torch.nonzero(
                                    gen_batch_output_1_2.batch['prompts'][i] != pad_token_id,
                                    as_tuple=False,
                                )[0][0]
                                gen_batch_output_1_2.batch['attention_mask'][i][non_pad_index:] = 1
                                gen_batch_output_1_2.batch['position_ids'][i] = torch.clip(
                                    torch.cumsum(
                                        gen_batch_output_1_2.batch['attention_mask'][i], dim=-1
                                    ) - 1,
                                    min=0,
                                )

                            gen_batch_2 = gen_batch_output_1_2
                            gen_batch_2.meta_info["n"] = n2
                            gen_batch_output_2 = self.actor_rollout_wg.generate_sequences(gen_batch_2)
                            timing_raw.update(gen_batch_output_2.meta_info.get("timing", {}))
                            gen_batch_output_2.meta_info.pop("timing", None)

                            # Combine stage 1 and stage 2
                            l1 = gen_batch_output_1.chunk(self.config.data.gen_batch_size)
                            l2 = gen_batch_output_2.chunk(self.config.data.gen_batch_size)
                            combined = []
                            for idx in range(self.config.data.gen_batch_size):
                                combined.append(DataProto.concat([l1[idx], l2[idx]]))
                            gen_batch_output = DataProto.concat(combined)
                            gen_batch_output.meta_info["n"] = n1 + n2 * n1

                        else:
                            # Condition A: standard GRPO rollout (no branching)
                            gen_batch.meta_info["n"] = self.config.actor_rollout_ref.rollout.n
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                            gen_batch_output.meta_info.pop("timing", None)

                    # ===== |S_t| MEASUREMENT =====
                    if self.global_steps % st_log_freq == 0 or self.global_steps == 1:
                        with marked_timer("st_metric", timing_raw, "magenta"):
                            try:
                                st_batch = gen_batch_output_1 if enable_branching else gen_batch_output
                                st_metrics = self.compute_st_metric(st_batch)
                                metrics.update(st_metrics)
                                st_trajectory.append({"step": self.global_steps, **st_metrics})
                                frac_str = f"{st_metrics.get('st/spark_fraction_fixed', 0):.1%}"
                                pos_mean = st_metrics.get('st/spark_pos_frac_mean', -1)
                                pos_late = st_metrics.get('st/spark_pos_late_frac', -1)
                                pos_str = f", pos_frac={pos_mean:.2f} (late>{0.7}: {pos_late:.0%})" if pos_mean >= 0 else ""
                                print(
                                    f"[{condition_name}] Step {self.global_steps} "
                                    f"|S_t| = {st_metrics['st/mean_candidates']:.1f} "
                                    f"(+/-{st_metrics['st/std_candidates']:.1f}), "
                                    f"fixed spark frac={frac_str}{pos_str}"
                                )
                            except Exception as e:
                                print(f"Warning: |S_t| computation failed: {e}")

                    # ===== REWARD =====
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                        dtype=object,
                    )
                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        reward_extra_infos_dict = {}
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception:
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = \
                                new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:
                        # DAPO-style filtering (same as ablation)
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"],
                            new_batch.non_tensor_batch[metric_name],
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = [
                            idx for idx, traj_uid in enumerate(new_batch.non_tensor_batch["uid"])
                            if traj_uid in kept_prompt_uids
                        ]
                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(
                                    f"Generated {num_gen_batches} batches but still < {prompt_bsz} prompts"
                                )
                        else:
                            traj_bsz = (
                                self.config.data.train_batch_size
                                * self.config.actor_rollout_ref.rollout.n
                            )
                            batch = batch[:traj_bsz]

                    # ===== RESPONSE MASKING (V(R) gate + prefix stop-grad) =====
                    # Must happen BEFORE old_log_prob recompute so the masked
                    # response_mask is used for entropy aggregation too.
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # ===== V(R) GATE =====
                    # If all N2 branches from a trigger position yield identical
                    # rewards (V(R)=0), the fork was aleatoric — a formatting
                    # synonym where path choice didn't affect the result. Lp-Reg
                    # protection is unnecessary (GRPO advantages are zero anyway).
                    # Zero response_mask for those branches to skip them cleanly.
                    vr_gate_stats = None
                    if enable_branching and use_3gate and branch_positions is not None:
                        response_mask = batch.batch["response_mask"]
                        seq_rewards = batch.batch["token_level_scores"].sum(dim=-1)
                        n1_vr = self.config.trainer.n1
                        n2_vr = self.config.trainer.n2
                        bsz_total = response_mask.shape[0]
                        n_per = n1_vr + n1_vr * n2_vr
                        n_prompts_vr = bsz_total // n_per

                        vr_zero = 0
                        vr_total = 0
                        vr_masked = 0

                        for p in range(n_prompts_vr):
                            base = p * n_per
                            for pi in range(n1_vr):
                                br_start = base + n1_vr + pi * n2_vr
                                br_end = min(br_start + n2_vr, bsz_total)
                                br_rewards = seq_rewards[br_start:br_end]
                                vr_total += 1
                                if br_rewards.numel() > 1 and torch.var(br_rewards).item() == 0:
                                    vr_zero += 1
                                    response_mask[br_start:br_end] = 0
                                    vr_masked += (br_end - br_start)

                        batch.batch["response_mask"] = response_mask
                        vr_gate_stats = {
                            "vr_zero_forks": vr_zero,
                            "vr_total_forks": vr_total,
                            "vr_masked_responses": vr_masked,
                        }
                        if vr_zero > 0:
                            print(
                                f"  V(R) gate: {vr_zero}/{vr_total} forks have V(R)=0, "
                                f"masked {vr_masked} stage-2 responses"
                            )

                    prefix_mask_check = None
                    if use_prefix_stopgrad and enable_branching and branch_positions is not None:
                        response_mask = batch.batch["response_mask"]
                        n1 = self.config.trainer.n1
                        n2 = self.config.trainer.n2
                        batch_size_total = response_mask.shape[0]
                        n_per_prompt = n1 + n1 * n2
                        n_prompts = batch_size_total // n_per_prompt

                        prefix_zeros_applied = 0
                        prefix_total_positions = 0

                        for p_idx in range(n_prompts):
                            base = p_idx * n_per_prompt
                            # Stage-2 response indices within this prompt group: n1 .. n_per_prompt-1
                            # parent mapping: parent_s1 = (s2_idx - n1) // n2
                            for s2_idx in range(n1, n_per_prompt):
                                resp_idx = base + s2_idx
                                if resp_idx >= batch_size_total:
                                    break
                                parent_s1 = (s2_idx - n1) // n2
                                parent_global = base + parent_s1
                                if parent_global < len(branch_positions):
                                    bp = branch_positions[parent_global].item()
                                    if bp > 0:
                                        old_sum = response_mask[resp_idx, :bp].sum().item()
                                        response_mask[resp_idx, :bp] = 0
                                        prefix_zeros_applied += int(old_sum)
                                        prefix_total_positions += bp

                        batch.batch["response_mask"] = response_mask
                        prefix_mask_check = {
                            "prefix_zeros_applied": prefix_zeros_applied,
                            "prefix_total_positions": prefix_total_positions,
                        }
                        diagnostics["prefix_mask_checks"].append(prefix_mask_check)
                        print(
                            f"  Prefix stop-grad: zeroed {prefix_zeros_applied} mask positions "
                            f"across {prefix_total_positions} prefix positions"
                        )

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # Recompute old log probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        # Preserve entropys for gradient routing before popping
                        routing_entropys = entropys.clone() if use_gradient_routing else None
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # ===== GRADIENT ROUTING (Condition D) =====
                    # Zero response_mask for positive-advantage tokens at low-entropy
                    # positions. These are deterministic execution steps that led to
                    # success — not informative for exploration. Negative-advantage
                    # tokens keep full gradient (wrong execution steps need correction).
                    routing_stats = None
                    if use_gradient_routing and routing_entropys is not None:
                        advantages = batch.batch["advantages"]
                        response_mask = batch.batch["response_mask"]

                        assert routing_entropys.shape == response_mask.shape, (
                            f"routing_entropys shape {routing_entropys.shape} != "
                            f"response_mask shape {response_mask.shape}"
                        )

                        # Identify valid response tokens
                        valid = response_mask > 0

                        # High-entropy mask: entropy > τ_h at each position
                        high_ent = routing_entropys > routing_tau_h

                        # Positive-advantage tokens at LOW entropy → zero gradient
                        pos_adv = advantages > 0
                        zero_mask = pos_adv & ~high_ent & valid  # positive + low-entropy + valid

                        # Apply: zero response_mask where routing says to drop
                        n_zeroed = zero_mask.sum().item()
                        n_valid_before = valid.sum().item()
                        response_mask = response_mask.clone()
                        response_mask[zero_mask] = 0
                        batch.batch["response_mask"] = response_mask

                        # Diagnostics
                        n_valid_after = (response_mask > 0).sum().item()
                        high_ent_valid = (high_ent & valid).sum().item()
                        high_ent_frac = high_ent_valid / max(n_valid_before, 1)

                        routing_stats = {
                            "routing_zeroed": int(n_zeroed),
                            "routing_valid_before": int(n_valid_before),
                            "routing_valid_after": int(n_valid_after),
                            "routing_frac_zeroed": n_zeroed / max(n_valid_before, 1),
                            "routing_high_ent_frac": high_ent_frac,
                            "routing_tau_h": routing_tau_h,
                        }
                        metrics.update({
                            "routing/zeroed_tokens": int(n_zeroed),
                            "routing/frac_zeroed": n_zeroed / max(n_valid_before, 1),
                            "routing/high_ent_frac": high_ent_frac,
                        })
                        print(
                            f"  Gradient routing: zeroed {n_zeroed}/{n_valid_before} "
                            f"({n_zeroed/max(n_valid_before,1):.1%}) low-ent positive-adv tokens, "
                            f"high_ent_frac={high_ent_frac:.1%}, τ_h={routing_tau_h:.4f}"
                        )

                    # ===== UPDATE ACTOR =====
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        # ===== DIAGNOSTIC: NaN CHECK =====
                        loss_val = actor_output_metrics.get("actor/pg_loss", None)
                        grad_norm_val = actor_output_metrics.get("actor/grad_norm", None)

                        step_diag = {
                            "step": self.global_steps,
                            "pg_loss": float(loss_val) if loss_val is not None else None,
                            "grad_norm": float(grad_norm_val) if grad_norm_val is not None else None,
                            "entropy": metrics.get("actor/entropy", None),
                            # Fixed |S_t| metric from this step's generation
                            "st_fixed_sparks": metrics.get("st/total_sparks_fixed", None),
                            "st_fixed_fraction": metrics.get("st/spark_fraction_fixed", None),
                            # Spark position diagnostic
                            "spark_pos_frac_mean": metrics.get("st/spark_pos_frac_mean", None),
                            "spark_pos_frac_median": metrics.get("st/spark_pos_frac_median", None),
                            "spark_pos_late_frac": metrics.get("st/spark_pos_late_frac", None),
                        }

                        # V(R) gate diagnostics
                        if vr_gate_stats is not None:
                            step_diag["vr_gate"] = vr_gate_stats
                            if "vr_gate_trajectory" not in diagnostics:
                                diagnostics["vr_gate_trajectory"] = []
                            diagnostics["vr_gate_trajectory"].append({
                                "step": self.global_steps,
                                **vr_gate_stats,
                            })

                        # Gradient routing diagnostics
                        if routing_stats is not None:
                            step_diag["routing"] = routing_stats
                            if "routing_trajectory" not in diagnostics:
                                diagnostics["routing_trajectory"] = []
                            diagnostics["routing_trajectory"].append({
                                "step": self.global_steps,
                                **routing_stats,
                            })

                        if loss_val is not None and (math.isnan(loss_val) or math.isinf(loss_val)):
                            print(
                                f"  *** NaN/Inf DETECTED in loss at step {self.global_steps}! ***"
                            )
                            diagnostics["nan_detected"] = True
                            step_diag["nan"] = True

                        # NLL protection magnitude flows through actor/pg_clipfrac_lower
                        nll_prot_val = actor_output_metrics.get("actor/pg_clipfrac_lower", 0.0)
                        if nll_prot_val is not None:
                            nll_prot_val = float(nll_prot_val)
                        else:
                            nll_prot_val = 0.0
                        step_diag["nll_protect_mean"] = nll_prot_val
                        # GRPO-only = total pg_loss minus protection contribution
                        # (pg_loss includes protection; grpo_only ≈ pg_loss when protection is small)
                        step_diag["grpo_only_loss"] = abs(float(loss_val)) if loss_val is not None else 0.0
                        if nll_prot_val > 0:
                            diagnostics["kl_magnitudes"].append(nll_prot_val)
                            step_diag["lpreg_loss_magnitude"] = nll_prot_val
                        else:
                            step_diag["lpreg_loss_magnitude"] = 0.0

                        diagnostics["steps"].append(step_diag)

                    # ===== MASS BALANCE CHECKPOINT =====
                    if mb_enabled and self.global_steps in mb_measure_steps:
                        self._measure_mass_balance(self.global_steps)

                    # Validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                        # Capture pass@1 from validation metrics
                        # verl uses keys like "val-core/openai/gsm8k/reward/mean@1"
                        for vk, vv in val_metrics.items():
                            if "reward" in vk.lower() and "mean" in vk.lower():
                                step_diag["pass_at_1"] = float(vv) if vv is not None else None
                                if "pass_at_1_trajectory" not in diagnostics:
                                    diagnostics["pass_at_1_trajectory"] = []
                                diagnostics["pass_at_1_trajectory"].append({
                                    "step": self.global_steps,
                                    "value": float(vv) if vv is not None else None,
                                    "key": vk,
                                })
                                break

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                # Collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(
                    batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                ))
                timing_raw = defaultdict(float)
                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()

                    # Save diagnostics, |S_t| trajectory, and mass balance log
                    results_dir = self.config.trainer.get("results_dir", "/results")
                    os.makedirs(results_dir, exist_ok=True)

                    # Include mass balance in diagnostics and write separate log
                    if mb_enabled and hasattr(self, '_mb_log'):
                        diagnostics["mass_balance"] = self._mb_log
                        mb_path = os.path.join(results_dir, f"mass_balance_{condition_name}.json")
                        with open(mb_path, "w") as f:
                            json.dump({
                                "condition": condition_name,
                                "n_sparks": len(self._mb_spark_refs),
                                "spark_refs": self._mb_spark_refs,
                                "trajectory": self._mb_log,
                            }, f, indent=2, default=str)
                        print(f"Saved mass balance log to {mb_path}")

                    diagnostics["final_val_metrics"] = last_val_metrics

                    diag_path = os.path.join(results_dir, f"diagnostics_{condition_name}.json")
                    with open(diag_path, "w") as f:
                        json.dump(diagnostics, f, indent=2)

                    traj_path = os.path.join(results_dir, f"st_trajectory_{condition_name}.json")
                    with open(traj_path, "w") as f:
                        json.dump({
                            "condition": condition_name,
                            "st_trajectory": st_trajectory,
                            "final_val_metrics": last_val_metrics,
                        }, f, indent=2)

                    print(f"Saved diagnostics to {diag_path}")
                    print(f"Saved |S_t| trajectory to {traj_path}")
                    return diagnostics

                progress_bar.update(1)
                self.global_steps += 1

        return diagnostics
'''

# ============================================================
# MAIN ENTRY CODE — uses SparkMCTSTrainer instead of AblationTrainer
# ============================================================

MAIN_ENTRY_CODE = r'''
"""Entry point for Spark-MCTS smoke test training."""
import os
import socket
import sys

import hydra
import ray
from omegaconf import OmegaConf

# Add CURE repo to path
sys.path.insert(0, "/root/CURE")

from verl.trainer.ppo.reward import get_custom_reward_fn

def _extract_answer(text):
    """Extract answer from model output, handling both \\boxed{} and #### formats."""
    import re
    # Try \boxed{} first
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip().replace(",", "")
    # Try #### format
    m = re.search(r'####\s*(\-?[0-9\.\,]+)', text)
    if m:
        return m.group(1).strip().replace(",", "")
    # Flexible: last number in text
    nums = re.findall(r'\-?[0-9]+(?:\.[0-9]+)?', text)
    if nums:
        return nums[-1]
    return None

def custom_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Reward function that handles both \\boxed{} and #### answer formats."""
    answer = _extract_answer(solution_str)
    if answer is None:
        return 0.0
    # Normalize both sides
    try:
        if float(answer) == float(ground_truth):
            return 1.0
    except (ValueError, TypeError):
        pass
    # String comparison fallback
    if answer.strip() == str(ground_truth).strip():
        return 1.0
    return 0.0


@hydra.main(config_path="config", config_name="spark_trainer", version_base=None)
def main(config):
    run_spark(config)


def run_spark(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        from verl.utils import hf_processor, hf_tokenizer
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)

        assert config.actor_rollout_ref.actor.strategy == "fsdp"

        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        from verl.workers.reward_manager import get_reward_manager_cls
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        reward_manager_cls = get_reward_manager_cls(reward_manager_name)

        compute_score = custom_compute_score
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
        )

        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
        )

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # Import the Spark-MCTS trainer
        from recipe.spark.spark_trainer import SparkMCTSTrainer

        trainer = SparkMCTSTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        return trainer.fit()


if __name__ == "__main__":
    main()
'''

# ============================================================
# CONFIG YAML
# ============================================================

CONFIG_YAML = r"""
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  gen_batch_size: ${data.train_batch_size}

reward_model:
  reward_manager: naive
  overlong_buffer:
    enable: False
    len: 0
    penalty_factor: 0.0
    log: False

algorithm:
  filter_groups:
    enable: False
    metric: null
    max_num_gen_batches: 0

trainer:
  project_name: spark-mcts-smoke
"""

# ============================================================
# SMOKE CONFIG
# ============================================================

SMOKE_CONFIG = {
    "model_id": "Qwen/Qwen2.5-Math-1.5B",
    "max_prompt_length": 512,
    "max_response_length": 512,
    "train_batch_size": 8,  # 8 prompts per batch — need reward variance for non-zero advantages
    "gen_batch_size": 8,
    "ppo_mini_batch_size": 8,  # must be <= train_batch_size when use_dynamic_bsz=False
    "ppo_micro_batch_size": 2,  # must fit within mini_batch_size
    "ppo_max_token_len": 16384,
    "n1": 2,  # 2 stage-1 rollouts
    "n2": 2,  # 2 branches per rollout
    "n_total": 6,  # n1 + n1*n2 = 2 + 4 = 6
    "critical_top_k": 20,
    "critical_token_type": "entropy",
    "total_steps": 5,
    "lr": 1e-6,
    "clip_ratio": 0.2,
    "temperature": 1.0,
    "st_log_freq": 1,  # log every step for smoke
    "test_freq": 5,  # validate at final step only
    # Spark NLL protection
    "spark_nll_alpha": 0.3,
    "spark_logp_neg_k_percent": 0.01,
    # Spark-MCTS specific — dynamic intervention trigger (per-sequence percentile)
    "spark_entropy_percentile": 95,
    "spark_prob_percentile": 30,
    "spark_kappa": 0.02,
    # Fixed metric thresholds — from Run 2 pilot calibration (the ruler)
    "fixed_tau_h": 2.2697,  # 95th-pct entropy from Run 2
    "fixed_tau_p": 0.7005,  # 30th-pct probability from Run 2
    "fixed_kappa": 0.02,
}

FULL_CONFIG = {
    "model_id": "Qwen/Qwen2.5-Math-1.5B",
    "max_prompt_length": 512,
    "max_response_length": 512,
    "train_batch_size": 16,
    "gen_batch_size": 16,
    "ppo_mini_batch_size": 16,
    "ppo_micro_batch_size": 4,
    "ppo_max_token_len": 32768,
    "n1": 4,
    "n2": 4,
    "n_total": 20,  # n1 + n1*n2 = 4 + 16 = 20
    "critical_top_k": 20,
    "critical_token_type": "entropy",
    "total_steps": 50,
    "lr": 1e-6,
    "clip_ratio": 0.2,
    "temperature": 1.0,
    "st_log_freq": 1,
    "test_freq": 10,
    # Spark NLL protection
    "spark_nll_alpha": 0.3,
    "spark_logp_neg_k_percent": 0.01,
    # Spark-MCTS specific
    "spark_entropy_percentile": 95,
    "spark_prob_percentile": 30,
    "spark_kappa": 0.02,
    # Fixed metric thresholds
    "fixed_tau_h": 2.2697,
    "fixed_tau_p": 0.7005,
    "fixed_kappa": 0.02,
    # Full-run hardware/infra
    "n_gpus": 2,
    "gpu_memory_utilization": 0.80,
    "use_remove_padding": True,
    "use_dynamic_bsz": True,
    "logger": "wandb",
    "project_name": "spark-mcts-ablation",
    "save_freq": 10,
}

SWEEP_CONFIG = {
    **SMOKE_CONFIG,
    "total_steps": 10,
    "test_freq": -1,  # no validation during sweep — only comparing loss magnitudes
    "spark_nll_alpha": 0.3,  # default, overridden per trial
}

VIABILITY_CONFIG = {
    **SMOKE_CONFIG,
    "total_steps": 100,
    "n_gpus": 2,
    "n1": 2,
    "n2": 2,
    "n_total": 6,
    "test_freq": 10,
    "spark_nll_alpha": 0.3,  # default, overridden by sweep winner
    "gpu_memory_utilization": 0.80,
    "use_remove_padding": True,
    "use_dynamic_bsz": True,
    "logger": "wandb",
    "project_name": "spark-mcts-viability",
    "save_freq": 25,
}

MATH_SMOKE_CONFIG = {
    **SMOKE_CONFIG,
    "dataset": "math_l34",
    "n1": 2,
    "n2": 4,  # 4 branches per fork — need epistemic signal
    "n_total": 10,  # n1 + n1*n2 = 2 + 2*4 = 10
    "total_steps": 5,
    "test_freq": 5,
    "max_response_length": 1024,  # MATH needs longer responses
    "ppo_max_token_len": 32768,  # accommodate longer sequences
    "n_gpus": 2,
    "gpu_memory_utilization": 0.80,
    "use_remove_padding": True,
    "use_dynamic_bsz": True,
}

MATH_VIABILITY_CONFIG = {
    **MATH_SMOKE_CONFIG,
    "total_steps": 100,
    "test_freq": 10,
    "logger": "wandb",
    "project_name": "spark-mcts-math-viability",
    "save_freq": 25,
    "spark_nll_alpha": 0.3,
}

# Condition D: Gradient routing without branching
ROUTING_SMOKE_CONFIG = {
    **SMOKE_CONFIG,
    "n_routing": 16,  # flat GRPO with 16 rollouts
    "n_total": 16,
    "routing_tau_h": 1.2800,  # entropy threshold for high-entropy classification
    "total_steps": 5,
    "test_freq": 5,
    "ppo_max_token_len": 32768,  # accommodate 16 rollouts
}

ROUTING_VIABILITY_CONFIG = {
    **ROUTING_SMOKE_CONFIG,
    "total_steps": 100,
    "n_gpus": 2,
    "test_freq": 10,
    "gpu_memory_utilization": 0.80,
    "use_remove_padding": True,
    "use_dynamic_bsz": True,
    "logger": "wandb",
    "project_name": "spark-routing-viability",
    "save_freq": 25,
}

# Run 11: Trigger signal diagnostic (forward pass only, no training)
TRIGGER_DIAG_CONFIG = {
    "model_id": "Qwen/Qwen2.5-Math-1.5B",
    "n_problems": 20,
    "n_branches": 4,
    "k_positions": 5,  # per set (H, S, R)
    "max_probe_pos": 200,  # only look at first 200 generated tokens
    "max_response_tokens": 1024,
    "temperature": 1.0,  # for branching rollouts
    "seed": 42,
}


def _compute_entropy_from_logprobs(logprobs_list):
    """Approximate entropy from vLLM top-k logprobs per token."""
    import math as _math

    entropies = []
    for token_logprobs in logprobs_list:
        if not token_logprobs:
            entropies.append(0.0)
            continue
        log_ps = [lp.logprob for lp in token_logprobs.values()]
        ps = [_math.exp(lp) for lp in log_ps]
        total = sum(ps)
        if total <= 0:
            entropies.append(0.0)
            continue
        ps_norm = [p / total for p in ps]
        ent = -sum(p * _math.log(p) for p in ps_norm if p > 0)
        entropies.append(ent)
    return entropies


def _find_structural_positions(token_ids, tokenizer, max_pos=200):
    """Find step-boundary token positions in first max_pos generated tokens.

    Structural markers: \\n\\n, Step N:, Therefore/Thus/Hence.
    Returns sorted list of up to 5 token positions.
    """
    import re

    limit = min(max_pos, len(token_ids))
    if limit == 0:
        return []

    # Build token→char offset mapping by decoding incrementally
    tok_start_char = []
    char_offset = 0
    pieces = []
    for i in range(limit):
        piece = tokenizer.decode([token_ids[i]])
        tok_start_char.append(char_offset)
        pieces.append(piece)
        char_offset += len(piece)

    full_text = "".join(pieces)

    # Build char→token reverse mapping
    char_to_tok = {}
    for i in range(limit):
        start = tok_start_char[i]
        end = start + len(pieces[i])
        for c in range(start, end):
            char_to_tok[c] = i

    # Find structural markers
    positions = set()
    for m in re.finditer(r"\n\n", full_text):
        tok = char_to_tok.get(m.start())
        if tok is not None and tok > 0:
            positions.add(tok)
    for m in re.finditer(r"Step\s+\d+[:\.]", full_text, re.IGNORECASE):
        tok = char_to_tok.get(m.start())
        if tok is not None and tok > 0:
            positions.add(tok)
    for m in re.finditer(r"\b(Therefore|Thus|Hence|So,)\b", full_text):
        tok = char_to_tok.get(m.start())
        if tok is not None and tok > 0:
            positions.add(tok)

    return sorted(positions)[:5]


def get_hydra_overrides(condition: str, config=None, results_base=None):
    """
    Generate Hydra config overrides for each condition.

    A: Vanilla GRPO  (no branching, vanilla loss)
    B: CURE          (entropy-only branching, vanilla loss, no prefix stop-grad)
    C: Spark-MCTS    (3-gate branching, Lp-Reg loss, prefix stop-grad)
    D: Gradient routing (no branching, NLL loss, entropy-based gradient mask)
    """
    cfg = config or SMOKE_CONFIG
    results_base = results_base or "/results/spark_smoke"

    # Extract config values with smoke-test defaults
    dataset = cfg.get("dataset", "gsm8k")
    use_remove_padding = cfg.get("use_remove_padding", False)
    use_dynamic_bsz = cfg.get("use_dynamic_bsz", False)
    gpu_mem_util = cfg.get("gpu_memory_utilization", 0.5)
    n_gpus = cfg.get("n_gpus", 1)
    logger_name = cfg.get("logger", "console")
    project_name = cfg.get("project_name", "spark-mcts-smoke")
    save_freq = cfg.get("save_freq", -1)

    # Dataset paths
    data_paths = {
        "gsm8k": ("/root/data/gsm8k/train.parquet", "/root/data/gsm8k/test.parquet"),
        "math_l34": (
            "/root/data/math_l34/train.parquet",
            "/root/data/math_l34/test.parquet",
        ),
    }
    train_path, val_path = data_paths[dataset]

    if condition == "A":
        loss_mode = "vanilla"
        enable_branching = False
        use_3gate = False
        use_prefix_stopgrad = False
        use_gradient_routing = False
        n_resp = cfg["n_total"]
    elif condition == "B":
        loss_mode = "vanilla"
        enable_branching = True
        use_3gate = False
        use_prefix_stopgrad = False
        use_gradient_routing = False
        n_resp = cfg["n_total"]
    elif condition == "C":
        loss_mode = "spark_nll"
        enable_branching = True
        use_3gate = True
        use_prefix_stopgrad = True
        use_gradient_routing = False
        n_resp = cfg["n_total"]
    elif condition == "D":
        # Gradient routing without branching: flat GRPO + entropy-based mask on positive-advantage tokens
        loss_mode = "spark_nll"
        enable_branching = False
        use_3gate = False
        use_prefix_stopgrad = False
        use_gradient_routing = True
        n_resp = cfg.get("n_routing", cfg["n_total"])
    else:
        raise ValueError(f"Unknown condition: {condition}")

    overrides = [
        # Data
        f"data.train_files={train_path}",
        f"data.val_files={val_path}",
        f"data.prompt_key=prompt",
        f"data.truncation=left",
        f"data.max_prompt_length={cfg['max_prompt_length']}",
        f"data.max_response_length={cfg['max_response_length']}",
        f"data.train_batch_size={cfg['train_batch_size']}",
        f"data.gen_batch_size={cfg['gen_batch_size']}",
        f"data.trust_remote_code=True",
        # Model
        f"actor_rollout_ref.model.path={cfg['model_id']}",
        f"actor_rollout_ref.model.trust_remote_code=True",
        f"actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.model.use_remove_padding={use_remove_padding}",
        # Actor
        f"actor_rollout_ref.actor.strategy=fsdp",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg['ppo_micro_batch_size']}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={use_dynamic_bsz}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={cfg['ppo_max_token_len']}",
        f"actor_rollout_ref.actor.grad_clip=1.0",
        f"actor_rollout_ref.actor.clip_ratio={cfg['clip_ratio']}",
        f"actor_rollout_ref.actor.clip_ratio_low={cfg['clip_ratio']}",
        f"actor_rollout_ref.actor.clip_ratio_high={cfg['clip_ratio']}",
        f"actor_rollout_ref.actor.loss_agg_mode=token-mean",
        f"actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.ppo_epochs=1",
        f"actor_rollout_ref.actor.use_kl_loss=False",
        f"actor_rollout_ref.actor.use_torch_compile=False",  # skip compile overhead
        f"actor_rollout_ref.actor.policy_loss.loss_mode={loss_mode}",
        f"actor_rollout_ref.actor.optim.lr={cfg['lr']}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps=1",
        f"actor_rollout_ref.actor.optim.weight_decay=0.01",
        f"actor_rollout_ref.actor.fsdp_config.param_offload=False",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        # Rollout
        f"actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.temperature={cfg['temperature']}",
        f"actor_rollout_ref.rollout.top_p=1.0",
        f"actor_rollout_ref.rollout.top_k=-1",
        f"actor_rollout_ref.rollout.n={n_resp}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_mem_util}",
        f"actor_rollout_ref.rollout.enforce_eager=True",
        f"actor_rollout_ref.rollout.free_cache_engine=True",
        f"actor_rollout_ref.rollout.max_num_batched_tokens=8192",
        f"actor_rollout_ref.rollout.enable_chunked_prefill=True",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={cfg['ppo_micro_batch_size']}",
        f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={use_dynamic_bsz}",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={cfg['ppo_max_token_len']}",
        # Algorithm
        f"algorithm.adv_estimator=grpo",
        f"algorithm.use_kl_in_reward=False",
        f"algorithm.filter_groups.enable=False",
        # Trainer
        f"trainer.nnodes=1",
        f"trainer.n_gpus_per_node={n_gpus}",
        f"trainer.total_epochs=1000",
        f"trainer.total_training_steps={cfg['total_steps']}",
        f"trainer.logger=['{logger_name}']",
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name=condition_{condition}",
        f"trainer.val_before_train=False",
        f"trainer.test_freq={cfg['test_freq']}",
        f"trainer.save_freq={save_freq}",
        f"trainer.balance_batch=False",
        f"trainer.resume_mode=disable",
        f"trainer.device=cuda",
        # Custom smoke params
        f"+trainer.enable_branching={enable_branching}",
        f"+trainer.use_gradient_routing={use_gradient_routing}",
        f"+trainer.st_log_freq={cfg['st_log_freq']}",
        f"+trainer.condition_name={condition}",
        f"+trainer.results_dir={results_base}/condition_{condition}",
    ]

    # CURE/Spark branching params
    if enable_branching:
        overrides.extend(
            [
                f"+trainer.n1={cfg['n1']}",
                f"+trainer.n2={cfg['n2']}",
                f"+trainer.critical_top_k={cfg['critical_top_k']}",
                f"+trainer.critical_token_type={cfg['critical_token_type']}",
            ]
        )

    # Fixed metric thresholds (all conditions — the ruler never changes)
    overrides.extend(
        [
            f"+trainer.fixed_tau_h={cfg['fixed_tau_h']}",
            f"+trainer.fixed_tau_p={cfg['fixed_tau_p']}",
            f"+trainer.fixed_kappa={cfg['fixed_kappa']}",
        ]
    )

    # Spark-MCTS specific params (condition C)
    if use_3gate:
        overrides.extend(
            [
                f"+trainer.use_3gate=True",
                f"+trainer.use_prefix_stopgrad={use_prefix_stopgrad}",
                f"+trainer.spark_entropy_percentile={cfg['spark_entropy_percentile']}",
                f"+trainer.spark_prob_percentile={cfg['spark_prob_percentile']}",
                f"+trainer.spark_kappa={cfg['spark_kappa']}",
            ]
        )
    else:
        overrides.extend(
            [
                f"+trainer.use_3gate=False",
                f"+trainer.use_prefix_stopgrad=False",
            ]
        )

    # Spark NLL protection params
    if loss_mode == "spark_nll":
        overrides.extend(
            [
                f"++actor_rollout_ref.actor.policy_loss.logp_neg_k_percent={cfg['spark_logp_neg_k_percent']}",
                f"++actor_rollout_ref.actor.policy_loss.nll_alpha={cfg['spark_nll_alpha']}",
            ]
        )

    # Gradient routing params (condition D)
    if use_gradient_routing:
        overrides.append(f"+trainer.routing_tau_h={cfg.get('routing_tau_h', 1.2800)}")

    overrides.append("ray_init.num_cpus=4")
    return overrides


# ============================================================
# PATCH APPLICATION — identical logic to modal_ablation.py
# ============================================================


def apply_patches():
    """Apply all patches to the CURE repo at runtime."""
    import os

    CURE_ROOT = "/root/CURE"

    # Patch 1: Append Lp-Reg loss to core_algos.py
    core_algos_path = os.path.join(CURE_ROOT, "verl/trainer/ppo/core_algos.py")
    with open(core_algos_path, "r") as f:
        existing = f.read()
    if "compute_policy_loss_spark_nll" not in existing:
        with open(core_algos_path, "a") as f:
            f.write(CORE_ALGOS_APPEND)
        print(f"Patched: {core_algos_path}")
    else:
        print(f"Already patched: {core_algos_path}")

    # Patch 2: Modify dp_actor.py
    dp_actor_path = os.path.join(CURE_ROOT, "verl/workers/actor/dp_actor.py")
    with open(dp_actor_path, "r") as f:
        content = f.read()

    patched = False

    # 2a: Add min_p to _forward_micro_batch signature
    if "min_p=None" not in content:
        assert (
            DP_ACTOR_SIG_OLD in content
        ), "Could not find _forward_micro_batch signature to patch"
        content = content.replace(DP_ACTOR_SIG_OLD, DP_ACTOR_SIG_NEW)
        patched = True

    # 2b: Add min-p filtering in non-remove_padding path
    if "min_p is not None and min_p > 0" not in content:
        assert (
            DP_ACTOR_FORWARD_OLD in content
        ), "Could not find forward_micro_batch body to patch"
        content = content.replace(DP_ACTOR_FORWARD_OLD, DP_ACTOR_FORWARD_NEW)
        patched = True

    # 2c: Add min-p filtering in remove_padding path
    if "Min-p filtering for Lp-Reg (remove_padding path)" not in content:
        assert (
            DP_ACTOR_RMPAD_OLD in content
        ), "Could not find remove_padding logprobs section to patch"
        content = content.replace(DP_ACTOR_RMPAD_OLD, DP_ACTOR_RMPAD_NEW)
        patched = True

    # 2d: Add import for compute_policy_loss_spark_nll (before 2e to avoid shadowing)
    if "compute_policy_loss_spark_nll" not in content:
        assert (
            DP_ACTOR_IMPORT_OLD in content
        ), "Could not find dp_actor import line to patch"
        content = content.replace(DP_ACTOR_IMPORT_OLD, DP_ACTOR_IMPORT_NEW)
        patched = True

    # 2e: Add Spark NLL branch in update_policy
    if 'loss_mode == "spark_nll"' not in content:
        assert (
            DP_ACTOR_LOSS_OLD in content
        ), "Could not find update_policy loss section to patch"
        content = content.replace(DP_ACTOR_LOSS_OLD, DP_ACTOR_LOSS_NEW)
        patched = True

    # 2f: Make flash_attn import safe (optional dep)
    if "try:" not in content.split("from flash_attn")[0].split("\n")[-1]:
        if DP_ACTOR_FLASH_OLD in content:
            content = content.replace(DP_ACTOR_FLASH_OLD, DP_ACTOR_FLASH_NEW)
            patched = True

    if patched:
        with open(dp_actor_path, "w") as f:
            f.write(content)
        print(f"Patched: {dp_actor_path}")
    else:
        print(f"Already patched: {dp_actor_path}")

    # Patch 3: Create spark recipe directory
    recipe_dir = os.path.join(CURE_ROOT, "recipe/spark")
    config_dir = os.path.join(recipe_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    with open(os.path.join(recipe_dir, "__init__.py"), "w") as f:
        f.write("")

    with open(os.path.join(recipe_dir, "spark_trainer.py"), "w") as f:
        f.write(SPARK_TRAINER_CODE)

    with open(os.path.join(recipe_dir, "main.py"), "w") as f:
        f.write(MAIN_ENTRY_CODE)

    with open(os.path.join(config_dir, "spark_trainer.yaml"), "w") as f:
        f.write(CONFIG_YAML)

    # Patch 4: Create data prep script
    with open(os.path.join(CURE_ROOT, "prepare_data.py"), "w") as f:
        f.write(DATA_PREP_SCRIPT)

    # Patch 5: Relax vLLM v1 input token validation for Qwen2.5
    # vLLM 0.8.4 rejects Qwen2.5 special tokens (ID > tokenizer base vocab
    # 151643) when fed back as input during branching stage 2.  The upstream
    # fix (PR #22471) uses max(tokenizer.max_token_id, model_vocab_size-1)
    # but is not in 0.8.4.  We simply disable the raise so the model
    # embedding handles it (invalid IDs would IndexError anyway).
    import glob as _glob
    import re as _re
    import subprocess as _sp

    # Find all vLLM files containing the error message
    vllm_pkg = "/usr/local/lib/python3.10/site-packages/vllm"
    try:
        result = _sp.run(
            ["grep", "-rl", "out of vocabulary", vllm_pkg],
            capture_output=True,
            text=True,
            timeout=10,
        )
        vllm_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        vllm_files = []
    # Also check the specific known paths
    for candidate in [
        os.path.join(vllm_pkg, "v1/engine/input_processor.py"),
        os.path.join(vllm_pkg, "v1/engine/processor.py"),
        os.path.join(vllm_pkg, "engine/input_processor.py"),
    ]:
        if os.path.exists(candidate) and candidate not in vllm_files:
            vllm_files.append(candidate)

    patched_any = False
    for vf in vllm_files:
        if not vf.endswith(".py"):
            continue
        with open(vf, "r") as f:
            vc = f.read()
        if "out of vocabulary" in vc and "# Patched:" not in vc:
            vc = _re.sub(
                r'raise ValueError\(f"Token id \{max_input_id\} is out of vocabulary"\)',
                "pass  # Patched: Qwen2.5 special tokens allowed (PR #22471 backport)",
                vc,
            )
            with open(vf, "w") as f:
                f.write(vc)
            print(f"Patched vLLM input validator: {vf}")
            patched_any = True
    if not patched_any:
        print("No vLLM files needed patching (already patched or not found)")

    print("All patches applied.")


# ============================================================
# VERIFICATION LOGIC
# ============================================================


def verify_diagnostics(all_results, conditions=None):
    """
    Verify the 3 backward pass criteria.

    Check 1: Loss finite (no NaN/Inf) in all conditions.
    Check 2: KL fires with non-zero magnitude in condition C.
    Check 3: Prefix stop-gradient was applied in condition C.

    Returns list of (status, message) tuples.
    """
    import numpy as np

    conditions = conditions or ["A", "B", "C"]
    checks = []

    # Check 1: No NaN in any condition
    for cond in conditions:
        result = all_results.get(cond, {})
        if result.get("nan_detected", False):
            checks.append(("FAIL", f"Condition {cond}: NaN/Inf detected in loss"))
        elif not result:
            checks.append(
                ("FAIL", f"Condition {cond}: no diagnostic data (run failed?)")
            )
        else:
            loss_vals = [
                s.get("pg_loss")
                for s in result.get("steps", [])
                if s.get("pg_loss") is not None
            ]
            if loss_vals:
                checks.append(
                    (
                        "PASS",
                        f"Condition {cond}: loss finite across {len(loss_vals)} steps "
                        f"(mean={np.mean(loss_vals):.4f})",
                    )
                )
            else:
                checks.append(("WARN", f"Condition {cond}: no pg_loss values recorded"))

    # Check 2: Lp-Reg loss fires in condition C (KL is embedded in pg_loss)
    c_result = all_results.get("C", {})
    kl_mags = c_result.get("kl_magnitudes", [])
    if kl_mags and any(m > 0 for m in kl_mags):
        checks.append(
            (
                "PASS",
                f"Condition C: Lp-Reg loss fires (mean={np.mean(kl_mags):.6f}, "
                f"max={max(kl_mags):.6f}, n={len(kl_mags)})",
            )
        )
    elif kl_mags:
        checks.append(
            ("FAIL", f"Condition C: Lp-Reg loss recorded but all zero ({kl_mags})")
        )
    else:
        checks.append(("FAIL", f"Condition C: no Lp-Reg loss magnitudes recorded"))

    # Check 3: Prefix stop-gradient applied in condition C
    prefix_checks = c_result.get("prefix_mask_checks", [])
    if prefix_checks and any(
        pc.get("prefix_zeros_applied", 0) > 0 for pc in prefix_checks
    ):
        total_zeros = sum(pc.get("prefix_zeros_applied", 0) for pc in prefix_checks)
        total_positions = sum(
            pc.get("prefix_total_positions", 0) for pc in prefix_checks
        )
        checks.append(
            (
                "PASS",
                f"Condition C: prefix stop-grad applied "
                f"({total_zeros} positions zeroed across {total_positions} prefix positions)",
            )
        )
    elif prefix_checks:
        checks.append(
            ("FAIL", "Condition C: prefix_mask_checks present but no zeros applied")
        )
    else:
        checks.append(
            ("FAIL", "Condition C: prefix stop-grad NOT applied (no checks recorded)")
        )

    # Spark diagnostic summary for condition C
    spark_diags = c_result.get("spark_diagnostics", [])
    if spark_diags:
        real_diags = [d for d in spark_diags if not d.get("fallback", False)]
        fallbacks = [d for d in spark_diags if d.get("fallback", False)]
        if real_diags:
            mean_frac = np.mean([d["spark_fraction"] for d in real_diags])
            checks.append(
                (
                    "INFO",
                    f"Condition C: 3-gate filter (per-seq dynamic) ran {len(real_diags)} times "
                    f"(fallback: {len(fallbacks)}), mean spark fraction={mean_frac:.1%}",
                )
            )
        else:
            checks.append(
                ("WARN", "Condition C: 3-gate filter always fell back to entropy-only")
            )

    # V(R) gate summary for condition C
    vr_traj = c_result.get("vr_gate_trajectory", [])
    if vr_traj:
        total_zero = sum(v["vr_zero_forks"] for v in vr_traj)
        total_forks = sum(v["vr_total_forks"] for v in vr_traj)
        total_masked = sum(v["vr_masked_responses"] for v in vr_traj)
        zero_rate = total_zero / total_forks if total_forks > 0 else 0
        checks.append(
            (
                "INFO",
                f"Condition C: V(R) gate fired {total_zero}/{total_forks} forks "
                f"({zero_rate:.0%} aleatoric), masked {total_masked} responses",
            )
        )

    # Check 4: Fixed-threshold |S_t| metric is logging for all conditions
    for cond in conditions:
        cond_result = all_results.get(cond, {})
        steps = cond_result.get("steps", [])
        st_vals = [
            s.get("st_fixed_sparks")
            for s in steps
            if s.get("st_fixed_sparks") is not None
        ]
        if st_vals:
            checks.append(
                (
                    "INFO",
                    f"Condition {cond}: fixed |S_t| logged {len(st_vals)} steps "
                    f"(mean={np.mean(st_vals):.1f}, range=[{min(st_vals):.0f}, {max(st_vals):.0f}])",
                )
            )

    # Check 5: Mass balance Σπ_t logged across conditions
    for cond in conditions:
        cond_result = all_results.get(cond, {})
        mb_log = cond_result.get("mass_balance", [])
        if mb_log and len(mb_log) > 1:
            pi_0 = mb_log[0]["sum_pi"]
            pi_final = mb_log[-1]["sum_pi"]
            ratio = mb_log[-1]["ratio"]
            checks.append(
                (
                    "INFO",
                    f"Condition {cond}: mass balance Σπ_t logged {len(mb_log)} points "
                    f"(Σπ_0={pi_0:.4f}, Σπ_final={pi_final:.4f}, ratio={ratio:.4f})",
                )
            )
        elif mb_log:
            checks.append(
                (
                    "INFO",
                    f"Condition {cond}: mass balance calibration only "
                    f"({mb_log[0].get('n_sparks', 0)} sparks, Σπ_0={mb_log[0].get('sum_pi', 0):.4f})",
                )
            )

    return checks


# ============================================================
# MODAL IMAGE — same as modal_ablation.py
# ============================================================

smoke_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("vllm==0.8.4")
    .pip_install(
        "transformers>=4.45.0,<5.0.0",
        "accelerate",
        "codetiming",
        "datasets",
        "dill",
        "hydra-core",
        "pandas",
        "peft",
        "pyarrow>=19.0.0",
        "pybind11",
        "pylatexenc",
        "ray[default]>=2.41.0",
        "torchdata",
        "tensordict<=0.6.2",
        "wandb",
        "math-verify",
        "huggingface_hub",
    )
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
        " || echo 'flash-attn prebuilt wheel not found; use_remove_padding=False required'",
    )
    .run_commands(
        "git clone https://github.com/bytedance/CURE /root/CURE",
        "touch /root/CURE/README.md && cd /root/CURE && pip install -e '.' --no-deps",
    )
)


# ============================================================
# SINGLE-CONDITION RUNNER (called locally within run_backward_smoke)
# ============================================================


def run_single_condition(condition: str, config=None, results_base=None):
    """
    Run one condition (A, B, or C).
    Returns the diagnostics dict produced by SparkMCTSTrainer.fit().
    """
    import subprocess
    import sys
    import time

    cfg = config or SMOKE_CONFIG
    results_base = results_base or "/results/spark_smoke"

    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"RUNNING CONDITION {condition} ({cfg['total_steps']} steps)")
    print(f"{'=' * 60}")

    overrides = get_hydra_overrides(condition, config=cfg, results_base=results_base)
    cmd = [sys.executable, "-m", "recipe.spark.main"] + overrides

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/CURE:" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        cwd="/root/CURE",
        env=env,
        capture_output=False,
    )

    elapsed = time.time() - t0
    print(f"\nCondition {condition} exit code: {result.returncode} ({elapsed:.0f}s)")

    # Read diagnostics from disk
    diag_path = f"{results_base}/condition_{condition}/diagnostics_{condition}.json"
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            diagnostics = json.load(f)
        print(f"Loaded diagnostics from {diag_path}")
    else:
        print(f"WARNING: no diagnostics file at {diag_path}")
        diagnostics = {
            "condition": condition,
            "exit_code": result.returncode,
            "nan_detected": result.returncode != 0,
            "kl_magnitudes": [],
            "prefix_mask_checks": [],
            "spark_diagnostics": [],
            "steps": [],
            "error": "diagnostics file not written",
        }

    diagnostics["exit_code"] = result.returncode
    diagnostics["elapsed_seconds"] = elapsed
    return diagnostics


# ============================================================
# MODAL FUNCTION
# ============================================================


@app.function(
    gpu="A100-80GB",
    timeout=7200,  # 2 hours max
    image=smoke_image,
    volumes={
        "/hf-cache": hf_cache,
        "/results": results_vol,
    },
)
def run_backward_smoke():
    """
    Run all 3 conditions for 5 steps each and verify backward pass.

    Checks:
      1. Loss finite (no NaN/Inf) in all conditions
      2. KL fires in condition C (Lp-Reg acting on spark tokens)
      3. Prefix stop-grad zeroes prefix positions in condition C
    """
    import sys
    import time

    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"
    os.environ["WANDB_MODE"] = "disabled"

    start = time.time()
    print("=" * 60)
    print("Spark-MCTS Backward Pass Smoke Test")
    print(f"Model:  {SMOKE_CONFIG['model_id']}")
    print(f"Steps:  {SMOKE_CONFIG['total_steps']} per condition")
    print(
        f"n1={SMOKE_CONFIG['n1']}, n2={SMOKE_CONFIG['n2']}, n_total={SMOKE_CONFIG['n_total']}"
    )
    print("=" * 60)

    # ---- Step 1: Apply patches ----
    print("\n--- Step 1: Applying patches ---")
    apply_patches()

    # Verify patches applied correctly
    with open("/root/CURE/verl/workers/actor/dp_actor.py") as f:
        dp_content = f.read()
    patch_checks = {
        "min_p=None": "signature patch",
        "Min-p filtering for Lp-Reg (remove_padding path)": "rmpad patch",
        "Min-p filtering for Lp-Reg target distribution": "non-rmpad patch",
        'loss_mode == "spark_nll"': "loss branch",
        "compute_policy_loss_spark_nll": "import",
    }
    for marker, desc in patch_checks.items():
        assert marker in dp_content, f"Patch verification FAILED: {desc} ({marker!r})"
        print(f"  [OK] {desc}")

    with open("/root/CURE/verl/trainer/ppo/core_algos.py") as f:
        ca_content = f.read()
    assert (
        "compute_policy_loss_spark_nll" in ca_content
    ), "Spark NLL loss not in core_algos"
    print("  [OK] Lp-Reg loss function in core_algos.py")

    # ---- Step 2: Prepare data ----
    print("\n--- Step 2: Preparing GSM8K data ---")
    import subprocess

    sys.path.insert(0, "/root/CURE")
    subprocess.run([sys.executable, "/root/CURE/prepare_data.py"], check=True)
    assert os.path.exists("/root/data/gsm8k/train.parquet"), "train.parquet missing"
    assert os.path.exists("/root/data/gsm8k/test.parquet"), "test.parquet missing"
    print("  GSM8K parquet files ready.")

    # ---- Step 3: Download model ----
    print("\n--- Step 3: Downloading model ---")
    from huggingface_hub import snapshot_download

    t0 = time.time()
    model_path = snapshot_download(SMOKE_CONFIG["model_id"], cache_dir="/hf-cache")
    print(f"  Model at: {model_path} ({time.time()-t0:.0f}s)")

    # ---- Step 4: Run each condition ----
    all_results = {}
    for condition in ["A", "B", "C"]:
        diag = run_single_condition(condition)
        all_results[condition] = diag
        results_vol.commit()  # checkpoint after each condition

    # ---- Step 5: Verify ----
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)

    checks = verify_diagnostics(all_results)
    passed = 0
    failed = 0
    warned = 0

    for status, message in checks:
        print(f"  [{status}] {message}")
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        elif status == "WARN":
            warned += 1

    overall = "PASS" if failed == 0 else "FAIL"
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"SMOKE TEST {overall}")
    print(f"  {passed} passed, {failed} failed, {warned} warnings")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    # Save final report
    os.makedirs("/results/spark_smoke", exist_ok=True)
    report = {
        "overall": overall,
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "per_condition": all_results,
    }
    with open("/results/spark_smoke/smoke_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved report to /results/spark_smoke/smoke_report.json")

    results_vol.commit()
    return report


# ============================================================
# FULL ABLATION RUNNER
# ============================================================


wandb_secret = modal.Secret.from_name("wandb-secret")


@app.function(
    gpu="A100-80GB:2",
    timeout=14400,  # 4 hours max
    image=smoke_image,
    volumes={
        "/hf-cache": hf_cache,
        "/results": results_vol,
    },
    secrets=[wandb_secret],
)
def run_full_training():
    """
    Run full 3-condition ablation: 50 steps each, 2x A100-80GB, WandB logging.

    Conditions:
      A: Vanilla GRPO (baseline)
      B: CURE (entropy-only branching, vanilla loss)
      C: Spark-MCTS (3-gate branching, Lp-Reg loss, prefix stop-gradient)
    """
    import sys
    import time

    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"

    cfg = FULL_CONFIG
    results_base = "/results/spark_full"

    start = time.time()
    print("=" * 60)
    print("Spark-MCTS Full Ablation")
    print(f"Model:  {cfg['model_id']}")
    print(f"Steps:  {cfg['total_steps']} per condition")
    print(f"n1={cfg['n1']}, n2={cfg['n2']}, n_total={cfg['n_total']}")
    print(f"GPUs:   {cfg['n_gpus']}x A100-80GB")
    print(f"Logger: {cfg['logger']}")
    print("=" * 60)

    # ---- Step 1: Apply patches ----
    print("\n--- Step 1: Applying patches ---")
    apply_patches()

    with open("/root/CURE/verl/workers/actor/dp_actor.py") as f:
        dp_content = f.read()
    for marker, desc in {
        "min_p=None": "signature patch",
        "Min-p filtering for Lp-Reg (remove_padding path)": "rmpad patch",
        'loss_mode == "spark_nll"': "loss branch",
        "compute_policy_loss_spark_nll": "import",
    }.items():
        assert marker in dp_content, f"Patch verification FAILED: {desc}"
        print(f"  [OK] {desc}")

    # ---- Step 2: Prepare data ----
    print("\n--- Step 2: Preparing GSM8K data ---")
    import subprocess

    sys.path.insert(0, "/root/CURE")
    subprocess.run([sys.executable, "/root/CURE/prepare_data.py"], check=True)
    assert os.path.exists("/root/data/gsm8k/train.parquet")
    print("  GSM8K parquet files ready.")

    # ---- Step 3: Download model ----
    print("\n--- Step 3: Downloading model ---")
    from huggingface_hub import snapshot_download

    t0 = time.time()
    model_path = snapshot_download(cfg["model_id"], cache_dir="/hf-cache")
    print(f"  Model at: {model_path} ({time.time()-t0:.0f}s)")

    # ---- Step 4: Run each condition ----
    all_results = {}
    for condition in ["A", "B", "C"]:
        diag = run_single_condition(condition, config=cfg, results_base=results_base)
        all_results[condition] = diag
        results_vol.commit()

    # ---- Step 5: Verify ----
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)

    checks = verify_diagnostics(all_results)
    passed = failed = warned = 0

    for status, message in checks:
        print(f"  [{status}] {message}")
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        elif status == "WARN":
            warned += 1

    overall = "PASS" if failed == 0 else "FAIL"
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"FULL ABLATION {overall}")
    print(f"  {passed} passed, {failed} failed, {warned} warnings")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    os.makedirs(results_base, exist_ok=True)
    report = {
        "overall": overall,
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "per_condition": all_results,
    }
    report_path = os.path.join(results_base, "full_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {report_path}")

    results_vol.commit()
    return report


# ============================================================
# NLL ALPHA COEFFICIENT SWEEP
# ============================================================


@app.function(
    gpu="A100-80GB",
    timeout=3600,  # 1 hour max
    image=smoke_image,
    volumes={
        "/hf-cache": hf_cache,
        "/results": results_vol,
    },
)
def run_lpreg_sweep():
    """
    Sweep NLL alpha coefficient to find value where pg_loss > 3x protection_magnitude.

    Tests alpha [0.1, 0.3, 0.5] on condition C only (10 steps each, no validation).
    Returns sweep_report.json with selected_coef and comparison table.
    """
    import sys
    import time

    import numpy as np

    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"
    os.environ["WANDB_MODE"] = "disabled"

    results_base = "/results/spark_sweep"
    start = time.time()

    print("=" * 60)
    print("NLL Alpha Coefficient Sweep")
    print(f"Model:  {SWEEP_CONFIG['model_id']}")
    print(f"Steps:  {SWEEP_CONFIG['total_steps']} per trial")
    print(f"Alphas: [0.1, 0.3, 0.5]")
    print("=" * 60)

    # ---- Step 1: Apply patches ----
    print("\n--- Step 1: Applying patches ---")
    apply_patches()

    # ---- Step 2: Prepare data ----
    print("\n--- Step 2: Preparing GSM8K data ---")
    import subprocess

    sys.path.insert(0, "/root/CURE")
    subprocess.run([sys.executable, "/root/CURE/prepare_data.py"], check=True)
    assert os.path.exists("/root/data/gsm8k/train.parquet")
    print("  GSM8K parquet files ready.")

    # ---- Step 3: Download model ----
    print("\n--- Step 3: Downloading model ---")
    from huggingface_hub import snapshot_download

    t0 = time.time()
    model_path = snapshot_download(SWEEP_CONFIG["model_id"], cache_dir="/hf-cache")
    print(f"  Model at: {model_path} ({time.time()-t0:.0f}s)")

    # ---- Step 4: Sweep alphas ----
    alphas = [0.1, 0.3, 0.5]
    trials = []

    for alpha in alphas:
        trial_cfg = {**SWEEP_CONFIG, "spark_nll_alpha": alpha}
        trial_base = f"{results_base}/alpha_{alpha}"
        print(f"\n--- Trial: spark_nll_alpha={alpha} ---")

        diag = run_single_condition("C", config=trial_cfg, results_base=trial_base)

        # Extract loss components (decomposed: GRPO-only vs NLL protection)
        steps = diag.get("steps", [])
        # Use grpo_only_loss (GRPO without protection) if available, else abs(pg_loss)
        grpo_losses = [
            abs(s["grpo_only_loss"])
            for s in steps
            if s.get("grpo_only_loss") is not None
        ]
        if not grpo_losses:
            grpo_losses = [
                abs(s["pg_loss"]) for s in steps if s.get("pg_loss") is not None
            ]
        # NLL protection mean per step (from module-level tracking)
        protect_mags = [
            s["nll_protect_mean"]
            for s in steps
            if s.get("nll_protect_mean") is not None and s["nll_protect_mean"] > 0
        ]
        protect_counts = [
            s["nll_protect_count"]
            for s in steps
            if s.get("nll_protect_count") is not None
        ]

        mean_grpo = float(np.mean(grpo_losses)) if grpo_losses else 0.0
        mean_protect = float(np.mean(protect_mags)) if protect_mags else 0.0
        mean_prot_count = float(np.mean(protect_counts)) if protect_counts else 0.0
        # Ratio: |GRPO loss| / NLL protection — want GRPO to dominate (ratio > 3)
        ratio = mean_grpo / mean_protect if mean_protect > 0 else float("inf")

        trial_result = {
            "coef": alpha,
            "mean_grpo_loss": mean_grpo,
            "mean_protect_magnitude": mean_protect,
            "mean_protect_count": mean_prot_count,
            "grpo_to_protect_ratio": ratio,
            "protection_fires": len(protect_mags) > 0
            and any(m > 0.001 for m in protect_mags),
            "n_steps": len(steps),
        }
        trials.append(trial_result)
        print(
            f"  |GRPO|={mean_grpo:.4f}, protect={mean_protect:.6f}, ratio={ratio:.1f}, n_prot={mean_prot_count:.0f}"
        )
        results_vol.commit()

    # ---- Step 5: Select best alpha ----
    # Pick alpha where |GRPO| > 3x protection AND protection fires
    valid = [
        t for t in trials if t["protection_fires"] and t["grpo_to_protect_ratio"] > 3.0
    ]
    if valid:
        # Among valid, pick highest alpha (strongest protection that doesn't dominate)
        selected = max(valid, key=lambda t: t["coef"])
    else:
        # Fallback: pick the alpha with highest ratio (closest to meeting criterion)
        firing = [t for t in trials if t["protection_fires"]]
        if firing:
            selected = max(firing, key=lambda t: t["grpo_to_protect_ratio"])
            print(
                f"WARNING: No alpha satisfies ratio > 3.0. Picking best ratio: {selected['grpo_to_protect_ratio']:.1f}"
            )
        else:
            selected = min(trials, key=lambda t: t["coef"])
            print(
                "WARNING: No alpha has protection firing. Using lowest alpha as fallback."
            )

    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"SWEEP COMPLETE")
    print(f"  Selected alpha: {selected['coef']}")
    print(f"  Ratio: {selected['grpo_to_protect_ratio']:.1f}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    os.makedirs(results_base, exist_ok=True)
    report = {
        "selected_coef": selected["coef"],
        "selection_reason": (
            f"|GRPO|/protect ratio={selected['grpo_to_protect_ratio']:.1f} > 3.0"
            if selected in valid
            else f"best ratio={selected['grpo_to_protect_ratio']:.1f} (criterion: > 3.0)"
        ),
        "trials": trials,
        "elapsed_seconds": elapsed,
    }
    report_path = os.path.join(results_base, "sweep_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {report_path}")

    results_vol.commit()
    return report


# ============================================================
# VIABILITY RUN (A vs C, 100 steps)
# ============================================================


@app.function(
    gpu="A100-80GB:2",
    timeout=14400,  # 4 hours max
    image=smoke_image,
    volumes={
        "/hf-cache": hf_cache,
        "/results": results_vol,
    },
    secrets=[wandb_secret],
)
def run_viability(lpreg_coef: float = 0.3):
    """
    Run 100-step viability test: conditions A and C only.

    Uses sweep-winning alpha. Checks pass@1 trajectory and C-vs-A comparison.
    """
    import sys
    import time

    import numpy as np

    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"

    cfg = {**VIABILITY_CONFIG, "spark_nll_alpha": lpreg_coef}
    results_base = "/results/spark_viability"

    start = time.time()
    print("=" * 60)
    print("Spark-MCTS Viability Run (A vs C)")
    print(f"Model:  {cfg['model_id']}")
    print(f"Steps:  {cfg['total_steps']} per condition")
    print(f"NLL alpha: {lpreg_coef}")
    print(f"GPUs:   {cfg['n_gpus']}x A100-80GB")
    print(f"Logger: {cfg['logger']}")
    print("=" * 60)

    # ---- Step 1: Apply patches ----
    print("\n--- Step 1: Applying patches ---")
    apply_patches()

    with open("/root/CURE/verl/workers/actor/dp_actor.py") as f:
        dp_content = f.read()
    for marker, desc in {
        "min_p=None": "signature patch",
        "Min-p filtering for Lp-Reg (remove_padding path)": "rmpad patch",
        'loss_mode == "spark_nll"': "loss branch",
        "compute_policy_loss_spark_nll": "import",
    }.items():
        assert marker in dp_content, f"Patch verification FAILED: {desc}"
        print(f"  [OK] {desc}")

    # ---- Step 2: Prepare data ----
    print("\n--- Step 2: Preparing GSM8K data ---")
    import subprocess

    sys.path.insert(0, "/root/CURE")
    subprocess.run([sys.executable, "/root/CURE/prepare_data.py"], check=True)
    assert os.path.exists("/root/data/gsm8k/train.parquet")
    print("  GSM8K parquet files ready.")

    # ---- Step 3: Download model ----
    print("\n--- Step 3: Downloading model ---")
    from huggingface_hub import snapshot_download

    t0 = time.time()
    model_path = snapshot_download(cfg["model_id"], cache_dir="/hf-cache")
    print(f"  Model at: {model_path} ({time.time()-t0:.0f}s)")

    # ---- Step 4: Run conditions A and C ----
    all_results = {}
    for condition in ["A", "C"]:
        diag = run_single_condition(condition, config=cfg, results_base=results_base)
        all_results[condition] = diag
        results_vol.commit()

    # ---- Step 5: Verify ----
    print("\n" + "=" * 60)
    print("VIABILITY VERIFICATION REPORT")
    print("=" * 60)

    checks = verify_diagnostics(all_results, conditions=["A", "C"])

    # Additional check: pass@1 trajectory
    for cond in ["A", "C"]:
        cond_result = all_results.get(cond, {})
        p1_traj = cond_result.get("pass_at_1_trajectory", [])
        if p1_traj:
            values = [p["value"] for p in p1_traj if p.get("value") is not None]
            if values:
                checks.append(
                    (
                        "INFO",
                        f"Condition {cond}: pass@1 logged {len(values)} points "
                        f"(first={values[0]:.3f}, last={values[-1]:.3f}, "
                        f"delta={values[-1]-values[0]:+.3f})",
                    )
                )
            else:
                checks.append(
                    ("WARN", f"Condition {cond}: pass@1 trajectory has no values")
                )
        else:
            checks.append(("WARN", f"Condition {cond}: no pass@1 trajectory recorded"))

    # C vs A comparison
    a_p1 = all_results.get("A", {}).get("pass_at_1_trajectory", [])
    c_p1 = all_results.get("C", {}).get("pass_at_1_trajectory", [])
    a_final = [p["value"] for p in a_p1 if p.get("value") is not None]
    c_final = [p["value"] for p in c_p1 if p.get("value") is not None]
    if a_final and c_final:
        delta = c_final[-1] - a_final[-1]
        checks.append(
            (
                "INFO",
                f"C vs A final pass@1: C={c_final[-1]:.3f}, A={a_final[-1]:.3f}, "
                f"delta={delta:+.3f} ({'C wins' if delta > 0 else 'A wins' if delta < 0 else 'tie'})",
            )
        )

    passed = failed = warned = 0
    for status, message in checks:
        print(f"  [{status}] {message}")
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        elif status == "WARN":
            warned += 1

    overall = "PASS" if failed == 0 else "FAIL"
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"VIABILITY RUN {overall}")
    print(f"  {passed} passed, {failed} failed, {warned} warnings")
    print(f"  Lp-Reg coef: {lpreg_coef}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    os.makedirs(results_base, exist_ok=True)
    report = {
        "overall": overall,
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "lpreg_coef": lpreg_coef,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "per_condition": all_results,
    }
    report_path = os.path.join(results_base, "viability_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {report_path}")

    results_vol.commit()
    return report


# ============================================================
# MATH SMOKE TEST (5 steps, condition C only, MATH Level 3-4)
# ============================================================


@app.function(
    image=smoke_image,
    gpu=modal.gpu.A100(count=2, size="80GB"),
    timeout=3600,
    volumes={"/results": results_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_math_smoke():
    """
    Small smoke test on MATH Level 3-4 with n2=4.

    Validates: data loading, reward parsing (boxed format), V(R) gate fires less,
    no crashes with n_total=10. Runs only condition C for 5 steps.
    """
    import sys
    import time

    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"

    cfg = {**MATH_SMOKE_CONFIG}
    results_base = "/results/math_smoke"

    start = time.time()
    print("=" * 60)
    print("MATH L3-4 Smoke Test (n2=4, condition C)")
    print(f"Model:  {cfg['model_id']}")
    print(f"Steps:  {cfg['total_steps']}")
    print(f"n1={cfg['n1']}, n2={cfg['n2']}, n_total={cfg['n_total']}")
    print(f"Dataset: MATH Level 3-4")
    print(f"GPUs:   {cfg['n_gpus']}x A100-80GB")
    print("=" * 60)

    # ---- Step 1: Apply patches ----
    print("\n--- Step 1: Applying patches ---")
    apply_patches()

    with open("/root/CURE/verl/workers/actor/dp_actor.py") as f:
        dp_content = f.read()
    for marker, desc in {
        "min_p=None": "signature patch",
        "Min-p filtering for Lp-Reg (remove_padding path)": "rmpad patch",
        'loss_mode == "spark_nll"': "loss branch",
        "compute_policy_loss_spark_nll": "import",
    }.items():
        assert marker in dp_content, f"Patch verification FAILED: {desc}"
        print(f"  [OK] {desc}")

    # ---- Step 2: Prepare MATH data ----
    print("\n--- Step 2: Preparing MATH Level 3-4 data ---")
    import subprocess

    sys.path.insert(0, "/root/CURE")

    # Write and run MATH prep script
    math_prep_path = "/root/CURE/prepare_math_data.py"
    with open(math_prep_path, "w") as f:
        f.write(MATH_DATA_PREP_SCRIPT)
    subprocess.run([sys.executable, math_prep_path], check=True)
    assert os.path.exists(
        "/root/data/math_l34/train.parquet"
    ), "MATH train.parquet missing"
    assert os.path.exists(
        "/root/data/math_l34/test.parquet"
    ), "MATH test.parquet missing"
    print("  MATH L3-4 parquet files ready.")

    # ---- Step 3: Download model ----
    print("\n--- Step 3: Downloading model ---")
    from huggingface_hub import snapshot_download

    t0 = time.time()
    model_path = snapshot_download(cfg["model_id"], cache_dir="/hf-cache")
    print(f"  Model at: {model_path} ({time.time()-t0:.0f}s)")

    # ---- Step 4: Run condition C only ----
    diag = run_single_condition("C", config=cfg, results_base=results_base)

    # ---- Step 5: Quick validation ----
    print("\n" + "=" * 60)
    print("MATH SMOKE TEST REPORT")
    print("=" * 60)

    checks = []
    steps = diag.get("steps", diag.get("step_diagnostics", []))
    if steps:
        losses = [
            s.get("pg_loss", s.get("loss"))
            for s in steps
            if s.get("pg_loss", s.get("loss")) is not None
        ]
        if losses and all(abs(l) < 100 for l in losses):
            checks.append(
                (
                    "PASS",
                    f"Loss finite across {len(losses)} steps (mean={sum(losses)/len(losses):.4f})",
                )
            )
        else:
            checks.append(("FAIL", f"Loss not finite: {losses}"))

        # Check V(R) gate fires less than 80%
        vr_total = sum(
            s.get("vr_gate", {}).get("vr_total_forks", s.get("vr_total_forks", 0))
            for s in steps
        )
        vr_aleatoric = sum(
            s.get("vr_gate", {}).get("vr_zero_forks", s.get("vr_aleatoric_forks", 0))
            for s in steps
        )
        if vr_total > 0:
            vr_rate = vr_aleatoric / vr_total
            checks.append(
                (
                    "INFO",
                    f"V(R) gate: {vr_aleatoric}/{vr_total} aleatoric ({vr_rate:.0%})",
                )
            )
        else:
            checks.append(
                ("INFO", "V(R) gate stats not available (checking diagnostics)")
            )

        # Spark position diagnostic
        pos_means = [
            s.get("spark_pos_frac_mean")
            for s in steps
            if s.get("spark_pos_frac_mean") is not None
        ]
        pos_lates = [
            s.get("spark_pos_late_frac")
            for s in steps
            if s.get("spark_pos_late_frac") is not None
        ]
        if pos_means:
            avg_pos = sum(pos_means) / len(pos_means)
            avg_late = sum(pos_lates) / len(pos_lates) if pos_lates else 0
            checks.append(
                (
                    "INFO",
                    f"Spark position: mean_frac={avg_pos:.3f}, late(>0.7)={avg_late:.0%}",
                )
            )

        # Check NLL protection
        nll_vals = [
            s.get("nll_protect_mean", 0)
            for s in steps
            if s.get("nll_protect_mean", 0) > 0
        ]
        if nll_vals:
            checks.append(
                (
                    "PASS",
                    f"NLL protection fires (mean={sum(nll_vals)/len(nll_vals):.3f}, n={len(nll_vals)})",
                )
            )

        # Check spark fraction
        spark_fracs = diag.get("spark_fractions", [])
        if spark_fracs:
            mean_sf = sum(spark_fracs) / len(spark_fracs)
            checks.append(("INFO", f"Spark fraction: mean={mean_sf:.1%}"))

        # Check pass@1 from final validation
        p1_traj = diag.get("pass_at_1_trajectory", [])
        if p1_traj:
            values = [p["value"] for p in p1_traj if p.get("value") is not None]
            if values:
                checks.append(("INFO", f"pass@1: {values[-1]:.3f} (from validation)"))
        final_val = diag.get("final_val_metrics", {})
        for vk, vv in final_val.items():
            if "reward" in vk.lower() and "mean" in vk.lower():
                checks.append(("INFO", f"Final val metric: {vk} = {vv}"))
    else:
        checks.append(("FAIL", "No step diagnostics recorded"))

    passed = failed = warned = 0
    for status, message in checks:
        print(f"  [{status}] {message}")
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        elif status == "WARN":
            warned += 1

    overall = "PASS" if failed == 0 and passed > 0 else "FAIL"
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"MATH SMOKE TEST {overall}")
    print(f"  {passed} passed, {failed} failed, {warned} warnings")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    os.makedirs(results_base, exist_ok=True)
    report = {
        "overall": overall,
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "per_condition": {"C": diag},
    }
    report_path = os.path.join(results_base, "math_smoke_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {report_path}")

    results_vol.commit()
    return report


# ============================================================
# MATH VIABILITY RUN (100 steps, A vs C, MATH Level 3-4, n2=4)
# ============================================================


@app.function(
    image=smoke_image,
    gpu=modal.gpu.A100(count=2, size="80GB"),
    timeout=14400,
    volumes={"/results": results_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_math_viability(lpreg_coef: float = 0.3):
    """
    100-step viability run on MATH Level 3-4 with n2=4.
    A vs C comparison.
    """
    import sys
    import time

    import numpy as np

    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"

    cfg = {**MATH_VIABILITY_CONFIG, "spark_nll_alpha": lpreg_coef}
    results_base = "/results/math_viability"

    start = time.time()
    print("=" * 60)
    print("MATH L3-4 Viability Run (A vs C, n2=4)")
    print(f"Model:  {cfg['model_id']}")
    print(f"Steps:  {cfg['total_steps']} per condition")
    print(f"n1={cfg['n1']}, n2={cfg['n2']}, n_total={cfg['n_total']}")
    print(f"NLL alpha: {lpreg_coef}")
    print(f"GPUs:   {cfg['n_gpus']}x A100-80GB")
    print(f"Logger: {cfg['logger']}")
    print("=" * 60)

    # ---- Step 1: Apply patches ----
    print("\n--- Step 1: Applying patches ---")
    apply_patches()

    with open("/root/CURE/verl/workers/actor/dp_actor.py") as f:
        dp_content = f.read()
    for marker, desc in {
        "min_p=None": "signature patch",
        "Min-p filtering for Lp-Reg (remove_padding path)": "rmpad patch",
        'loss_mode == "spark_nll"': "loss branch",
        "compute_policy_loss_spark_nll": "import",
    }.items():
        assert marker in dp_content, f"Patch verification FAILED: {desc}"
        print(f"  [OK] {desc}")

    # ---- Step 2: Prepare MATH data ----
    print("\n--- Step 2: Preparing MATH Level 3-4 data ---")
    import subprocess

    sys.path.insert(0, "/root/CURE")

    math_prep_path = "/root/CURE/prepare_math_data.py"
    with open(math_prep_path, "w") as f:
        f.write(MATH_DATA_PREP_SCRIPT)
    subprocess.run([sys.executable, math_prep_path], check=True)
    assert os.path.exists(
        "/root/data/math_l34/train.parquet"
    ), "MATH train.parquet missing"
    assert os.path.exists(
        "/root/data/math_l34/test.parquet"
    ), "MATH test.parquet missing"
    print("  MATH L3-4 parquet files ready.")

    # ---- Step 3: Download model ----
    print("\n--- Step 3: Downloading model ---")
    from huggingface_hub import snapshot_download

    t0 = time.time()
    model_path = snapshot_download(cfg["model_id"], cache_dir="/hf-cache")
    print(f"  Model at: {model_path} ({time.time()-t0:.0f}s)")

    # ---- Step 4: Run conditions A and C ----
    all_results = {}
    for condition in ["A", "C"]:
        diag = run_single_condition(condition, config=cfg, results_base=results_base)
        all_results[condition] = diag
        results_vol.commit()

    # ---- Step 5: Verify ----
    print("\n" + "=" * 60)
    print("MATH VIABILITY REPORT")
    print("=" * 60)

    checks = verify_diagnostics(all_results, conditions=["A", "C"])

    # pass@1 trajectory
    for cond in ["A", "C"]:
        cond_result = all_results.get(cond, {})
        p1_traj = cond_result.get("pass_at_1_trajectory", [])
        if p1_traj:
            values = [p["value"] for p in p1_traj if p.get("value") is not None]
            if values:
                checks.append(
                    (
                        "INFO",
                        f"Condition {cond}: pass@1 logged {len(values)} points "
                        f"(first={values[0]:.3f}, last={values[-1]:.3f}, delta={values[-1]-values[0]:+.3f})",
                    )
                )
            else:
                checks.append(
                    ("WARN", f"Condition {cond}: pass@1 trajectory has no values")
                )
        else:
            checks.append(("WARN", f"Condition {cond}: no pass@1 trajectory recorded"))

    # C vs A comparison
    a_p1 = all_results.get("A", {}).get("pass_at_1_trajectory", [])
    c_p1 = all_results.get("C", {}).get("pass_at_1_trajectory", [])
    a_final = [p["value"] for p in a_p1 if p.get("value") is not None]
    c_final = [p["value"] for p in c_p1 if p.get("value") is not None]
    if a_final and c_final:
        delta = c_final[-1] - a_final[-1]
        checks.append(
            (
                "INFO",
                f"C vs A final pass@1: C={c_final[-1]:.3f}, A={a_final[-1]:.3f}, "
                f"delta={delta:+.3f} ({'C wins' if delta > 0 else 'A wins' if delta < 0 else 'tie'})",
            )
        )

    # V(R) gate rate for C
    c_diag = all_results.get("C", {})
    c_steps = c_diag.get("steps", c_diag.get("step_diagnostics", []))
    vr_total = sum(
        s.get("vr_gate", {}).get("vr_total_forks", s.get("vr_total_forks", 0))
        for s in c_steps
    )
    vr_aleatoric = sum(
        s.get("vr_gate", {}).get("vr_zero_forks", s.get("vr_aleatoric_forks", 0))
        for s in c_steps
    )
    if vr_total > 0:
        checks.append(
            (
                "INFO",
                f"V(R) gate: {vr_aleatoric}/{vr_total} aleatoric ({vr_aleatoric/vr_total:.0%})",
            )
        )

    passed = failed = warned = 0
    for status, message in checks:
        print(f"  [{status}] {message}")
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        elif status == "WARN":
            warned += 1

    overall = "PASS" if failed == 0 else "FAIL"
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"MATH VIABILITY RUN {overall}")
    print(f"  {passed} passed, {failed} failed, {warned} warnings")
    print(f"  NLL alpha: {lpreg_coef}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    os.makedirs(results_base, exist_ok=True)
    report = {
        "overall": overall,
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "lpreg_coef": lpreg_coef,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "per_condition": all_results,
    }
    report_path = os.path.join(results_base, "math_viability_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {report_path}")

    results_vol.commit()
    return report


# ============================================================
# ROUTING SMOKE TEST (Condition D, 5 steps)
# ============================================================


@app.function(
    gpu="A100-80GB",
    timeout=7200,
    image=smoke_image,
    volumes={
        "/hf-cache": hf_cache,
        "/results": results_vol,
    },
)
def run_routing_smoke():
    """
    Smoke test for Condition D: gradient routing without branching.

    Checks:
      1. Loss finite (no NaN/Inf)
      2. Routing zeroes >0% tokens
      3. high_ent_frac between 2-30%
      4. NLL protection fires
      5. No OOM at n=16
    """
    import sys
    import time

    import numpy as np

    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"
    os.environ["WANDB_MODE"] = "disabled"

    cfg = {**ROUTING_SMOKE_CONFIG}
    results_base = "/results/routing_smoke"

    start = time.time()
    print("=" * 60)
    print("Condition D: Gradient Routing Smoke Test")
    print(f"Model:  {cfg['model_id']}")
    print(f"Steps:  {cfg['total_steps']}")
    print(f"n:      {cfg['n_total']} (flat rollouts)")
    print(f"tau_h:  {cfg['routing_tau_h']}")
    print("=" * 60)

    # ---- Step 1: Apply patches ----
    print("\n--- Step 1: Applying patches ---")
    apply_patches()

    with open("/root/CURE/verl/workers/actor/dp_actor.py") as f:
        dp_content = f.read()
    for marker, desc in {
        "min_p=None": "signature patch",
        'loss_mode == "spark_nll"': "loss branch",
        "compute_policy_loss_spark_nll": "import",
    }.items():
        assert marker in dp_content, f"Patch verification FAILED: {desc}"
        print(f"  [OK] {desc}")

    # ---- Step 2: Prepare data ----
    print("\n--- Step 2: Preparing GSM8K data ---")
    import subprocess

    sys.path.insert(0, "/root/CURE")
    subprocess.run([sys.executable, "/root/CURE/prepare_data.py"], check=True)
    assert os.path.exists("/root/data/gsm8k/train.parquet")
    print("  GSM8K parquet files ready.")

    # ---- Step 3: Download model ----
    print("\n--- Step 3: Downloading model ---")
    from huggingface_hub import snapshot_download

    t0 = time.time()
    model_path = snapshot_download(cfg["model_id"], cache_dir="/hf-cache")
    print(f"  Model at: {model_path} ({time.time()-t0:.0f}s)")

    # ---- Step 4: Run condition D ----
    diag = run_single_condition("D", config=cfg, results_base=results_base)
    results_vol.commit()

    # ---- Step 5: Verify routing-specific checks ----
    print("\n" + "=" * 60)
    print("ROUTING SMOKE VERIFICATION")
    print("=" * 60)

    checks = []

    # Check 1: Loss finite
    if diag.get("nan_detected", False):
        checks.append(("FAIL", "Condition D: NaN/Inf detected in loss"))
    else:
        loss_vals = [
            s.get("pg_loss")
            for s in diag.get("steps", [])
            if s.get("pg_loss") is not None
        ]
        if loss_vals:
            checks.append(
                (
                    "PASS",
                    f"Condition D: loss finite across {len(loss_vals)} steps (mean={np.mean(loss_vals):.4f})",
                )
            )
        else:
            checks.append(("WARN", "Condition D: no pg_loss values recorded"))

    # Check 2: Routing zeroes >0% tokens
    routing_traj = diag.get("routing_trajectory", [])
    if routing_traj:
        frac_zeroed = [r["routing_frac_zeroed"] for r in routing_traj]
        any_zeroed = any(f > 0 for f in frac_zeroed)
        mean_frac = np.mean(frac_zeroed)
        checks.append(
            (
                "PASS" if any_zeroed else "FAIL",
                f"Routing zeroes tokens: mean={mean_frac:.1%}, any>0={any_zeroed}",
            )
        )
    else:
        checks.append(("FAIL", "No routing trajectory recorded"))

    # Check 3: high_ent_frac between 2-30%
    if routing_traj:
        hef_vals = [r["routing_high_ent_frac"] for r in routing_traj]
        mean_hef = np.mean(hef_vals)
        in_range = 0.02 <= mean_hef <= 0.30
        checks.append(
            (
                "PASS" if in_range else "WARN",
                f"high_ent_frac={mean_hef:.1%} (expect 2-30%)",
            )
        )

    # Check 4: NLL protection fires
    nll_vals = [
        s.get("nll_protect_mean", 0)
        for s in diag.get("steps", [])
        if s.get("nll_protect_mean", 0) > 0
    ]
    if nll_vals:
        checks.append(
            (
                "PASS",
                f"NLL protection fires (mean={np.mean(nll_vals):.4f}, n={len(nll_vals)})",
            )
        )
    else:
        checks.append(
            ("WARN", "NLL protection did not fire (may be expected for smoke)")
        )

    passed = failed = warned = 0
    for status, message in checks:
        print(f"  [{status}] {message}")
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        elif status == "WARN":
            warned += 1

    overall = "PASS" if failed == 0 else "FAIL"
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"ROUTING SMOKE {overall}")
    print(f"  {passed} passed, {failed} failed, {warned} warnings")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    os.makedirs(results_base, exist_ok=True)
    report = {
        "overall": overall,
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "per_condition": {"D": diag},
    }
    report_path = os.path.join(results_base, "routing_smoke_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {report_path}")

    results_vol.commit()
    return report


# ============================================================
# ROUTING VIABILITY RUN (A vs D, 100 steps)
# ============================================================


@app.function(
    gpu="A100-80GB:2",
    timeout=14400,
    image=smoke_image,
    volumes={
        "/hf-cache": hf_cache,
        "/results": results_vol,
    },
    secrets=[wandb_secret],
)
def run_routing_viability(lpreg_coef: float = 0.3):
    """
    100-step viability test: A vs D.

    Paper claim if D > A: "Token-level gradient routing via entropy-based masking
    improves exploration preservation in standard GRPO, independent of tree search."
    """
    import sys
    import time

    import numpy as np

    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"

    cfg = {**ROUTING_VIABILITY_CONFIG, "spark_nll_alpha": lpreg_coef}
    results_base = "/results/routing_viability"

    start = time.time()
    print("=" * 60)
    print("Routing Viability Run (A vs D)")
    print(f"Model:  {cfg['model_id']}")
    print(f"Steps:  {cfg['total_steps']} per condition")
    print(f"n:      {cfg['n_total']} (flat rollouts)")
    print(f"tau_h:  {cfg['routing_tau_h']}")
    print(f"NLL alpha: {lpreg_coef}")
    print(f"GPUs:   {cfg['n_gpus']}x A100-80GB")
    print(f"Logger: {cfg['logger']}")
    print("=" * 60)

    # ---- Step 1: Apply patches ----
    print("\n--- Step 1: Applying patches ---")
    apply_patches()

    with open("/root/CURE/verl/workers/actor/dp_actor.py") as f:
        dp_content = f.read()
    for marker, desc in {
        "min_p=None": "signature patch",
        'loss_mode == "spark_nll"': "loss branch",
        "compute_policy_loss_spark_nll": "import",
    }.items():
        assert marker in dp_content, f"Patch verification FAILED: {desc}"
        print(f"  [OK] {desc}")

    # ---- Step 2: Prepare data ----
    print("\n--- Step 2: Preparing GSM8K data ---")
    import subprocess

    sys.path.insert(0, "/root/CURE")
    subprocess.run([sys.executable, "/root/CURE/prepare_data.py"], check=True)
    assert os.path.exists("/root/data/gsm8k/train.parquet")
    print("  GSM8K parquet files ready.")

    # ---- Step 3: Download model ----
    print("\n--- Step 3: Downloading model ---")
    from huggingface_hub import snapshot_download

    t0 = time.time()
    model_path = snapshot_download(cfg["model_id"], cache_dir="/hf-cache")
    print(f"  Model at: {model_path} ({time.time()-t0:.0f}s)")

    # ---- Step 4: Run conditions A and D ----
    all_results = {}
    for condition in ["A", "D"]:
        diag = run_single_condition(condition, config=cfg, results_base=results_base)
        all_results[condition] = diag
        results_vol.commit()

    # ---- Step 5: Verify ----
    print("\n" + "=" * 60)
    print("ROUTING VIABILITY VERIFICATION")
    print("=" * 60)

    checks = []

    # Loss finite for both conditions
    for cond in ["A", "D"]:
        result = all_results.get(cond, {})
        if result.get("nan_detected", False):
            checks.append(("FAIL", f"Condition {cond}: NaN/Inf detected"))
        else:
            loss_vals = [
                s.get("pg_loss")
                for s in result.get("steps", [])
                if s.get("pg_loss") is not None
            ]
            if loss_vals:
                checks.append(
                    (
                        "PASS",
                        f"Condition {cond}: loss finite ({len(loss_vals)} steps, mean={np.mean(loss_vals):.4f})",
                    )
                )
            else:
                checks.append(("WARN", f"Condition {cond}: no pg_loss values"))

    # Routing diagnostics for D
    d_result = all_results.get("D", {})
    routing_traj = d_result.get("routing_trajectory", [])
    if routing_traj:
        frac_zeroed = [r["routing_frac_zeroed"] for r in routing_traj]
        hef_vals = [r["routing_high_ent_frac"] for r in routing_traj]
        checks.append(
            (
                "INFO",
                f"D routing: mean zeroed={np.mean(frac_zeroed):.1%}, high_ent_frac={np.mean(hef_vals):.1%}",
            )
        )

    # NLL protection for D
    nll_vals = [
        s.get("nll_protect_mean", 0)
        for s in d_result.get("steps", [])
        if s.get("nll_protect_mean", 0) > 0
    ]
    if nll_vals:
        checks.append(
            (
                "PASS",
                f"D NLL protection fires (mean={np.mean(nll_vals):.4f}, n={len(nll_vals)})",
            )
        )

    # pass@1 trajectory
    for cond in ["A", "D"]:
        cond_result = all_results.get(cond, {})
        p1_traj = cond_result.get("pass_at_1_trajectory", [])
        if p1_traj:
            values = [p["value"] for p in p1_traj if p.get("value") is not None]
            if values:
                checks.append(
                    (
                        "INFO",
                        f"Condition {cond}: pass@1 ({len(values)} pts, "
                        f"first={values[0]:.3f}, last={values[-1]:.3f}, delta={values[-1]-values[0]:+.3f})",
                    )
                )
            else:
                checks.append(
                    ("WARN", f"Condition {cond}: pass@1 trajectory has no values")
                )
        else:
            checks.append(("WARN", f"Condition {cond}: no pass@1 trajectory recorded"))

    # D vs A comparison
    a_p1 = all_results.get("A", {}).get("pass_at_1_trajectory", [])
    d_p1 = all_results.get("D", {}).get("pass_at_1_trajectory", [])
    a_final = [p["value"] for p in a_p1 if p.get("value") is not None]
    d_final = [p["value"] for p in d_p1 if p.get("value") is not None]
    if a_final and d_final:
        delta = d_final[-1] - a_final[-1]
        checks.append(
            (
                "INFO",
                f"D vs A final pass@1: D={d_final[-1]:.3f}, A={a_final[-1]:.3f}, "
                f"delta={delta:+.3f} ({'D wins' if delta > 0 else 'A wins' if delta < 0 else 'tie'})",
            )
        )

    passed = failed = warned = 0
    for status, message in checks:
        print(f"  [{status}] {message}")
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        elif status == "WARN":
            warned += 1

    overall = "PASS" if failed == 0 else "FAIL"
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"ROUTING VIABILITY {overall}")
    print(f"  {passed} passed, {failed} failed, {warned} warnings")
    print(f"  NLL alpha: {lpreg_coef}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    os.makedirs(results_base, exist_ok=True)
    report = {
        "overall": overall,
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "lpreg_coef": lpreg_coef,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "per_condition": all_results,
    }
    report_path = os.path.join(results_base, "routing_viability_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {report_path}")

    results_vol.commit()
    return report


# ============================================================
# TRIGGER SIGNAL DIAGNOSTIC (Run 11, forward pass only)
# ============================================================


@app.function(
    gpu="A100-80GB",
    timeout=3600,
    image=smoke_image,
    volumes={
        "/hf-cache": hf_cache,
        "/results": results_vol,
    },
)
def run_trigger_diagnostic():
    """
    Forward-pass diagnostic: which trigger signal best predicts V(R)>0?

    Compares three position-selection strategies:
      H (Entropy): top-5 entropy positions in first 200 generated tokens
      S (Structural): first-5 step boundaries (\\n\\n, Step N:, Therefore/Thus/Hence)
      R (Random): 5 random positions (control)

    For each position, branch N=4 rollouts and score with terminal verifier.
    No training — pure inference.
    """
    import math as _math
    import re
    import sys
    import time

    import numpy as np

    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"

    cfg = {**TRIGGER_DIAG_CONFIG}
    results_base = "/results/trigger_diagnostic"
    os.makedirs(results_base, exist_ok=True)

    # Local scoring function (module-level custom_compute_score is inside SPARK_TRAINER_CODE string)
    def _score(solution_str, ground_truth):
        """Score a completion against ground truth. Returns 0.0 or 1.0."""
        import re as _re

        # Extract answer: try \boxed{} first, then ####, then last number
        text = solution_str
        m = _re.search(r"\\boxed\{([^}]+)\}", text)
        if m:
            answer = m.group(1).strip().replace(",", "")
        else:
            m = _re.search(r"####\s*(\-?[0-9\.\,]+)", text)
            if m:
                answer = m.group(1).strip().replace(",", "")
            else:
                nums = _re.findall(r"\-?[0-9]+(?:\.[0-9]+)?", text)
                answer = nums[-1] if nums else None
        if answer is None:
            return 0.0
        try:
            if float(answer) == float(ground_truth):
                return 1.0
        except (ValueError, TypeError):
            pass
        if answer.strip() == str(ground_truth).strip():
            return 1.0
        return 0.0

    start = time.time()
    print("=" * 60)
    print("Run 11: Trigger Signal Diagnostic")
    print(f"Model:      {cfg['model_id']}")
    print(f"Problems:   {cfg['n_problems']}")
    print(f"Branches:   {cfg['n_branches']} per position")
    print(f"Positions:  {cfg['k_positions']} per set (H/S/R)")
    print(f"Max probe:  first {cfg['max_probe_pos']} tokens")
    print(f"Temperature: {cfg['temperature']}")
    print("=" * 60)

    # ---- Step 1: Download model ----
    print("\n--- Step 1: Downloading model ---")
    from huggingface_hub import snapshot_download

    t0 = time.time()
    model_path = snapshot_download(cfg["model_id"], cache_dir="/hf-cache")
    print(f"  Model at: {model_path} ({time.time()-t0:.0f}s)")

    # ---- Step 2: Load model with vLLM ----
    print("\n--- Step 2: Loading model with vLLM ---")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    print(f"  vLLM loaded. Vocab size: {tokenizer.vocab_size}")

    # ---- Step 3: Load MATH L3-4 problems ----
    print("\n--- Step 3: Loading MATH L3-4 dataset ---")
    import datasets

    ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    # Filter to level 3 for ~30-50% base accuracy
    level3 = [row for row in ds if row.get("level", 0) == 3]
    print(f"  MATH-500 Level 3: {len(level3)} problems")

    if len(level3) < cfg["n_problems"]:
        # Fall back to level 3+4
        level3 = [row for row in ds if row.get("level", 0) in (3, 4)]
        print(f"  Extended to Level 3-4: {len(level3)} problems")

    rng = np.random.RandomState(cfg["seed"])
    indices = rng.choice(
        len(level3), size=min(cfg["n_problems"], len(level3)), replace=False
    )
    problems = [level3[int(i)] for i in sorted(indices)]
    print(f"  Selected {len(problems)} problems")

    instruction = "Let's think step by step and put your final answer within \\boxed{}."

    # Format prompts using chat template
    prompts = []
    ground_truths = []
    for p in problems:
        content = p["problem"] + " " + instruction
        messages = [{"role": "user", "content": content}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)
        answer = p.get("answer") or ""
        ground_truths.append(answer)

    # ---- Step 4: Generate greedy rollouts ----
    print("\n--- Step 4: Generating greedy rollouts ---")
    greedy_params = SamplingParams(
        temperature=0,
        max_tokens=cfg["max_response_tokens"],
        logprobs=20,  # vLLM 0.8.4 max is 20; sufficient for entropy ranking at 1.5B
    )
    t0 = time.time()
    greedy_outputs = llm.generate(prompts, greedy_params)
    print(f"  Generated {len(greedy_outputs)} greedy rollouts ({time.time()-t0:.1f}s)")

    # Check base accuracy
    base_correct = 0
    for i, out in enumerate(greedy_outputs):
        score = _score(out.outputs[0].text, ground_truths[i])
        if score > 0:
            base_correct += 1
    base_acc = base_correct / len(greedy_outputs)
    print(
        f"  Base greedy accuracy: {base_correct}/{len(greedy_outputs)} = {base_acc:.1%}"
    )

    # ---- Step 5: Identify positions for each problem ----
    print("\n--- Step 5: Identifying H/S/R positions ---")
    all_problem_data = []

    for prob_idx, out in enumerate(greedy_outputs):
        gen_output = out.outputs[0]
        token_ids = list(gen_output.token_ids)
        logprobs_list = gen_output.logprobs  # list of dicts per token
        n_tokens = len(token_ids)
        probe_limit = min(cfg["max_probe_pos"], n_tokens)

        if probe_limit < 5:
            print(f"  Problem {prob_idx}: only {n_tokens} tokens, skipping")
            continue

        # Compute entropy per token
        entropies = _compute_entropy_from_logprobs(logprobs_list[:probe_limit])

        # Set H: top-k by entropy
        ent_arr = np.array(entropies)
        # Exclude position 0 (immediately after prompt)
        ent_arr_masked = ent_arr.copy()
        ent_arr_masked[0] = -1.0
        h_indices = np.argsort(ent_arr_masked)[::-1][: cfg["k_positions"]]
        h_positions = sorted(h_indices.tolist())

        # Set S: structural boundaries
        s_positions = _find_structural_positions(
            token_ids, tokenizer, max_pos=probe_limit
        )

        # Set R: random positions (excluding 0)
        r_rng = np.random.RandomState(cfg["seed"] + prob_idx)
        r_candidates = list(range(1, probe_limit))
        if len(r_candidates) >= cfg["k_positions"]:
            r_positions = sorted(
                r_rng.choice(
                    r_candidates, size=cfg["k_positions"], replace=False
                ).tolist()
            )
        else:
            r_positions = r_candidates

        # Score greedy output
        greedy_score = _score(gen_output.text, ground_truths[prob_idx])

        prob_data = {
            "idx": prob_idx,
            "problem": problems[prob_idx]["problem"][:200],  # truncate for JSON
            "ground_truth": ground_truths[prob_idx],
            "greedy_correct": greedy_score > 0,
            "n_generated_tokens": n_tokens,
            "positions": {"H": [], "S": [], "R": []},
            "_h_pos": h_positions,
            "_s_pos": s_positions,
            "_r_pos": r_positions,
            "_entropies": entropies,
            "_token_ids": token_ids,
        }
        all_problem_data.append(prob_data)

        print(
            f"  Problem {prob_idx}: {n_tokens} tokens, "
            f"H={len(h_positions)}, S={len(s_positions)}, R={len(r_positions)}, "
            f"greedy={'correct' if greedy_score > 0 else 'wrong'}"
        )

    # ---- Step 6: Branch from each position ----
    print(
        f"\n--- Step 6: Branching from {sum(len(d['_h_pos'])+len(d['_s_pos'])+len(d['_r_pos']) for d in all_problem_data)} positions ---"
    )

    # Build all branch prompts
    branch_tasks = []  # (prob_data_idx, set_name, pos, prompt_text)
    for di, pdata in enumerate(all_problem_data):
        prob_prompt = prompts[pdata["idx"]]
        token_ids = pdata["_token_ids"]

        for set_name, positions in [
            ("H", pdata["_h_pos"]),
            ("S", pdata["_s_pos"]),
            ("R", pdata["_r_pos"]),
        ]:
            for pos in positions:
                # Build prefix: original prompt + first `pos` generated tokens
                prefix_token_ids = token_ids[:pos]
                prefix_text = tokenizer.decode(prefix_token_ids)
                full_prefix = prob_prompt + prefix_text
                remaining = cfg["max_response_tokens"] - pos
                branch_tasks.append(
                    {
                        "di": di,
                        "set": set_name,
                        "pos": pos,
                        "prompt": full_prefix,
                        "max_tokens": max(remaining, 64),
                    }
                )

    if not branch_tasks:
        print("  ERROR: No branch tasks generated!")
        return {"error": "no branch tasks"}

    print(
        f"  Total branch prompts: {len(branch_tasks)} × {cfg['n_branches']} = {len(branch_tasks) * cfg['n_branches']} completions"
    )

    branch_params = SamplingParams(
        temperature=cfg["temperature"],
        max_tokens=cfg["max_response_tokens"],  # use uniform max for batching
        n=cfg["n_branches"],
        top_p=1.0,
    )

    t0 = time.time()
    branch_outputs = llm.generate(
        [bt["prompt"] for bt in branch_tasks],
        branch_params,
    )
    print(f"  Branching complete ({time.time()-t0:.1f}s)")

    # ---- Step 7: Score and compute V(R) ----
    print("\n--- Step 7: Scoring and computing V(R) ---")
    for task_idx, (bt, bout) in enumerate(zip(branch_tasks, branch_outputs)):
        pdata = all_problem_data[bt["di"]]
        gt = pdata["ground_truth"]
        rewards = []
        for completion in bout.outputs:
            score = _score(completion.text, gt)
            rewards.append(score)

        vr = float(np.var(rewards))
        entropy_at_pos = (
            pdata["_entropies"][bt["pos"]]
            if bt["pos"] < len(pdata["_entropies"])
            else 0.0
        )

        # Determine marker type for S positions
        marker = None
        if bt["set"] == "S":
            tok_text = tokenizer.decode([pdata["_token_ids"][bt["pos"]]])
            if "\n" in tok_text:
                marker = "\\n\\n"
            elif re.search(r"[Ss]tep", tok_text):
                marker = "Step N:"
            else:
                marker = "discourse"

        pos_entry = {
            "pos": bt["pos"],
            "pos_frac": bt["pos"] / pdata["n_generated_tokens"],
            "entropy": round(entropy_at_pos, 4),
            "vr": round(vr, 4),
            "rewards": rewards,
            "mean_reward": round(float(np.mean(rewards)), 4),
        }
        if marker:
            pos_entry["marker"] = marker

        pdata["positions"][bt["set"]].append(pos_entry)

    # ---- Step 8: Aggregate results ----
    print("\n--- Step 8: Aggregating results ---")
    summary = {}
    for set_name in ["H", "S", "R"]:
        all_vr = []
        all_ent = []
        all_pos_frac = []
        for pdata in all_problem_data:
            for pe in pdata["positions"][set_name]:
                all_vr.append(pe["vr"])
                all_ent.append(pe["entropy"])
                all_pos_frac.append(pe["pos_frac"])

        n = len(all_vr)
        if n > 0:
            vr_gt0 = sum(1 for v in all_vr if v > 0) / n
            mean_vr = float(np.mean(all_vr))
            mean_ent = float(np.mean(all_ent))
            mean_pos = float(np.mean(all_pos_frac))
        else:
            vr_gt0 = mean_vr = mean_ent = mean_pos = 0.0

        summary[set_name] = {
            "vr_gt0_rate": round(vr_gt0, 4),
            "mean_vr": round(mean_vr, 4),
            "mean_entropy": round(mean_ent, 4),
            "mean_pos_frac": round(mean_pos, 4),
            "n_positions": n,
        }

    # ---- Step 9: Report ----
    print("\n" + "=" * 60)
    print("TRIGGER SIGNAL DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(
        f"  Base greedy accuracy: {base_acc:.1%} ({base_correct}/{len(greedy_outputs)})"
    )
    print(f"  Problems analyzed: {len(all_problem_data)}")
    print()
    print(
        f"  {'Set':<6} {'V(R)>0 rate':>12} {'Mean V(R)':>10} {'Mean H':>8} {'Mean pos':>10} {'N':>4}"
    )
    print(f"  {'---':<6} {'-'*12:>12} {'-'*10:>10} {'-'*8:>8} {'-'*10:>10} {'-'*4:>4}")
    for set_name in ["H", "S", "R"]:
        s = summary[set_name]
        print(
            f"  {set_name:<6} {s['vr_gt0_rate']:>11.1%} {s['mean_vr']:>10.4f} "
            f"{s['mean_entropy']:>8.3f} {s['mean_pos_frac']:>10.3f} {s['n_positions']:>4}"
        )

    # Decision matrix
    print()
    h_rate = summary["H"]["vr_gt0_rate"]
    s_rate = summary["S"]["vr_gt0_rate"]
    r_rate = summary["R"]["vr_gt0_rate"]
    if s_rate > h_rate * 1.3 and s_rate > r_rate * 1.3:
        verdict = "S >> H: Structural transitions are better branching triggers"
    elif h_rate > s_rate * 1.3 and h_rate > r_rate * 1.3:
        verdict = "H >> S: Entropy signal is real but scale-dependent"
    elif max(h_rate, s_rate) <= r_rate * 1.2:
        verdict = "Both ≈ R: Strong invisible leash — answer basin determined near t=0"
    else:
        verdict = f"Inconclusive: H={h_rate:.1%}, S={s_rate:.1%}, R={r_rate:.1%} — no clear winner"
    print(f"  VERDICT: {verdict}")

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 60)

    # Clean internal fields from output
    for pdata in all_problem_data:
        for key in ["_h_pos", "_s_pos", "_r_pos", "_entropies", "_token_ids"]:
            pdata.pop(key, None)

    report = {
        "config": cfg,
        "base_accuracy": base_acc,
        "n_problems": len(all_problem_data),
        "elapsed_seconds": elapsed,
        "verdict": verdict,
        "summary": summary,
        "problems": all_problem_data,
    }

    report_path = os.path.join(results_base, "trigger_diagnostic_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved report to {report_path}")

    results_vol.commit()
    return report


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================


@app.local_entrypoint()
def main(mode: str = "smoke", lpreg_coef: float = 0.3):
    """
    Launch Spark-MCTS on Modal.

    Usage:
      modal run pilot/modal_spark_smoke.py                            # smoke test (default)
      modal run pilot/modal_spark_smoke.py --mode full                # full ablation
      modal run pilot/modal_spark_smoke.py --mode sweep               # Lp-Reg coef sweep
      modal run pilot/modal_spark_smoke.py --mode viability --lpreg-coef 0.3  # viability run
      modal run pilot/modal_spark_smoke.py --mode math-smoke              # MATH L3-4 smoke test
      modal run pilot/modal_spark_smoke.py --mode math-viability          # MATH L3-4 viability run
      modal run pilot/modal_spark_smoke.py --mode routing-smoke           # condition D smoke
      modal run pilot/modal_spark_smoke.py --mode routing-viability       # A vs D viability
      modal run pilot/modal_spark_smoke.py --mode trigger-diag             # trigger signal diagnostic
    """
    if mode == "smoke":
        print(
            "Launching Spark-MCTS backward pass smoke test on Modal (1x A100-80GB)..."
        )
        print(f"  3 conditions x {SMOKE_CONFIG['total_steps']} steps each")
        print(f"  Model: {SMOKE_CONFIG['model_id']}")
        print("")

        result = run_backward_smoke.remote()

    elif mode == "full":
        print("Launching Spark-MCTS full ablation on Modal (2x A100-80GB)...")
        print(f"  3 conditions x {FULL_CONFIG['total_steps']} steps each")
        print(f"  Model: {FULL_CONFIG['model_id']}")
        print(
            f"  n1={FULL_CONFIG['n1']}, n2={FULL_CONFIG['n2']}, n_total={FULL_CONFIG['n_total']}"
        )
        print("")

        result = run_full_training.remote()

    elif mode == "sweep":
        print("Launching NLL alpha coefficient sweep on Modal (1x A100-80GB)...")
        print(f"  Alphas: [0.1, 0.3, 0.5]")
        print(f"  {SWEEP_CONFIG['total_steps']} steps per trial, condition C only")
        print(f"  Model: {SWEEP_CONFIG['model_id']}")
        print("")

        result = run_lpreg_sweep.remote()

    elif mode == "viability":
        print(f"Launching viability run on Modal (2x A100-80GB)...")
        print(f"  Conditions: A, C")
        print(f"  {VIABILITY_CONFIG['total_steps']} steps per condition")
        print(f"  NLL alpha: {lpreg_coef}")
        print(f"  Model: {VIABILITY_CONFIG['model_id']}")
        print("")

        result = run_viability.remote(lpreg_coef=lpreg_coef)

    elif mode == "math-smoke":
        print("Launching MATH L3-4 smoke test on Modal (2x A100-80GB)...")
        print(f"  Condition C only, {MATH_SMOKE_CONFIG['total_steps']} steps")
        print(
            f"  n1={MATH_SMOKE_CONFIG['n1']}, n2={MATH_SMOKE_CONFIG['n2']}, n_total={MATH_SMOKE_CONFIG['n_total']}"
        )
        print(f"  Model: {MATH_SMOKE_CONFIG['model_id']}")
        print("")

        result = run_math_smoke.remote()

    elif mode == "math-viability":
        print(f"Launching MATH L3-4 viability run on Modal (2x A100-80GB)...")
        print(f"  Conditions: A, C")
        print(f"  {MATH_VIABILITY_CONFIG['total_steps']} steps per condition")
        print(
            f"  n1={MATH_VIABILITY_CONFIG['n1']}, n2={MATH_VIABILITY_CONFIG['n2']}, n_total={MATH_VIABILITY_CONFIG['n_total']}"
        )
        print(f"  NLL alpha: {lpreg_coef}")
        print("")

        result = run_math_viability.remote(lpreg_coef=lpreg_coef)

    elif mode == "routing-smoke":
        print("Launching Condition D routing smoke test on Modal (1x A100-80GB)...")
        print(
            f"  {ROUTING_SMOKE_CONFIG['total_steps']} steps, n={ROUTING_SMOKE_CONFIG['n_total']}"
        )
        print(f"  tau_h={ROUTING_SMOKE_CONFIG['routing_tau_h']}")
        print(f"  Model: {ROUTING_SMOKE_CONFIG['model_id']}")
        print("")

        result = run_routing_smoke.remote()

    elif mode == "routing-viability":
        print(f"Launching routing viability run on Modal (2x A100-80GB)...")
        print(f"  Conditions: A, D")
        print(f"  {ROUTING_VIABILITY_CONFIG['total_steps']} steps per condition")
        print(
            f"  n={ROUTING_VIABILITY_CONFIG['n_total']}, tau_h={ROUTING_VIABILITY_CONFIG['routing_tau_h']}"
        )
        print(f"  NLL alpha: {lpreg_coef}")
        print(f"  Model: {ROUTING_VIABILITY_CONFIG['model_id']}")
        print("")

        result = run_routing_viability.remote(lpreg_coef=lpreg_coef)

    elif mode == "trigger-diag":
        print("Launching trigger signal diagnostic on Modal (1x A100-80GB)...")
        print(
            f"  {TRIGGER_DIAG_CONFIG['n_problems']} problems, "
            f"{TRIGGER_DIAG_CONFIG['k_positions']} positions/set, "
            f"{TRIGGER_DIAG_CONFIG['n_branches']} branches/position"
        )
        print(f"  Model: {TRIGGER_DIAG_CONFIG['model_id']}")
        print("")

        result = run_trigger_diagnostic.remote()

    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Use 'smoke', 'full', 'sweep', 'viability', "
            f"'math-smoke', 'math-viability', 'routing-smoke', 'routing-viability', "
            f"or 'trigger-diag'."
        )

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    if "selected_coef" in result:
        # Sweep result format
        print(f"Selected coef: {result['selected_coef']}")
        print(f"Reason: {result['selection_reason']}")
        print(
            f"Elapsed: {result['elapsed_seconds']:.0f}s ({result['elapsed_seconds']/60:.1f} min)"
        )
        print("\nTrials:")
        for t in result["trials"]:
            fires = "yes" if t.get("protection_fires", t.get("lpreg_fires")) else "NO"
            protect_mag = t.get(
                "mean_protect_magnitude", t.get("mean_lpreg_magnitude", 0)
            )
            ratio_val = t.get(
                "grpo_to_protect_ratio",
                t.get("pg_to_protect_ratio", t.get("pg_to_lpreg_ratio", 0)),
            )
            grpo_loss = t.get("mean_grpo_loss", t.get("mean_pg_loss", 0))
            print(
                f"  alpha={t['coef']}: |GRPO|={grpo_loss:.4f}, "
                f"protect={protect_mag:.6f}, "
                f"ratio={ratio_val:.1f}, fires={fires}"
            )
    elif "verdict" in result:
        # Trigger diagnostic format
        print(f"Verdict: {result['verdict']}")
        print(f"Base accuracy: {result['base_accuracy']:.1%}")
        print(
            f"Elapsed: {result['elapsed_seconds']:.0f}s ({result['elapsed_seconds']/60:.1f} min)"
        )
        print("\nSummary:")
        for set_name, s in result["summary"].items():
            print(
                f"  {set_name}: V(R)>0={s['vr_gt0_rate']:.1%}, mean_vr={s['mean_vr']:.4f}, n={s['n_positions']}"
            )
    else:
        # Standard result format (smoke, full, viability)
        print(f"Overall: {result['overall']}")
        print(f"Passed:  {result['passed']}")
        print(f"Failed:  {result['failed']}")
        print(
            f"Elapsed: {result['elapsed_seconds']:.0f}s ({result['elapsed_seconds']/60:.1f} min)"
        )
        print("\nChecks:")
        for status, message in result["checks"]:
            print(f"  [{status}] {message}")

        if result["overall"] != "PASS":
            print(f"\n{mode.upper()} run FAILED.")
        else:
            print(f"\n{mode.upper()} run PASSED.")
