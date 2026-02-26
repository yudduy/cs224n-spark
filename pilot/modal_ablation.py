"""
CURE × Lp-Reg Ablation Training
================================
3-condition ablation testing CURE + Lp-Reg interaction on Qwen2.5-Math-1.5B.

Conditions:
  A: Vanilla GRPO (baseline — standard rollout, no branching)
  B: CURE (entropy-guided branching, vanilla GRPO loss)
  C: CURE + Lp-Reg (branching + low-prob token protection via forward KL)

Primary metric: |S_t| — branch trigger pool size over training steps.
  Hypothesis: B shows monotonic |S_t| decay; C decays slower.
  Observable divergence expected within 20-30 steps.

Secondary metric: accuracy (with caveat — Qwen2.5-Math improves even with
  random rewards, so accuracy gains are NOT clean evidence of improved
  exploration. Lead with |S_t| divergence).

Papers:
  - CURE (2508.11016): bytedance/CURE, verl 0.4.1.dev
  - Lp-Reg (2510.03222): CarlanLark/Lp-Reg-dev, verl 0.3.1.dev
  - STAPO (2602.15620): S2T mask for spurious token silencing

Hardware: 1× A100-80GB on Modal
Budget: ~$15-20 for all 3 conditions × 50 steps each (optimized)

Usage:
  modal run pilot/modal_ablation.py --condition A
  modal run pilot/modal_ablation.py --condition B
  modal run pilot/modal_ablation.py --condition C
  modal run pilot/modal_ablation.py --condition all
"""

import modal
import json
import os

# ============================================================
# MODAL SETUP
# ============================================================

app = modal.App("cure-lpreg-ablation")

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ablation-results", create_if_missing=True)

# ============================================================
# PATCH CODE: Appended to verl/trainer/ppo/core_algos.py
# Ports compute_policy_loss_lp_reg from CarlanLark/Lp-Reg
# ============================================================

CORE_ALGOS_APPEND = r'''

# === Lp-Reg loss function ===
# Ported from CarlanLark/Lp-Reg-dev verl/trainer/ppo/core_algos.py
# Applies selective KL regularization on low-probability tokens to prevent
# GRPO from suppressing "reasoning spark" tokens in failed branches.

def compute_policy_loss_lp_reg(
    old_log_prob,
    log_prob,
    pos_tgt_log_prob,
    neg_tgt_log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode="token-mean",
    logp_pos_k_percent=0,
    logp_neg_k_percent=0.01,
    ppo_kl_coef=1.0,
    kl_type="low_var_kl",
):
    """Policy loss with Lp-Reg: standard clipped PPO + selective KL on low-prob tokens."""
    import torch

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)

    with torch.no_grad():
        ppo_kl_mean = verl_F.masked_mean(
            kl_penalty(log_prob, old_log_prob, kl_type), response_mask
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
        return pg_loss, pg_clipfrac, ppo_kl_mean

    _, seq_len = advantages.shape
    advantages_flat = advantages.reshape(-1)
    log_prob_flat = log_prob.reshape(-1)

    pos_indices = all_valid_flat_idx[advantages_flat[all_valid_flat_idx] > 0]
    neg_indices = all_valid_flat_idx[advantages_flat[all_valid_flat_idx] < 0]

    def apply_kl_penalty_selective(target_flat_idx, tgt_log_prob):
        if len(target_flat_idx) == 0:
            return
        batch_idx = target_flat_idx // seq_len
        seq_idx = target_flat_idx % seq_len

        # Only regularize tokens where current prob < target prob
        mask = log_prob[batch_idx, seq_idx].detach() < tgt_log_prob[batch_idx, seq_idx]
        batch_idx, seq_idx = batch_idx[mask], seq_idx[mask]

        if len(batch_idx) == 0:
            return

        target_kl = kl_penalty(
            log_prob[batch_idx, seq_idx],
            tgt_log_prob[batch_idx, seq_idx],
            kl_type,
        )
        pg_losses[batch_idx, seq_idx] = (
            -advantages[batch_idx, seq_idx] * ratio[batch_idx, seq_idx]
            + ppo_kl_coef * target_kl
        )

    # Positive-advantage tokens
    if len(pos_indices) > 0 and logp_pos_k_percent > 0:
        pos_logp_k = max(1, int(len(pos_indices) * logp_pos_k_percent))
        if pos_logp_k > 1:
            with torch.no_grad():
                pos_logp_values = log_prob_flat[pos_indices]
                _, pos_logp_indices = pos_logp_values.topk(k=pos_logp_k, largest=False)
                pos_target_idx = pos_indices[pos_logp_indices]
            apply_kl_penalty_selective(pos_target_idx, pos_tgt_log_prob)

    # Negative-advantage tokens (the core Lp-Reg mechanism)
    if len(neg_indices) > 0 and logp_neg_k_percent > 0:
        neg_logp_k = max(1, int(len(neg_indices) * logp_neg_k_percent))
        if neg_logp_k > 1:
            with torch.no_grad():
                neg_logp_values = log_prob_flat[neg_indices]
                _, neg_logp_indices = neg_logp_values.topk(k=neg_logp_k, largest=False)
                neg_target_idx = neg_indices[neg_logp_indices]
            apply_kl_penalty_selective(neg_target_idx, neg_tgt_log_prob)

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl_mean
'''

# ============================================================
# PATCH CODE: Modifications to verl/workers/actor/dp_actor.py
# Adds min-p filtering and Lp-Reg loss mode to update_policy
# ============================================================

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
                    elif self.config.policy_loss.loss_mode == "lp_reg":
                        # Lp-Reg: compute min-p filtered log probs as target distribution
                        from verl.trainer.ppo.core_algos import compute_policy_loss_lp_reg
                        minp_p = self.config.policy_loss.get("minp_p_threshold", 0.02)
                        # Compute target log probs with min-p filtering (inline, Option B)
                        with torch.no_grad():
                            _, tgt_log_prob = self._forward_micro_batch(
                                micro_batch=data, temperature=temperature,
                                calculate_entropy=False, min_p=minp_p,
                            )
                            tgt_log_prob = tgt_log_prob.detach()
                        pg_loss, pg_clipfrac, ppo_kl = compute_policy_loss_lp_reg(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            pos_tgt_log_prob=tgt_log_prob,
                            neg_tgt_log_prob=tgt_log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            loss_agg_mode=loss_agg_mode,
                            logp_pos_k_percent=self.config.policy_loss.get("logp_pos_k_percent", 0),
                            logp_neg_k_percent=self.config.policy_loss.get("logp_neg_k_percent", 0.01),
                            ppo_kl_coef=self.config.policy_loss.get("ppo_kl_coef", 1.0),
                            kl_type=self.config.policy_loss.get("kl_type", "low_var_kl"),
                        )
                        pg_clipfrac_lower = torch.tensor(0.0)
                    else:"""

# Patch 4: Min-p filtering in the remove_padding path of _forward_micro_batch
# Required for use_remove_padding=True (2-3x faster training, avoids padding waste)
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

# Patch 5: Import for compute_policy_loss_lp_reg in dp_actor.py
DP_ACTOR_IMPORT_OLD = (
    "from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, "
    "get_policy_loss_fn, kl_penalty"
)
DP_ACTOR_IMPORT_NEW = (
    "from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, "
    "get_policy_loss_fn, kl_penalty, compute_policy_loss_lp_reg"
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
# PATCH CODE: Data preparation (GSM8K to parquet)
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
# ABLATION TRAINER: Custom trainer with |S_t| logging
# ============================================================

ABLATION_TRAINER_CODE = r'''
"""
Ablation trainer for CURE × Lp-Reg experiment.
Extends CURE's RayDAPOTrainer with:
  - Configurable branching (on/off for Condition A vs B/C)
  - |S_t| metric: count of valid branch candidates per response
  - Logged every st_log_freq steps
"""

import uuid
import json
import os
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


class AblationTrainer(RayPPOTrainer):
    """
    Training loop for the 3-condition ablation.
    Supports: vanilla GRPO (no branching), CURE (branching), CURE + Lp-Reg.
    Logs |S_t| metric every st_log_freq steps.
    """

    def compute_st_metric(self, batch):
        """
        Compute |S_t|: the number of valid branch candidates per response.
        Uses the same logic as CURE's get_critic_key_token_mask but returns
        counts instead of a single selection.

        A token is a valid branch candidate if:
          1. Not at position 0
          2. Not a special token
          3. Decoded text length > 1
          4. Among top-k entropy tokens

        Returns dict of metrics.
        """
        top_k = self.config.trainer.get("critical_top_k", 20)

        # Compute entropies via forward pass
        log_prob_and_entropy = self.actor_rollout_wg.compute_log_prob(batch)
        entropys = log_prob_and_entropy.batch["entropys"]
        response_ids = batch.batch["responses"]
        response_mask = batch.batch.get("response_mask", None)

        batch_size = response_ids.shape[0]
        special_ids = set(int(x) for x in self.tokenizer.all_special_ids)

        candidate_counts = []
        entropy_values = []

        for i in range(batch_size):
            sorted_indices = torch.argsort(entropys[i], descending=True)
            count = 0
            checked = 0
            for idx in sorted_indices:
                if idx == 0:
                    continue
                if checked >= top_k:
                    break
                if idx < len(response_ids[i]):
                    token_id = response_ids[i, idx].item()
                    if int(token_id) not in special_ids:
                        token_text = self.tokenizer.decode(token_id)
                        if len(token_text.strip()) > 1:
                            count += 1
                            checked += 1
            candidate_counts.append(count)

            # Also compute mean entropy for valid response tokens
            if response_mask is not None:
                valid_entropy = entropys[i][response_mask[i] > 0]
                if len(valid_entropy) > 0:
                    entropy_values.append(valid_entropy.mean().item())

        st_metrics = {
            "st/mean_candidates": float(np.mean(candidate_counts)),
            "st/std_candidates": float(np.std(candidate_counts)),
            "st/min_candidates": float(np.min(candidate_counts)),
            "st/max_candidates": float(np.max(candidate_counts)),
        }
        if entropy_values:
            st_metrics["st/mean_entropy"] = float(np.mean(entropy_values))

        return st_metrics

    def get_critic_key_token_mask(self, batch, top_k, mode='entropy'):
        """Copied from CURE's RayDAPOTrainer for use in branching conditions."""
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

    def fit(self):
        """
        Main training loop. Supports three modes:
          - enable_branching=False: Condition A (vanilla GRPO)
          - enable_branching=True, loss_mode=vanilla: Condition B (CURE)
          - enable_branching=True, loss_mode=lp_reg: Condition C (CURE + Lp-Reg)
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        enable_branching = self.config.trainer.get("enable_branching", True)
        st_log_freq = self.config.trainer.get("st_log_freq", 5)
        condition_name = self.config.trainer.get("condition_name", "unknown")

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        # Skip val_before_train for speed
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

        # Accumulate |S_t| trajectory for final output
        st_trajectory = []

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
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
                    with marked_timer("gen", timing_raw, "red"):

                        if enable_branching:
                            # CURE two-stage generation
                            n1 = self.config.trainer.n1
                            n2 = self.config.trainer.n2

                            # Stage 1: generate n1 full responses
                            gen_batch.meta_info["n"] = n1
                            gen_batch_output_1 = self.actor_rollout_wg.generate_sequences(gen_batch)
                            timing_raw.update(gen_batch_output_1.meta_info.get("timing", {}))
                            gen_batch_output_1.meta_info.pop("timing", None)

                            # Find branch points
                            key_token_mask, result_idx = self.get_critic_key_token_mask(
                                gen_batch_output_1,
                                top_k=self.config.trainer.critical_top_k,
                                mode=self.config.trainer.critical_token_type,
                            )

                            # Stage 2: re-concatenate and generate n2 branches
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
                    # Compute on stage 1 responses only (for branching conditions)
                    # to measure the POLICY's natural exploration capacity,
                    # not the artificially branched continuations.
                    if self.global_steps % st_log_freq == 0 or self.global_steps == 1:
                        with marked_timer("st_metric", timing_raw, "magenta"):
                            try:
                                st_batch = (
                                    gen_batch_output_1
                                    if enable_branching
                                    else gen_batch_output
                                )
                                st_metrics = self.compute_st_metric(st_batch)
                                metrics.update(st_metrics)
                                st_trajectory.append({
                                    "step": self.global_steps,
                                    **st_metrics,
                                })
                                print(
                                    f"[{condition_name}] Step {self.global_steps} "
                                    f"|S_t| = {st_metrics['st/mean_candidates']:.1f} "
                                    f"(±{st_metrics['st/std_candidates']:.1f})"
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
                        # DAPO-style filtering (same as CURE)
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

                    # ===== UPDATING =====
                    batch.batch["response_mask"] = compute_response_mask(batch)

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
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item()
                        }
                        metrics.update(old_log_prob_metrics)
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
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

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

                    # Save |S_t| trajectory
                    results_dir = self.config.trainer.get(
                        "results_dir", "/results"
                    )
                    os.makedirs(results_dir, exist_ok=True)
                    trajectory_path = os.path.join(
                        results_dir, f"st_trajectory_{condition_name}.json"
                    )
                    with open(trajectory_path, "w") as f:
                        json.dump({
                            "condition": condition_name,
                            "st_trajectory": st_trajectory,
                            "final_val_metrics": last_val_metrics,
                        }, f, indent=2)
                    print(f"Saved |S_t| trajectory to {trajectory_path}")
                    return

                progress_bar.update(1)
                self.global_steps += 1
'''

# ============================================================
# MAIN ENTRY POINT (runs inside the CURE repo)
# ============================================================

MAIN_ENTRY_CODE = r'''
"""Entry point for ablation training. Replaces CURE's main_dapo.py."""
import os
import socket
import sys

import hydra
import ray
from omegaconf import OmegaConf

# Add CURE repo to path
sys.path.insert(0, "/root/CURE")

from verl.trainer.ppo.reward import get_custom_reward_fn


@hydra.main(config_path="config", config_name="ablation_trainer", version_base=None)
def main(config):
    run_ablation(config)


def run_ablation(config) -> None:
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

        # Only use FSDP strategy
        assert config.actor_rollout_ref.actor.strategy == "fsdp"

        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        }

        # No critic for GRPO
        # No reference policy (Lp-Reg uses min-p filtered logs, not ref policy)

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

        compute_score = get_custom_reward_fn(config)
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

        # Import the ablation trainer
        from recipe.ablation.ablation_trainer import AblationTrainer

        trainer = AblationTrainer(
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
        trainer.fit()


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
  project_name: cure-lpreg-ablation
"""

# ============================================================
# PATCH APPLICATION SCRIPT
# ============================================================


# _build_patch_script removed — using apply_patches() instead


# ============================================================
# MODAL IMAGE
# ============================================================

ablation_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    # Let vllm pin its own torch version to avoid dependency conflicts
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
    # flash-attn: try prebuilt wheel for torch 2.6 + CUDA 12, fall back gracefully
    # (use_remove_padding=True requires flash-attn; False works without it)
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
# CONFIGURATION
# ============================================================

# Ablation configuration — optimized for 1× A100-80GB with Qwen2.5-Math-1.5B
# Memory budget: model 3GB + grads 3GB + optimizer 18GB = 24GB fixed.
# Activations with micro_batch=32: ~11GB. Total peak: ~38GB / 80GB.
ABLATION_CONFIG = {
    "model_id": "Qwen/Qwen2.5-Math-1.5B",
    # Sequence lengths (512 covers 99th-pct GSM8K response length)
    "max_prompt_length": 512,
    "max_response_length": 512,
    # Batch sizes
    "train_batch_size": 4,  # prompts per step (min for fastest wall-clock)
    "gen_batch_size": 4,  # = train_batch_size (no accumulation)
    # PPO batch: process all 64 responses as one mini-batch, 32 at a time
    "ppo_mini_batch_size": 64,  # = train_batch_size * n_total (full batch)
    "ppo_micro_batch_size": 32,  # 64/32 = 2 grad-accum steps (peak ~11GB activations)
    "ppo_max_token_len": 32768,  # for dynamic batching: 32 * 1024 tokens
    # CURE params
    "n1": 4,  # first-stage rollouts
    "n2": 3,  # branches per rollout
    "n_total": 16,  # n1 + n1*n2 = 4 + 12 = 16 responses per prompt
    "critical_top_k": 20,
    "critical_token_type": "entropy",
    # Training
    "total_steps": 50,
    "lr": 1e-6,
    "clip_ratio": 0.2,
    "temperature": 1.0,
    "st_log_freq": 5,  # log |S_t| every N steps
    "test_freq": 25,  # validate at step 25 and 50
    # Lp-Reg params (from CarlanLark/Lp-Reg codebase)
    # NOTE: Codebase uses ppo_kl_coef=1.0, kl_type=low_var_kl
    # Reviewer suggested ppo_kl_coef=0.02 — using codebase defaults for reproducibility
    "lpreg_ppo_kl_coef": 1.0,
    "lpreg_kl_type": "low_var_kl",
    "lpreg_logp_neg_k_percent": 0.01,
    "lpreg_logp_pos_k_percent": 0,
    "lpreg_minp_p_threshold": 0.02,
}


def get_hydra_overrides(condition: str):
    """Generate Hydra config overrides for a given condition."""
    cfg = ABLATION_CONFIG

    if condition == "A":
        loss_mode = "vanilla"
        enable_branching = False
        n_resp = cfg["n_total"]  # 16 standard rollouts for fair comparison
    elif condition == "B":
        loss_mode = "vanilla"
        enable_branching = True
        n_resp = cfg["n_total"]
    elif condition == "C":
        loss_mode = "lp_reg"
        enable_branching = True
        n_resp = cfg["n_total"]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    overrides = [
        # Data
        f"data.train_files=/root/data/gsm8k/train.parquet",
        f"data.val_files=/root/data/gsm8k/test.parquet",
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
        # remove_padding=True: 2-3x faster training (avoids 60-70% wasted compute on padding)
        # Min-p patch supports both paths (rmpad and non-rmpad)
        f"actor_rollout_ref.model.use_remove_padding=True",
        # Actor — optimized batch sizes for A100-80GB
        f"actor_rollout_ref.actor.strategy=fsdp",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg['ppo_micro_batch_size']}",
        # dynamic_bsz packs variable-length GSM8K responses efficiently
        f"actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={cfg['ppo_max_token_len']}",
        f"actor_rollout_ref.actor.grad_clip=1.0",
        f"actor_rollout_ref.actor.clip_ratio={cfg['clip_ratio']}",
        f"actor_rollout_ref.actor.clip_ratio_low={cfg['clip_ratio']}",
        f"actor_rollout_ref.actor.clip_ratio_high={cfg['clip_ratio']}",
        f"actor_rollout_ref.actor.loss_agg_mode=token-mean",
        f"actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.ppo_epochs=1",
        f"actor_rollout_ref.actor.use_kl_loss=False",
        # torch.compile: ~10-15% speedup, 2-3min one-time compile cost amortized over 50 steps
        f"actor_rollout_ref.actor.use_torch_compile=True",
        f"actor_rollout_ref.actor.policy_loss.loss_mode={loss_mode}",
        f"actor_rollout_ref.actor.optim.lr={cfg['lr']}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps=5",
        f"actor_rollout_ref.actor.optim.weight_decay=0.01",
        f"actor_rollout_ref.actor.fsdp_config.param_offload=False",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        # Rollout — vLLM optimized for 1.5B on 80GB
        f"actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.temperature={cfg['temperature']}",
        f"actor_rollout_ref.rollout.top_p=1.0",
        f"actor_rollout_ref.rollout.top_k=-1",
        f"actor_rollout_ref.rollout.n={n_resp}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        # 0.85: KV cache needs only ~1.75GB for 64 seqs × 1024 tokens; leave headroom
        f"actor_rollout_ref.rollout.gpu_memory_utilization=0.85",
        # enforce_eager required when free_cache_engine=True (CUDA graphs incompatible)
        f"actor_rollout_ref.rollout.enforce_eager=True",
        f"actor_rollout_ref.rollout.free_cache_engine=True",
        # 8192 vs old 1024: 4-8x faster prefill (was severely bottlenecked)
        f"actor_rollout_ref.rollout.max_num_batched_tokens=8192",
        f"actor_rollout_ref.rollout.enable_chunked_prefill=True",
        # Log-prob computation also benefits from larger batches
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={cfg['ppo_micro_batch_size']}",
        f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={cfg['ppo_max_token_len']}",
        # Algorithm
        f"algorithm.adv_estimator=grpo",
        f"algorithm.use_kl_in_reward=False",
        f"algorithm.filter_groups.enable=False",
        # Trainer
        f"trainer.nnodes=1",
        f"trainer.n_gpus_per_node=1",
        f"trainer.total_epochs=1000",  # large enough; we stop via total_training_steps
        f"trainer.total_training_steps={cfg['total_steps']}",
        f"trainer.logger=['console']",  # No wandb for now
        f"trainer.project_name=cure-lpreg-ablation",
        f"trainer.experiment_name=condition_{condition}",
        f"trainer.val_before_train=False",
        f"trainer.test_freq={cfg['test_freq']}",
        f"trainer.save_freq=-1",  # No checkpointing
        f"trainer.balance_batch=False",  # Single GPU, no need
        f"trainer.resume_mode=disable",
        f"trainer.device=cuda",
        # Custom ablation params
        f"+trainer.enable_branching={enable_branching}",
        f"+trainer.st_log_freq={cfg['st_log_freq']}",
        f"+trainer.condition_name={condition}",
        f"+trainer.results_dir=/results/condition_{condition}",
    ]

    # CURE branching params (only used when enable_branching=True)
    if enable_branching:
        overrides.extend(
            [
                f"+trainer.n1={cfg['n1']}",
                f"+trainer.n2={cfg['n2']}",
                f"+trainer.critical_top_k={cfg['critical_top_k']}",
                f"+trainer.critical_token_type={cfg['critical_token_type']}",
            ]
        )

    # Lp-Reg params (only used when loss_mode=lp_reg)
    if loss_mode == "lp_reg":
        overrides.extend(
            [
                f"++actor_rollout_ref.actor.policy_loss.logp_pos_k_percent={cfg['lpreg_logp_pos_k_percent']}",
                f"++actor_rollout_ref.actor.policy_loss.logp_neg_k_percent={cfg['lpreg_logp_neg_k_percent']}",
                f"++actor_rollout_ref.actor.policy_loss.ppo_kl_coef={cfg['lpreg_ppo_kl_coef']}",
                f"++actor_rollout_ref.actor.policy_loss.kl_type={cfg['lpreg_kl_type']}",
                f"++actor_rollout_ref.actor.policy_loss.minp_p_threshold={cfg['lpreg_minp_p_threshold']}",
            ]
        )

    # Ray init
    overrides.append("ray_init.num_cpus=4")

    return overrides


# ============================================================
# MODAL FUNCTIONS
# ============================================================


@app.function(
    gpu="A100-80GB",
    timeout=14400,  # 4 hours
    image=ablation_image,
    volumes={
        "/hf-cache": hf_cache,
        "/results": results_vol,
    },
    # Qwen2.5-Math-1.5B is public; add secrets=[modal.Secret.from_name("huggingface-secret")]
    # if using a gated model
)
def run_condition(condition: str):
    """Run a single ablation condition (A, B, or C)."""
    import subprocess
    import sys
    import time

    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"
    os.environ["WANDB_MODE"] = "disabled"

    start_time = time.time()
    print(f"=" * 60)
    print(f"Starting condition {condition}")
    print(f"=" * 60)

    # Step 1: Apply patches to CURE repo
    print("\n--- Applying patches ---")
    apply_patches()

    # Step 2: Prepare GSM8K data
    print("\n--- Preparing GSM8K data ---")
    sys.path.insert(0, "/root/CURE")
    subprocess.run([sys.executable, "/root/CURE/prepare_data.py"], check=True)

    # Step 3: Download model (cached in volume)
    print("\n--- Downloading model ---")
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        ABLATION_CONFIG["model_id"],
        cache_dir="/hf-cache",
    )
    print(f"Model downloaded to: {model_path}")

    # Step 4: Run training via Hydra
    print(f"\n--- Running condition {condition} ---")
    overrides = get_hydra_overrides(condition)

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "recipe.ablation.main",
    ] + overrides

    print(f"Command: {' '.join(cmd[:5])} ... ({len(overrides)} overrides)")

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/CURE:" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        cwd="/root/CURE",
        env=env,
        capture_output=False,
    )

    elapsed = time.time() - start_time
    print(f"\n--- Condition {condition} completed in {elapsed/60:.1f} min ---")
    print(f"Exit code: {result.returncode}")

    # Commit results to volume
    results_vol.commit()

    return {
        "condition": condition,
        "elapsed_minutes": elapsed / 60,
        "exit_code": result.returncode,
    }


def apply_patches():
    """Apply all patches to the CURE repo at runtime."""
    import os

    CURE_ROOT = "/root/CURE"

    # Patch 1: Append Lp-Reg loss to core_algos.py
    core_algos_path = os.path.join(CURE_ROOT, "verl/trainer/ppo/core_algos.py")
    with open(core_algos_path, "r") as f:
        existing = f.read()
    if "compute_policy_loss_lp_reg" not in existing:
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
        ), f"Could not find _forward_micro_batch signature to patch"
        content = content.replace(DP_ACTOR_SIG_OLD, DP_ACTOR_SIG_NEW)
        patched = True

    # 2b: Add min-p filtering in non-remove_padding path
    if "min_p is not None and min_p > 0" not in content:
        assert (
            DP_ACTOR_FORWARD_OLD in content
        ), f"Could not find forward_micro_batch body to patch"
        content = content.replace(DP_ACTOR_FORWARD_OLD, DP_ACTOR_FORWARD_NEW)
        patched = True

    # 2c: Add min-p filtering in remove_padding path
    if "Min-p filtering for Lp-Reg (remove_padding path)" not in content:
        assert (
            DP_ACTOR_RMPAD_OLD in content
        ), f"Could not find remove_padding logprobs section to patch"
        content = content.replace(DP_ACTOR_RMPAD_OLD, DP_ACTOR_RMPAD_NEW)
        patched = True

    # 2d: Add import for compute_policy_loss_lp_reg (BEFORE 2e, since 2e
    # introduces the same string via inline import and would shadow this check)
    if "compute_policy_loss_lp_reg" not in content:
        assert (
            DP_ACTOR_IMPORT_OLD in content
        ), f"Could not find dp_actor import line to patch"
        content = content.replace(DP_ACTOR_IMPORT_OLD, DP_ACTOR_IMPORT_NEW)
        patched = True

    # 2e: Add Lp-Reg branch in update_policy
    if 'loss_mode == "lp_reg"' not in content:
        assert (
            DP_ACTOR_LOSS_OLD in content
        ), f"Could not find update_policy loss section to patch"
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

    # Patch 3: Create ablation recipe
    recipe_dir = os.path.join(CURE_ROOT, "recipe/ablation")
    config_dir = os.path.join(recipe_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    with open(os.path.join(recipe_dir, "__init__.py"), "w") as f:
        f.write("")

    with open(os.path.join(recipe_dir, "ablation_trainer.py"), "w") as f:
        f.write(ABLATION_TRAINER_CODE)

    with open(os.path.join(recipe_dir, "main.py"), "w") as f:
        f.write(MAIN_ENTRY_CODE)

    with open(os.path.join(config_dir, "ablation_trainer.yaml"), "w") as f:
        f.write(CONFIG_YAML)

    # Patch 4: Create data prep script
    with open(os.path.join(CURE_ROOT, "prepare_data.py"), "w") as f:
        f.write(DATA_PREP_SCRIPT)

    print("All patches applied.")


@app.function(
    gpu="A100-80GB",
    timeout=43200,  # 12 hours (for all 3 conditions)
    image=ablation_image,
    volumes={
        "/hf-cache": hf_cache,
        "/results": results_vol,
    },
    # Qwen2.5-Math-1.5B is public; add secrets=[modal.Secret.from_name("huggingface-secret")]
    # if using a gated model
)
def run_all_conditions():
    """Run all 3 conditions sequentially and produce comparison."""
    import time

    results = {}
    for condition in ["A", "B", "C"]:
        print(f"\n{'=' * 60}")
        print(f"STARTING CONDITION {condition}")
        print(f"{'=' * 60}\n")

        result = run_condition.local(condition)
        results[condition] = result

        if result["exit_code"] != 0:
            print(
                f"WARNING: Condition {condition} failed with exit code {result['exit_code']}"
            )

    # Save combined results
    os.makedirs("/results/combined", exist_ok=True)
    with open("/results/combined/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Load and compare |S_t| trajectories
    print("\n" + "=" * 60)
    print("COMPARISON: |S_t| trajectories")
    print("=" * 60)

    for condition in ["A", "B", "C"]:
        traj_path = f"/results/condition_{condition}/st_trajectory_{condition}.json"
        if os.path.exists(traj_path):
            with open(traj_path) as f:
                data = json.load(f)
            traj = data.get("st_trajectory", [])
            print(f"\nCondition {condition}:")
            for point in traj:
                print(
                    f"  Step {point['step']:3d}: "
                    f"|S_t| = {point['st/mean_candidates']:.1f} "
                    f"(±{point['st/std_candidates']:.1f})"
                )

    results_vol.commit()
    return results


# ============================================================
# SMOKE TEST: minimal 2-step run of condition C to verify pipeline
# ============================================================

SMOKE_CONFIG = {
    **ABLATION_CONFIG,
    "train_batch_size": 2,
    "gen_batch_size": 2,
    "n1": 2,
    "n2": 1,
    "n_total": 4,  # 2 + 2*1 = 4 responses per prompt
    "max_response_length": 256,  # shorter for speed
    "ppo_mini_batch_size": 2,  # must be <= train_batch_size
    "ppo_micro_batch_size": 2,  # must be <= ppo_mini_batch_size
    "ppo_max_token_len": 8192,
    "total_steps": 2,
    "st_log_freq": 1,  # log every step
    "test_freq": 2,  # validate at final step
}


def get_smoke_overrides():
    """Hydra overrides for the smoke test (condition C with minimal config)."""
    cfg = SMOKE_CONFIG
    loss_mode = "lp_reg"
    enable_branching = True

    overrides = [
        # Data
        f"data.train_files=/root/data/gsm8k/train.parquet",
        f"data.val_files=/root/data/gsm8k/test.parquet",
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
        # use_remove_padding=False for smoke: avoids flash-attn dependency risk
        f"actor_rollout_ref.model.use_remove_padding=False",
        # Actor — small batch for smoke test
        f"actor_rollout_ref.actor.strategy=fsdp",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg['ppo_micro_batch_size']}",
        f"actor_rollout_ref.actor.use_dynamic_bsz=False",  # simpler for smoke test
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={cfg['ppo_max_token_len']}",
        f"actor_rollout_ref.actor.grad_clip=1.0",
        f"actor_rollout_ref.actor.clip_ratio={cfg['clip_ratio']}",
        f"actor_rollout_ref.actor.clip_ratio_low={cfg['clip_ratio']}",
        f"actor_rollout_ref.actor.clip_ratio_high={cfg['clip_ratio']}",
        f"actor_rollout_ref.actor.loss_agg_mode=token-mean",
        f"actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.ppo_epochs=1",
        f"actor_rollout_ref.actor.use_kl_loss=False",
        f"actor_rollout_ref.actor.use_torch_compile=False",  # skip compile overhead for smoke
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
        f"actor_rollout_ref.rollout.n={cfg['n_total']}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        f"actor_rollout_ref.rollout.enforce_eager=True",
        f"actor_rollout_ref.rollout.free_cache_engine=True",
        f"actor_rollout_ref.rollout.max_num_batched_tokens=8192",
        f"actor_rollout_ref.rollout.enable_chunked_prefill=True",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={cfg['ppo_micro_batch_size']}",
        f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={cfg['ppo_max_token_len']}",
        # Algorithm
        f"algorithm.adv_estimator=grpo",
        f"algorithm.use_kl_in_reward=False",
        f"algorithm.filter_groups.enable=False",
        # Trainer
        f"trainer.nnodes=1",
        f"trainer.n_gpus_per_node=1",
        f"trainer.total_epochs=1000",
        f"trainer.total_training_steps={cfg['total_steps']}",
        f"trainer.logger=['console']",
        f"trainer.project_name=cure-lpreg-smoke",
        f"trainer.experiment_name=smoke_test",
        f"trainer.val_before_train=False",
        f"trainer.test_freq={cfg['test_freq']}",
        f"trainer.save_freq=-1",
        f"trainer.balance_batch=False",
        f"trainer.resume_mode=disable",
        f"trainer.device=cuda",
        # Custom
        f"+trainer.enable_branching={enable_branching}",
        f"+trainer.n1={cfg['n1']}",
        f"+trainer.n2={cfg['n2']}",
        f"+trainer.critical_top_k={cfg['critical_top_k']}",
        f"+trainer.critical_token_type={cfg['critical_token_type']}",
        f"+trainer.st_log_freq={cfg['st_log_freq']}",
        f"+trainer.condition_name=smoke",
        f"+trainer.results_dir=/results/smoke",
        # Lp-Reg
        f"++actor_rollout_ref.actor.policy_loss.logp_pos_k_percent={cfg['lpreg_logp_pos_k_percent']}",
        f"++actor_rollout_ref.actor.policy_loss.logp_neg_k_percent={cfg['lpreg_logp_neg_k_percent']}",
        f"++actor_rollout_ref.actor.policy_loss.ppo_kl_coef={cfg['lpreg_ppo_kl_coef']}",
        f"++actor_rollout_ref.actor.policy_loss.kl_type={cfg['lpreg_kl_type']}",
        f"++actor_rollout_ref.actor.policy_loss.minp_p_threshold={cfg['lpreg_minp_p_threshold']}",
    ]

    overrides.append("ray_init.num_cpus=4")
    return overrides


@app.function(
    gpu="A100-80GB",
    timeout=1800,  # 30 min max for smoke test
    image=ablation_image,
    volumes={
        "/hf-cache": hf_cache,
        "/results": results_vol,
    },
    # Qwen2.5-Math-1.5B is public; add secrets=[modal.Secret.from_name("huggingface-secret")]
    # if using a gated model
)
def run_smoke_test():
    """Smoke test: 2-step condition C (branching + Lp-Reg) to verify full pipeline."""
    import subprocess
    import sys
    import time

    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"
    os.environ["WANDB_MODE"] = "disabled"

    start = time.time()
    print("=" * 60)
    print("SMOKE TEST: 2-step condition C (CURE + Lp-Reg)")
    print("=" * 60)

    # Step 1: Apply patches
    print("\n--- Step 1: Applying patches ---")
    apply_patches()

    # Verify patches
    with open("/root/CURE/verl/workers/actor/dp_actor.py") as f:
        dp_content = f.read()
    checks = {
        "min_p=None": "signature patch",
        "Min-p filtering for Lp-Reg (remove_padding path)": "rmpad patch",
        "Min-p filtering for Lp-Reg target distribution": "non-rmpad patch",
        'loss_mode == "lp_reg"': "loss branch",
        "compute_policy_loss_lp_reg": "import",
    }
    for marker, desc in checks.items():
        assert marker in dp_content, f"Patch verification FAILED: {desc} ({marker})"
        print(f"  [OK] {desc}")

    with open("/root/CURE/verl/trainer/ppo/core_algos.py") as f:
        ca_content = f.read()
    assert "compute_policy_loss_lp_reg" in ca_content, "Lp-Reg loss not in core_algos"
    print("  [OK] Lp-Reg loss function appended to core_algos.py")
    print(f"  Patches verified in {time.time()-start:.1f}s")

    # Step 2: Prepare data
    print("\n--- Step 2: Preparing GSM8K data ---")
    t0 = time.time()
    sys.path.insert(0, "/root/CURE")
    subprocess.run([sys.executable, "/root/CURE/prepare_data.py"], check=True)
    assert os.path.exists("/root/data/gsm8k/train.parquet"), "train.parquet missing"
    assert os.path.exists("/root/data/gsm8k/test.parquet"), "test.parquet missing"
    print(f"  Data prepared in {time.time()-t0:.1f}s")

    # Step 3: Download model
    print("\n--- Step 3: Downloading model ---")
    t0 = time.time()
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(SMOKE_CONFIG["model_id"], cache_dir="/hf-cache")
    print(f"  Model at: {model_path} ({time.time()-t0:.1f}s)")

    # Step 4: Run 2-step training
    print("\n--- Step 4: Running 2-step condition C ---")
    t0 = time.time()
    overrides = get_smoke_overrides()

    cmd = [sys.executable, "-m", "recipe.ablation.main"] + overrides

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/CURE:" + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, cwd="/root/CURE", env=env, capture_output=False)

    train_time = time.time() - t0
    print(f"\n  Training exit code: {result.returncode} ({train_time:.1f}s)")

    # Step 5: Verify outputs
    print("\n--- Step 5: Verifying outputs ---")
    traj_path = "/results/smoke/st_trajectory_smoke.json"
    if os.path.exists(traj_path):
        with open(traj_path) as f:
            traj_data = json.load(f)
        st_traj = traj_data.get("st_trajectory", [])
        print(f"  |S_t| trajectory has {len(st_traj)} data points")
        for pt in st_traj:
            print(f"    Step {pt['step']}: |S_t| = {pt['st/mean_candidates']:.1f}")
        assert len(st_traj) > 0, "|S_t| trajectory is empty"
        print("  [OK] |S_t| metric collected")
    else:
        print(f"  [WARN] No trajectory file at {traj_path}")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    if result.returncode == 0:
        print(f"SMOKE TEST PASSED in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    else:
        print(f"SMOKE TEST FAILED (exit code {result.returncode})")
    print(f"{'=' * 60}")

    results_vol.commit()

    return {
        "status": "pass" if result.returncode == 0 else "fail",
        "exit_code": result.returncode,
        "elapsed_seconds": elapsed,
        "st_trajectory_points": len(st_traj) if os.path.exists(traj_path) else 0,
    }


# ============================================================
# CLI ENTRY POINT
# ============================================================


@app.local_entrypoint()
def main(condition: str = "all"):
    """
    Run the ablation experiment.

    Args:
        condition: Which condition to run. One of: A, B, C, all, smoke
    """
    if condition == "smoke":
        result = run_smoke_test.remote()
    elif condition == "all":
        result = run_all_conditions.remote()
    elif condition in ("A", "B", "C"):
        result = run_condition.remote(condition)
    else:
        raise ValueError(f"Unknown condition: {condition}. Use A, B, C, all, or smoke.")

    print(f"\nResult: {json.dumps(result, indent=2)}")
