"""
Semantic Branch Diagnostic (MVE)
================================
Inference-only validation of semantic-directed branching with automatic fallback pivots.

Primary MVE:
  1) Entropy tripwire on greedy trace
  2) K=3 micro-rollouts (W tokens)
  3) Sentence-embedder similarity on micro-rollouts
  4) Oracle completion for all branches
  5) False Convergence / False Divergence metrics

Automatic fallback decision tree (if embedder fails):
  Pivot 1: Symbolic math extraction gate
  Pivot 2: Implicit likelihood divergence gate (next-token KL)
  Pivot 3: Constrained action-space branching (reasoning-move seeds)
  Pivot 4: LLM-as-judge equivalence gate

Usage:
  modal run pilot/modal_embedder_diagnostic.py::run_smoke
  modal run pilot/modal_embedder_diagnostic.py::run_full
  modal run pilot/modal_embedder_diagnostic.py::run_auto
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import modal

app = modal.App("spark-embedder-diagnostic")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.4")
    .pip_install(
        "transformers>=4.45.0,<5.0.0",
        "datasets",
        "numpy",
        "sentence-transformers",
        "scipy",
        "huggingface_hub",
    )
)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("spark-pilot-results", create_if_missing=True)

PRIMARY_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
FALLBACK_MODEL = "Qwen/Qwen2.5-Math-1.5B"

EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SMOKE_CFG = {
    "name": "smoke",
    "n_problems": 8,
    "level": 5,
    "seed": 42,
    "max_new_tokens": 512,
    "oracle_max_new_tokens": 768,
    "tripwire_theta": 1.28,
    "tripwire_max_pos": 220,
    "K": 3,
    "W": 10,
    "kl_probe_k": 20,
    "cos_low": 0.50,
    "cos_high": 0.85,
    "kl_low": 0.02,
    "kl_high": 0.20,
    "fail_fc": 0.20,
    "fail_fd": 0.20,
}

FULL_CFG = {
    "name": "full",
    "n_problems": 50,
    "level": 5,
    "seed": 42,
    "max_new_tokens": 1024,
    "oracle_max_new_tokens": 1024,
    "tripwire_theta": 1.28,
    "tripwire_max_pos": 260,
    "K": 3,
    "W": 10,
    "kl_probe_k": 20,
    "cos_low": 0.50,
    "cos_high": 0.85,
    "kl_low": 0.02,
    "kl_high": 0.20,
    "fail_fc": 0.20,
    "fail_fd": 0.20,
}

REASONING_SEEDS = [" Let", " Assume", " Notice"]


@dataclass
class PairEval:
    sim: float
    ans_a: str
    ans_b: str
    step_a: int
    step_b: int
    pred_close: bool
    pred_far: bool
    oracle_same_answer: bool
    oracle_same_steps: bool


# -----------------------------
# Utilities
# -----------------------------


def _extract_boxed_answer(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _norm_answer(x: Optional[str]) -> str:
    if x is None:
        return ""
    return x.strip().replace(" ", "").replace(",", "")


def _verify_answer(pred: Optional[str], gt: str) -> bool:
    p = _norm_answer(pred)
    g = _norm_answer(gt)
    if not p:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except Exception:
        return False


def _extract_math_symbols(text: str) -> str:
    spans = []
    spans.extend(re.findall(r"\$([^$]+)\$", text))
    spans.extend(re.findall(r"\\\[([^\]]+)\\\]", text))
    spans.extend(re.findall(r"\\\(([^\)]+)\\\)", text))
    expr = " ".join(spans)
    if not expr:
        expr = " ".join(re.findall(r"[A-Za-z0-9_\\\^\{\}\+\-\*/=<>]+", text))
    expr = re.sub(r"\s+", "", expr)
    return expr


def _trunc_probs_from_logprobs(logprob_dict: Dict[int, float], top_k: int = 20) -> Dict[int, float]:
    def _lp(v):
        return float(v.logprob) if hasattr(v, "logprob") else float(v)

    items = sorted(logprob_dict.items(), key=lambda x: _lp(x[1]), reverse=True)[:top_k]
    probs = {int(tid): math.exp(_lp(lp)) for tid, lp in items}
    z = sum(probs.values())
    if z <= 0:
        return {}
    return {k: v / z for k, v in probs.items()}


def _sym_kl(p: Dict[int, float], q: Dict[int, float], eps: float = 1e-8) -> float:
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0
    p2 = {k: p.get(k, eps) for k in keys}
    q2 = {k: q.get(k, eps) for k in keys}
    zp = sum(p2.values())
    zq = sum(q2.values())
    p2 = {k: v / zp for k, v in p2.items()}
    q2 = {k: v / zq for k, v in q2.items()}
    kl_pq = sum(p2[k] * math.log((p2[k] + eps) / (q2[k] + eps)) for k in keys)
    kl_qp = sum(q2[k] * math.log((q2[k] + eps) / (p2[k] + eps)) for k in keys)
    return 0.5 * (kl_pq + kl_qp)


def _entropy_from_logprobs(pos_lps: Dict[int, float]) -> float:
    probs = []
    for v in pos_lps.values():
        lp = float(v.logprob) if hasattr(v, "logprob") else float(v)
        probs.append(math.exp(lp))
    z = sum(probs)
    if z <= 0:
        return 0.0
    probs = [p / z for p in probs]
    return -sum(p * math.log(p + 1e-10) for p in probs if p > 0)


def _problem_prompt(tokenizer, problem: str) -> str:
    instruction = "Let's think step by step and put your final answer within \\boxed{}."
    msgs = [{"role": "user", "content": f"{problem} {instruction}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _load_math_problems(n_problems: int, level: int, seed: int):
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [r for r in ds if r.get("level", 0) == level]
    if len(rows) < n_problems:
        rows = [r for r in ds if r.get("level", 0) in (4, 5)]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(rows), size=min(n_problems, len(rows)), replace=False)
    out = []
    for i in sorted(idx.tolist()):
        r = rows[i]
        out.append({"problem": r["problem"], "answer": r.get("answer", "")})
    return out


def _pick_tripwire_position(token_ids, logprobs_list, tokenizer, theta: float, max_pos: int) -> Optional[int]:
    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    n = min(len(token_ids), len(logprobs_list), max_pos)
    for pos in range(1, n):
        lps = logprobs_list[pos]
        if not lps:
            continue
        tid = int(token_ids[pos])
        if tid in special_ids:
            continue
        ttxt = tokenizer.decode(tid)
        if len(ttxt.strip()) <= 1:
            continue
        H = _entropy_from_logprobs(lps)
        if H > theta:
            return pos
    return None


def _pairwise_indices(k: int) -> List[Tuple[int, int]]:
    return list(combinations(range(k), 2))


# -----------------------------
# Core evaluator
# -----------------------------


def _run_case(cfg: dict, mode: str) -> dict:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from vllm import LLM, SamplingParams

    start = time.time()
    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"

    model_id = PRIMARY_MODEL
    llm = None

    # Model load with fallback
    for cand in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            llm = LLM(
                model=cand,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
                dtype="bfloat16",
                max_model_len=4096,
                trust_remote_code=True,
            )
            model_id = cand
            break
        except Exception as e:
            print(f"Model load failed for {cand}: {e}")
    if llm is None:
        raise RuntimeError("Unable to load reasoning model")

    tokenizer = llm.get_tokenizer()
    embedder = SentenceTransformer(EMBEDDER_MODEL)

    problems = _load_math_problems(cfg["n_problems"], cfg["level"], cfg["seed"])
    prompts = [_problem_prompt(tokenizer, p["problem"]) for p in problems]

    greedy_params = SamplingParams(
        temperature=0,
        max_tokens=cfg["max_new_tokens"],
        logprobs=20,
    )

    greedy_out = llm.generate(prompts, greedy_params)

    examples = []
    pair_records = []

    micro_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=cfg["W"],
        n=cfg["K"],
        logprobs=20,
    )

    if mode == "constrained":
        micro_params = SamplingParams(
            temperature=0.8,
            top_p=1.0,
            max_tokens=max(2, cfg["W"] - 1),
            n=1,
            logprobs=20,
        )

    judge_params = SamplingParams(temperature=0, max_tokens=2, logprobs=5)

    usable = 0

    for i, (g, prob) in enumerate(zip(greedy_out, problems)):
        out = g.outputs[0]
        token_ids = list(out.token_ids)
        lps = out.logprobs
        bp = _pick_tripwire_position(token_ids, lps, tokenizer, cfg["tripwire_theta"], cfg["tripwire_max_pos"])
        if bp is None:
            continue

        usable += 1
        prefix_text = tokenizer.decode(token_ids[:bp])
        base_prompt = prompts[i] + prefix_text

        # Stage 1: micro branches
        micro_texts = []
        micro_logprobs = []
        branch_prompts = []

        if mode == "constrained":
            for s in REASONING_SEEDS[: cfg["K"]]:
                forced_prefix = base_prompt + s
                branch_prompts.append(forced_prefix)
            m_out = llm.generate(branch_prompts, micro_params)
            for j, mo in enumerate(m_out):
                t = REASONING_SEEDS[j] + (mo.outputs[0].text or "")
                micro_texts.append(t)
                micro_logprobs.append(mo.outputs[0].logprobs)
        else:
            m_out = llm.generate([base_prompt], micro_params)[0]
            for m in m_out.outputs[: cfg["K"]]:
                micro_texts.append(m.text)
                micro_logprobs.append(m.logprobs)

        # Stage 2: scores that define predicted equivalence/divergence
        pair_scores = {}
        if mode == "embedder":
            emb = embedder.encode(micro_texts, normalize_embeddings=True)
            for a, b in _pairwise_indices(cfg["K"]):
                sim = float(np.dot(emb[a], emb[b]))
                pair_scores[(a, b)] = sim
        elif mode == "symbolic":
            mats = [_extract_math_symbols(t) for t in micro_texts]
            for a, b in _pairwise_indices(cfg["K"]):
                sim = 1.0 if mats[a] and mats[a] == mats[b] else 0.0
                pair_scores[(a, b)] = sim
        elif mode == "kl":
            probe_prompts = [base_prompt + t for t in micro_texts]
            probe_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1, n=1, logprobs=cfg["kl_probe_k"])
            probes = llm.generate(probe_prompts, probe_params)
            dists = []
            for pr in probes:
                if not pr.outputs or not pr.outputs[0].logprobs:
                    dists.append({})
                    continue
                lp0 = pr.outputs[0].logprobs[0]
                dists.append(_trunc_probs_from_logprobs(lp0, cfg["kl_probe_k"]))
            for a, b in _pairwise_indices(cfg["K"]):
                pair_scores[(a, b)] = _sym_kl(dists[a], dists[b])
        elif mode == "constrained":
            emb = embedder.encode(micro_texts, normalize_embeddings=True)
            for a, b in _pairwise_indices(cfg["K"]):
                sim = float(np.dot(emb[a], emb[b]))
                pair_scores[(a, b)] = sim
        elif mode == "judge":
            for a, b in _pairwise_indices(cfg["K"]):
                judge_prompt = (
                    "Context math solution prefix. "
                    f"Step A: {micro_texts[a]}\n"
                    f"Step B: {micro_texts[b]}\n"
                    "Are Step A and Step B mathematically equivalent? Answer Yes or No."
                )
                jo = llm.generate([judge_prompt], judge_params)[0].outputs[0]
                txt = (jo.text or "").strip().lower()
                # Represent as pseudo-sim in [0,1]
                pair_scores[(a, b)] = 1.0 if txt.startswith("yes") else 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Stage 3: oracle completion on all branches
        oracle_prompts = [base_prompt + t for t in micro_texts]
        oracle_params = SamplingParams(
            temperature=0,
            max_tokens=cfg["oracle_max_new_tokens"],
            n=1,
            logprobs=0,
        )
        o_out = llm.generate(oracle_prompts, oracle_params)

        final_answers = []
        step_counts = []
        final_correct = []
        for oo in o_out:
            txt = oo.outputs[0].text
            ans = _extract_boxed_answer(txt)
            final_answers.append(_norm_answer(ans))
            step_counts.append(len(oo.outputs[0].token_ids))
            final_correct.append(_verify_answer(ans, prob["answer"]))

        problem_pairs = []
        for a, b in _pairwise_indices(cfg["K"]):
            score = float(pair_scores[(a, b)])

            if mode == "kl":
                pred_close = score <= cfg["kl_low"]
                pred_far = score >= cfg["kl_high"]
            else:
                pred_close = score >= cfg["cos_high"]
                pred_far = score <= cfg["cos_low"]

            same_answer = final_answers[a] != "" and final_answers[a] == final_answers[b]
            same_steps = step_counts[a] == step_counts[b]

            rec = {
                "problem_idx": i,
                "pair": [a, b],
                "score": score,
                "pred_close": pred_close,
                "pred_far": pred_far,
                "same_answer": same_answer,
                "same_steps": same_steps,
                "ans_a": final_answers[a],
                "ans_b": final_answers[b],
                "correct_a": final_correct[a],
                "correct_b": final_correct[b],
                "tripwire_pos": bp,
            }
            pair_records.append(rec)
            problem_pairs.append(rec)

        examples.append(
            {
                "problem_idx": i,
                "tripwire_pos": bp,
                "problem": prob["problem"][:240],
                "micro_texts": micro_texts,
                "final_answers": final_answers,
                "final_correct": final_correct,
                "pairs": problem_pairs,
            }
        )

    # Aggregate metrics
    pred_far = [r for r in pair_records if r["pred_far"]]
    pred_close = [r for r in pair_records if r["pred_close"]]

    false_div_num = sum(1 for r in pred_far if r["same_answer"] and r["same_steps"])
    false_div_den = max(1, len(pred_far))
    false_div = false_div_num / false_div_den

    false_conv_num = sum(1 for r in pred_close if not r["same_answer"])
    false_conv_den = max(1, len(pred_close))
    false_conv = false_conv_num / false_conv_den

    overall_pairs = len(pair_records)
    any_correct = sum(1 for ex in examples if any(ex["final_correct"]))

    summary = {
        "mode": mode,
        "model_id": model_id,
        "embedder_model": EMBEDDER_MODEL,
        "n_requested": cfg["n_problems"],
        "n_usable": usable,
        "n_pairs": overall_pairs,
        "false_divergence_rate": false_div,
        "false_divergence_num": false_div_num,
        "false_divergence_den": false_div_den,
        "false_convergence_rate": false_conv,
        "false_convergence_num": false_conv_num,
        "false_convergence_den": false_conv_den,
        "has_any_correct_branch_problem_count": any_correct,
        "elapsed_seconds": time.time() - start,
        "thresholds": {
            "cos_low": cfg["cos_low"],
            "cos_high": cfg["cos_high"],
            "kl_low": cfg["kl_low"],
            "kl_high": cfg["kl_high"],
        },
    }

    verdict = {
        "pass": (false_conv <= cfg["fail_fc"] and false_div <= cfg["fail_fd"]),
        "failed_fc": false_conv > cfg["fail_fc"],
        "failed_fd": false_div > cfg["fail_fd"],
        "criteria": {"max_false_conv": cfg["fail_fc"], "max_false_div": cfg["fail_fd"]},
    }

    return {
        "summary": summary,
        "verdict": verdict,
        "examples": examples,
        "pair_records": pair_records,
    }


def _write_report(out_dir: str, name: str, payload: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    path = os.path.join(out_dir, f"{name}_{ts}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# -----------------------------
# Modal entrypoints
# -----------------------------


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
)
def run_smoke():
    base = "/results/embedder_diagnostic"
    os.makedirs(base, exist_ok=True)
    payload = {
        "run_type": "smoke",
        "timestamp": int(time.time()),
        "cfg": SMOKE_CFG,
        "stages": [],
    }

    primary = _run_case(SMOKE_CFG, "embedder")
    payload["stages"].append(primary)

    if not primary["verdict"]["pass"]:
        p1 = _run_case(SMOKE_CFG, "symbolic")
        payload["stages"].append(p1)
        if not p1["verdict"]["pass"]:
            p2 = _run_case(SMOKE_CFG, "kl")
            payload["stages"].append(p2)
            if not p2["verdict"]["pass"]:
                p3 = _run_case(SMOKE_CFG, "constrained")
                payload["stages"].append(p3)
                if not p3["verdict"]["pass"]:
                    p4 = _run_case(SMOKE_CFG, "judge")
                    payload["stages"].append(p4)

    path = _write_report(base, "smoke_report", payload)
    return {"saved": path, "summary": [s["summary"] for s in payload["stages"]]}


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
)
def run_full():
    base = "/results/embedder_diagnostic"
    os.makedirs(base, exist_ok=True)
    payload = {
        "run_type": "full",
        "timestamp": int(time.time()),
        "cfg": FULL_CFG,
        "stages": [],
    }

    ckpt_path = os.path.join(base, "full_checkpoint.json")

    primary = _run_case(FULL_CFG, "embedder")
    payload["stages"].append(primary)
    with open(ckpt_path, "w") as f:
        json.dump(payload, f, indent=2)

    if not primary["verdict"]["pass"]:
        p1 = _run_case(FULL_CFG, "symbolic")
        payload["stages"].append(p1)
        with open(ckpt_path, "w") as f:
            json.dump(payload, f, indent=2)
        if not p1["verdict"]["pass"]:
            p2 = _run_case(FULL_CFG, "kl")
            payload["stages"].append(p2)
            with open(ckpt_path, "w") as f:
                json.dump(payload, f, indent=2)
            if not p2["verdict"]["pass"]:
                p3 = _run_case(FULL_CFG, "constrained")
                payload["stages"].append(p3)
                with open(ckpt_path, "w") as f:
                    json.dump(payload, f, indent=2)
                if not p3["verdict"]["pass"]:
                    p4 = _run_case(FULL_CFG, "judge")
                    payload["stages"].append(p4)
                    with open(ckpt_path, "w") as f:
                        json.dump(payload, f, indent=2)

    path = _write_report(base, "full_report", payload)
    return {"saved": path, "summary": [s["summary"] for s in payload["stages"]]}


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 16,
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
)
def run_auto():
    smoke = run_smoke.local()
    full = run_full.local()
    return {"smoke": smoke, "full": full}


@app.local_entrypoint()
def main(mode: str = "auto"):
    if mode == "smoke":
        out = run_smoke.remote()
    elif mode == "full":
        out = run_full.remote()
    else:
        out = run_auto.remote()
    print(json.dumps(out, indent=2))
