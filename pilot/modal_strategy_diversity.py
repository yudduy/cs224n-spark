"""
Strategy Diversity Smoke Test (Phase 1)
=======================================
Generate K=8 rollouts per problem for 10 MATH Level 4-5 problems.
Extract structural features from reasoning traces.
Cluster by features to validate strategy diversity is measurable.

Usage:
  modal run pilot/modal_strategy_diversity.py
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from typing import Optional

import modal

app = modal.App("spark-strategy-diversity")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.4")
    .pip_install(
        "transformers>=4.45.0,<5.0.0",
        "datasets",
        "numpy",
        "scikit-learn",
        "huggingface_hub",
    )
)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("spark-pilot-results", create_if_missing=True)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

CFG = {
    "n_problems": 10,
    "level": 5,
    "seed": 42,
    "K": 8,
    "max_new_tokens": 2048,
    "temperature": 1.0,
}


# ---- helpers ----

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
        out.append({
            "problem": r["problem"],
            "answer": r.get("answer", ""),
            "subject": r.get("type", r.get("subject", "unknown")),
        })
    return out


def _problem_prompt(tokenizer, problem: str) -> str:
    instruction = "Let's think step by step and put your final answer within \\boxed{}."
    msgs = [{"role": "user", "content": f"{problem} {instruction}"}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def _extract_boxed_answer(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _verify_answer(predicted: Optional[str], ground_truth: str) -> bool:
    if predicted is None:
        return False
    p = predicted.strip().replace(" ", "").replace(",", "")
    g = ground_truth.strip().replace(" ", "").replace(",", "")
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


# ---- strategy feature extraction ----

# Mathematical operation markers and their patterns
OPERATION_PATTERNS = {
    "substitution": [
        r"(?i)substitut",
        r"(?i)let\s+\w+\s*=",
        r"(?i)plug\s*(in|ging)",
        r"(?i)replace\s+\w+\s+with",
    ],
    "factoring": [
        r"(?i)factor",
        r"\([^)]*\)\s*\([^)]*\)\s*=\s*0",
        r"(?i)roots?\s+of",
    ],
    "quadratic_formula": [
        r"(?i)quadratic\s+formula",
        r"\\frac\{-b\s*\\pm",
        r"(?i)discriminant",
    ],
    "contradiction": [
        r"(?i)contradiction",
        r"(?i)suppose\s+(not|the\s+contrary|for\s+contradiction)",
        r"(?i)assume\s+.*\s+false",
        r"(?i)leads?\s+to\s+a?\s*contradiction",
    ],
    "induction": [
        r"(?i)induction",
        r"(?i)base\s+case",
        r"(?i)inductive\s+(step|hypothesis)",
    ],
    "case_analysis": [
        r"(?i)case\s+[1-9IViv]",
        r"(?i)consider\s+the\s+case",
        r"(?i)if\s+.*\s+then\s+.*\s+otherwise",
        r"(?i)there\s+are\s+(two|three|four)\s+cases",
    ],
    "coordinate_geometry": [
        r"(?i)coordinat",
        r"\(\s*x\s*,\s*y\s*\)",
        r"(?i)distance\s+formula",
        r"(?i)slope\s*(=|of)",
        r"(?i)midpoint",
    ],
    "trigonometry": [
        r"(?i)\\?(sin|cos|tan|cot|sec|csc)\s*[({\\]",
        r"(?i)trigonometric",
        r"(?i)law\s+of\s+(sines|cosines)",
    ],
    "modular_arithmetic": [
        r"(?i)mod\s+\d+",
        r"\\pmod",
        r"(?i)modular",
        r"(?i)congruent?\s+to",
        r"(?i)remainder\s+(when|is)",
    ],
    "combinatorics": [
        r"(?i)\\binom",
        r"(?i)choose\s+\d+",
        r"(?i)combination",
        r"(?i)permutation",
        r"(?i)ways?\s+to\s+(choose|select|arrange)",
    ],
    "inequality": [
        r"(?i)AM-GM",
        r"(?i)Cauchy-Schwarz",
        r"(?i)Jensen",
        r"(?i)triangle\s+inequality",
        r"(?i)inequality",
    ],
    "generating_function": [
        r"(?i)generating\s+function",
        r"(?i)power\s+series",
    ],
    "direct_computation": [
        r"(?i)comput(e|ing|ation)",
        r"(?i)calculat(e|ing)",
        r"(?i)evaluat(e|ing)",
        r"(?i)simplif(y|ying)",
    ],
    "symmetry": [
        r"(?i)symmetr(y|ic)",
        r"(?i)WLOG",
        r"(?i)without\s+loss\s+of\s+generality",
    ],
    "pigeonhole": [
        r"(?i)pigeonhole",
        r"(?i)drawer\s+principle",
    ],
    "recursion": [
        r"(?i)recursi(on|ve)",
        r"(?i)recurrence",
        r"a_\{?n\}?\s*=.*a_\{?n-1\}?",
    ],
}

# Structural markers
STRUCTURE_PATTERNS = {
    "backtracking": [
        r"(?i)(wait|actually|no,?\s+(that|this)|let me reconsider|I made an error|mistake|wrong|hmm)",
        r"(?i)going\s+back",
        r"(?i)start\s+over",
        r"(?i)try\s+(a\s+)?different",
    ],
    "verification": [
        r"(?i)(let's?\s+)?check",
        r"(?i)verif(y|ication)",
        r"(?i)confirm",
        r"(?i)sanity\s+check",
        r"(?i)plug\s*(back|it)\s*(in|into)",
    ],
    "subgoal_decomposition": [
        r"(?i)(first|step\s+[1-9]|next),?\s+(we\s+)?(need|find|compute|determine|show)",
        r"(?i)break\s+(this|it|the\s+problem)\s+down",
        r"(?i)to\s+do\s+this,?\s+we",
    ],
    "lemma_citation": [
        r"(?i)(by|using|from)\s+(the\s+)?(theorem|lemma|corollary|result|formula|identity)",
        r"(?i)Euler|Fermat|Lagrange|Vieta|Bezout",
    ],
}


def extract_strategy_features(text: str) -> dict:
    """Extract structural features from a reasoning trace."""
    features = {}

    # 1. Operation types detected
    ops_found = []
    ops_counts = {}
    for op_name, patterns in OPERATION_PATTERNS.items():
        count = 0
        for pat in patterns:
            count += len(re.findall(pat, text))
        if count > 0:
            ops_found.append(op_name)
            ops_counts[op_name] = count
    features["operations"] = sorted(ops_found)
    features["operation_counts"] = ops_counts
    features["n_operations"] = len(ops_found)

    # 2. Structural markers
    struct_found = []
    struct_counts = {}
    for s_name, patterns in STRUCTURE_PATTERNS.items():
        count = 0
        for pat in patterns:
            count += len(re.findall(pat, text))
        if count > 0:
            struct_found.append(s_name)
            struct_counts[s_name] = count
    features["structures"] = sorted(struct_found)
    features["structure_counts"] = struct_counts

    # 3. Trace length metrics
    features["total_chars"] = len(text)
    features["total_words"] = len(text.split())
    # Count "thinking" tokens (DeepSeek-R1 uses <think>...</think>)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1)
        features["think_words"] = len(think_text.split())
        features["think_ratio"] = features["think_words"] / max(features["total_words"], 1)
    else:
        features["think_words"] = 0
        features["think_ratio"] = 0.0

    # 4. Number of distinct math expressions (LaTeX)
    latex_exprs = re.findall(r"\$[^$]+\$|\\[a-zA-Z]+\{[^}]*\}", text)
    features["n_latex_exprs"] = len(latex_exprs)

    # 5. Self-correction count (backtracking instances)
    backtrack_count = 0
    for pat in STRUCTURE_PATTERNS["backtracking"]:
        backtrack_count += len(re.findall(pat, text))
    features["n_backtracks"] = backtrack_count

    # 6. Verification steps
    verify_count = 0
    for pat in STRUCTURE_PATTERNS["verification"]:
        verify_count += len(re.findall(pat, text))
    features["n_verifications"] = verify_count

    # 7. Number of subgoals (rough proxy: count "step N", "first/next/then we")
    subgoal_markers = re.findall(
        r"(?i)(step\s+\d+|first,?\s+we|next,?\s+we|then,?\s+we|finally,?\s+we)", text
    )
    features["n_subgoals"] = len(subgoal_markers)

    # 8. Primary operation (highest count, excluding direct_computation)
    non_trivial_ops = {
        k: v for k, v in ops_counts.items() if k != "direct_computation"
    }
    if non_trivial_ops:
        features["primary_operation"] = max(non_trivial_ops, key=non_trivial_ops.get)
    elif "direct_computation" in ops_counts:
        features["primary_operation"] = "direct_computation"
    else:
        features["primary_operation"] = "unclassified"

    return features


def compute_strategy_fingerprint(features: dict) -> str:
    """Create a hashable fingerprint from the most salient strategy features.
    Two traces get the same fingerprint if they use the same primary operation
    and the same set of mathematical operations (ignoring counts)."""
    primary = features.get("primary_operation", "unclassified")
    ops = tuple(features.get("operations", []))
    has_backtrack = features.get("n_backtracks", 0) > 0
    has_verify = features.get("n_verifications", 0) > 0
    return f"{primary}|{ops}|bt={has_backtrack}|vf={has_verify}"


def cluster_by_strategy(rollout_features: list[dict]) -> dict:
    """Cluster rollouts by strategy fingerprint. Returns cluster info."""
    fingerprints = []
    for rf in rollout_features:
        fp = compute_strategy_fingerprint(rf)
        fingerprints.append(fp)

    # Group by fingerprint
    clusters = {}
    for idx, fp in enumerate(fingerprints):
        if fp not in clusters:
            clusters[fp] = []
        clusters[fp].append(idx)

    # Also try coarser clustering: just primary operation
    coarse_clusters = {}
    for idx, rf in enumerate(rollout_features):
        primary = rf.get("primary_operation", "unclassified")
        if primary not in coarse_clusters:
            coarse_clusters[primary] = []
        coarse_clusters[primary].append(idx)

    return {
        "n_fine_clusters": len(clusters),
        "n_coarse_clusters": len(coarse_clusters),
        "fine_clusters": {k: v for k, v in clusters.items()},
        "coarse_clusters": {k: v for k, v in coarse_clusters.items()},
        "fingerprints": fingerprints,
        "fine_distribution": {k: len(v) for k, v in clusters.items()},
        "coarse_distribution": {k: len(v) for k, v in coarse_clusters.items()},
    }


def compute_diversity_metrics(cluster_info: dict, n_rollouts: int) -> dict:
    """Compute diversity metrics from clustering."""
    # Shannon entropy of fine cluster distribution
    fine_sizes = list(cluster_info["fine_distribution"].values())
    total = sum(fine_sizes)
    if total > 0:
        probs = [s / total for s in fine_sizes]
        fine_entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        max_entropy = math.log(n_rollouts)
        normalized_entropy = fine_entropy / max_entropy if max_entropy > 0 else 0
    else:
        fine_entropy = 0
        normalized_entropy = 0

    # Simpson's diversity index (1 - sum(p_i^2))
    if total > 0:
        probs = [s / total for s in fine_sizes]
        simpson = 1 - sum(p ** 2 for p in probs)
    else:
        simpson = 0

    # Coarse entropy
    coarse_sizes = list(cluster_info["coarse_distribution"].values())
    total_c = sum(coarse_sizes)
    if total_c > 0:
        probs_c = [s / total_c for s in coarse_sizes]
        coarse_entropy = -sum(p * math.log(p + 1e-10) for p in probs_c if p > 0)
    else:
        coarse_entropy = 0

    return {
        "n_fine_clusters": cluster_info["n_fine_clusters"],
        "n_coarse_clusters": cluster_info["n_coarse_clusters"],
        "fine_entropy": round(fine_entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "coarse_entropy": round(coarse_entropy, 4),
        "simpson_diversity": round(simpson, 4),
    }


# ---- main runner ----

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 4,
    volumes={"/hf-cache": hf_cache, "/results": results_vol},
)
def run_strategy_diversity():
    from vllm import LLM, SamplingParams

    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"

    print("=" * 60)
    print("STRATEGY DIVERSITY SMOKE TEST - Phase 1")
    print("=" * 60)

    # 1. Load model
    print(f"\n[1/5] Loading model: {MODEL_ID}")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    print("  Model loaded.")

    # 2. Load problems
    print(f"\n[2/5] Loading {CFG['n_problems']} MATH Level {CFG['level']} problems")
    problems = _load_math_problems(CFG["n_problems"], CFG["level"], CFG["seed"])
    prompts = [_problem_prompt(tokenizer, p["problem"]) for p in problems]
    print(f"  Loaded {len(problems)} problems")

    # 3. Generate K rollouts per problem
    print(f"\n[3/5] Generating K={CFG['K']} rollouts per problem")
    params = SamplingParams(
        temperature=CFG["temperature"],
        top_p=1.0,
        max_tokens=CFG["max_new_tokens"],
        n=CFG["K"],
    )
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    gen_time = time.time() - t0
    print(f"  Generated {len(problems) * CFG['K']} rollouts in {gen_time:.1f}s")

    # 4. Extract features and cluster
    print(f"\n[4/5] Extracting strategy features and clustering")
    all_results = []

    for prob_idx, (output_obj, problem) in enumerate(zip(outputs, problems)):
        prob_rollouts = []
        prob_features = []

        for roll_idx, completion in enumerate(output_obj.outputs):
            text = completion.text
            final_answer = _extract_boxed_answer(text)
            is_correct = _verify_answer(final_answer, problem["answer"])
            features = extract_strategy_features(text)

            rollout = {
                "rollout_idx": roll_idx,
                "response": text,
                "final_answer": final_answer,
                "is_correct": is_correct,
                "features": features,
            }
            prob_rollouts.append(rollout)
            prob_features.append(features)

        # Cluster this problem's rollouts
        cluster_info = cluster_by_strategy(prob_features)
        diversity = compute_diversity_metrics(cluster_info, CFG["K"])

        # Answer-level metrics
        answers = [r["final_answer"] for r in prob_rollouts]
        correct_count = sum(1 for r in prob_rollouts if r["is_correct"])
        unique_answers = len(set(a for a in answers if a is not None))
        pass_rate = correct_count / CFG["K"]

        prob_result = {
            "problem_idx": prob_idx,
            "problem": problem["problem"],
            "answer_gt": problem["answer"],
            "subject": problem.get("subject", "unknown"),
            "n_correct": correct_count,
            "pass_rate": pass_rate,
            "n_unique_answers": unique_answers,
            "diversity_metrics": diversity,
            "cluster_info": {
                "fine_distribution": cluster_info["fine_distribution"],
                "coarse_distribution": cluster_info["coarse_distribution"],
                "fingerprints": cluster_info["fingerprints"],
            },
            "rollouts": prob_rollouts,
        }
        all_results.append(prob_result)

        # Print summary for this problem
        print(f"\n  Problem {prob_idx}: {problem['problem'][:80]}...")
        print(f"    Subject: {problem.get('subject', 'unknown')}")
        print(f"    Pass rate: {correct_count}/{CFG['K']} ({pass_rate:.1%})")
        print(f"    Unique answers: {unique_answers}")
        print(f"    Fine clusters: {diversity['n_fine_clusters']}")
        print(f"    Coarse clusters: {diversity['n_coarse_clusters']}")
        print(f"    Simpson diversity: {diversity['simpson_diversity']:.3f}")
        print(f"    Fine distribution: {cluster_info['fine_distribution']}")
        print(f"    Coarse distribution: {cluster_info['coarse_distribution']}")

    # 5. Aggregate summary
    print(f"\n[5/5] Computing aggregate statistics")

    fine_counts = [r["diversity_metrics"]["n_fine_clusters"] for r in all_results]
    coarse_counts = [r["diversity_metrics"]["n_coarse_clusters"] for r in all_results]
    simpson_vals = [r["diversity_metrics"]["simpson_diversity"] for r in all_results]
    pass_rates = [r["pass_rate"] for r in all_results]

    summary = {
        "config": CFG,
        "model": MODEL_ID,
        "timestamp": int(time.time()),
        "generation_time_sec": round(gen_time, 1),
        "n_problems": len(all_results),
        "aggregate": {
            "mean_fine_clusters": round(sum(fine_counts) / len(fine_counts), 2),
            "median_fine_clusters": sorted(fine_counts)[len(fine_counts) // 2],
            "min_fine_clusters": min(fine_counts),
            "max_fine_clusters": max(fine_counts),
            "mean_coarse_clusters": round(sum(coarse_counts) / len(coarse_counts), 2),
            "mean_simpson_diversity": round(sum(simpson_vals) / len(simpson_vals), 3),
            "mean_pass_rate": round(sum(pass_rates) / len(pass_rates), 3),
            "problems_with_3plus_fine_clusters": sum(1 for c in fine_counts if c >= 3),
            "problems_with_3plus_coarse_clusters": sum(1 for c in coarse_counts if c >= 3),
            "pct_3plus_fine": round(
                sum(1 for c in fine_counts if c >= 3) / len(fine_counts), 2
            ),
        },
    }

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"  Mean fine clusters per problem:   {summary['aggregate']['mean_fine_clusters']}")
    print(f"  Median fine clusters:             {summary['aggregate']['median_fine_clusters']}")
    print(f"  Range:                            [{summary['aggregate']['min_fine_clusters']}, {summary['aggregate']['max_fine_clusters']}]")
    print(f"  Mean coarse clusters:             {summary['aggregate']['mean_coarse_clusters']}")
    print(f"  Mean Simpson diversity:           {summary['aggregate']['mean_simpson_diversity']}")
    print(f"  Mean pass rate:                   {summary['aggregate']['mean_pass_rate']}")
    print(f"  Problems with ≥3 fine clusters:   {summary['aggregate']['problems_with_3plus_fine_clusters']}/{len(all_results)} ({summary['aggregate']['pct_3plus_fine']:.0%})")
    print(f"  Problems with ≥3 coarse clusters: {summary['aggregate']['problems_with_3plus_coarse_clusters']}/{len(all_results)}")

    # Success criterion check
    pct_3plus = summary["aggregate"]["pct_3plus_fine"]
    print(f"\n  SUCCESS CRITERION: ≥60% problems with ≥3 fine clusters")
    print(f"  RESULT: {pct_3plus:.0%} {'PASS' if pct_3plus >= 0.6 else 'FAIL'}")

    # Save results
    out_dir = "/results/strategy_diversity"
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())

    # Full results (with rollout text)
    full_path = os.path.join(out_dir, f"full_results_{ts}.json")
    with open(full_path, "w") as f:
        json.dump({"summary": summary, "problems": all_results}, f, indent=2)

    # Summary only (lightweight)
    summary_path = os.path.join(out_dir, f"summary_{ts}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    results_vol.commit()

    print(f"\n  Saved full results to {full_path}")
    print(f"  Saved summary to {summary_path}")

    return {"summary": summary, "full_path": full_path}


@app.local_entrypoint()
def main():
    result = run_strategy_diversity.remote()
    summary = result["summary"]

    print("\n" + "=" * 60)
    print("LOCAL SUMMARY")
    print("=" * 60)
    agg = summary["aggregate"]
    print(f"  Fine clusters:  {agg['mean_fine_clusters']} avg, [{agg['min_fine_clusters']}-{agg['max_fine_clusters']}] range")
    print(f"  Coarse clusters: {agg['mean_coarse_clusters']} avg")
    print(f"  Simpson diversity: {agg['mean_simpson_diversity']}")
    print(f"  Pass rate: {agg['mean_pass_rate']}")
    print(f"  ≥3 fine clusters: {agg['problems_with_3plus_fine_clusters']}/{summary['n_problems']}")
    print(f"  Verdict: {'PASS - proceed to Phase 2' if agg['pct_3plus_fine'] >= 0.6 else 'FAIL - structural features insufficient'}")
