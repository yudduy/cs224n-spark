#!/usr/bin/env python3
"""
Build strategy archives from LLM judge data for archive-guided pilot.
Reads baseline judge classifications + raw traces, outputs per-problem
strategy archive with exemplar traces.

Usage:
  python3 pilot/build_archive.py

Output: data/analysis/strategy_archives.json
"""

import json
from pathlib import Path

JUDGE_PATH = Path("data/analysis/llm_judge_pilot.json")
PROBLEMS_PATH = Path("data/modal_runs/gen_traces_full/problems.json")
TRACES_DIR = Path("data/modal_runs/gen_traces_full")
OUT_PATH = Path("data/analysis/strategy_archives.json")

MODELS = ["r1-distill", "nemotron-v1", "nemotron-v2", "nemotron-brorl"]
MODEL_PREFIXES = {"r1-distill": "A", "nemotron-v1": "B", "nemotron-v2": "C", "nemotron-brorl": "D"}

# 10 pilot problems: mix of multi-strategy, narrowing, and v2-only
PILOT_PIDS = [2, 13, 19, 35, 38, 42, 46, 52, 54, 58]

EXEMPLAR_MAX_CHARS = 1500


def main():
    with open(JUDGE_PATH) as f:
        judge_data = json.load(f)
    with open(PROBLEMS_PATH) as f:
        problems = json.load(f)

    # Load traces for all models
    all_traces = {}
    for m in MODELS:
        with open(TRACES_DIR / m / "traces.json") as f:
            all_traces[m] = json.load(f)

    # Index judge data by problem_id
    judge_by_pid = {r["problem_id"]: r for r in judge_data}

    archives = {}
    for pid in PILOT_PIDS:
        prob = problems[pid]
        judge = judge_by_pid[pid]
        resp = judge.get("llm_response", {})
        strategies = resp.get("strategies", [])
        classifications = resp.get("classifications", {})

        # Determine which models use each strategy
        strat_models = {}
        for sid_info in strategies:
            sid = sid_info["id"]
            strat_models[sid] = set()

        for tid, sid in classifications.items():
            prefix = tid[0]  # A, B, C, D
            for m, p in MODEL_PREFIXES.items():
                if p == prefix:
                    if sid in strat_models:
                        strat_models[sid].add(p)

        # Find one correct exemplar trace per strategy
        strat_exemplars = {}
        for tid, sid in classifications.items():
            if sid in strat_exemplars:
                continue
            prefix = tid[0]
            idx = int(tid[1:]) - 1  # A1 -> 0, A2 -> 1, etc.
            for m, p in MODEL_PREFIXES.items():
                if p == prefix:
                    p_data = all_traces[m]["problems"][pid]
                    correct_rollouts = [r for r in p_data["rollouts"] if r["is_correct"]]
                    step = max(1, len(correct_rollouts) // 8)
                    sampled = correct_rollouts[::step][:8]
                    if idx < len(sampled):
                        text = sampled[idx]["response"][:EXEMPLAR_MAX_CHARS]
                        strat_exemplars[sid] = text

        # Build archive entry
        strat_list = []
        for s in strategies:
            sid = s["id"]
            strat_list.append({
                "id": sid,
                "name": s["name"],
                "description": s["description"],
                "used_by": sorted(strat_models.get(sid, set())),
            })

        archives[str(pid)] = {
            "problem_id": pid,
            "problem_text": prob["problem"],
            "answer": prob.get("answer", ""),
            "tier": prob.get("tier", ""),
            "subject": prob.get("subject", ""),
            "n_strategies": resp.get("n_strategies", 0),
            "strategies": strat_list,
            "exemplars": {str(k): v for k, v in strat_exemplars.items()},
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(archives, f, indent=2)

    # Summary
    print(f"Built archives for {len(archives)} problems:")
    for pid_str, arch in sorted(archives.items(), key=lambda x: int(x[0])):
        ns = arch["n_strategies"]
        nexemplars = len(arch["exemplars"])
        print(f"  P{arch['problem_id']:2d} [{arch['tier']:6s}]: {ns} strategies, {nexemplars} exemplars")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
