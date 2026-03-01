#!/usr/bin/env python3
"""
Analyze archive-guided pilot results.
Step 1: LLM judge classifies new traces against baseline strategies.
Step 2: Compare archive-guided vs baseline strategy coverage.
Step 3: GO/NO-GO verdict against 3 success criteria.

Usage:
  # First download results from Modal volume:
  modal volume get spark-pilot-results archive_pilot/ data/modal_runs/archive_pilot/

  # Then run analysis:
  OPENAI_API_KEY=sk-... python3 pilot/analyze_archive_pilot.py
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────

ARCHIVE_PATH = Path("data/analysis/strategy_archives.json")
JUDGE_PATH = Path("data/analysis/llm_judge_pilot.json")
TRACES_DIR = Path("data/modal_runs/archive_pilot/full")
OUT_PATH = Path("data/analysis/archive_pilot_results.json")

MODELS = ["r1-distill", "nemotron-v2"]
MODEL_LABELS = {"r1-distill": "R1-Distill", "nemotron-v2": "Nemotron v2"}

MODEL_NAME = "gpt-5-nano"
SAMPLE_PER_MODEL = 8  # max correct traces to send per model per iteration


# ── Helpers ────────────────────────────────────────────────────


def load_archive_traces():
    """Load archive-guided traces for both models."""
    traces = {}
    for m in MODELS:
        path = TRACES_DIR / m / "traces.json"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {m}")
            continue
        with open(path) as f:
            traces[m] = json.load(f)
    return traces


def collect_correct_traces(problem_data: dict, max_per_iter: int = SAMPLE_PER_MODEL) -> list[dict]:
    """Collect correct traces across all iterations, sampling evenly."""
    correct = []
    for it_data in problem_data["iterations"]:
        it_correct = [r for r in it_data["rollouts"] if r["is_correct"]]
        n = min(max_per_iter, len(it_correct))
        if n > 0:
            step = max(1, len(it_correct) // n)
            sampled = it_correct[::step][:n]
            for r in sampled:
                correct.append({
                    "iter": it_data["iter"],
                    "response": r["response"],
                    "final_answer": r["final_answer"],
                })
    return correct


def build_judge_prompt(problem_text: str, answer: str,
                       baseline_strategies: list[dict],
                       model_traces: dict[str, list[dict]]) -> str:
    """Build LLM judge prompt for classifying archive-guided traces."""

    # Baseline strategies section
    strat_lines = []
    for s in baseline_strategies:
        strat_lines.append(f"  Strategy {s['id']}: {s['name']} — {s['description']}")
    strat_block = "\n".join(strat_lines)

    # New traces section
    traces_block = []
    trace_ids = []
    for model, traces in model_traces.items():
        label = MODEL_LABELS[model]
        for i, t in enumerate(traces):
            tid = f"{label[:2]}_iter{t['iter']}_{i+1}"
            trace_ids.append(tid)
            text = t["response"][:3000]
            if len(t["response"]) > 3000:
                text += "\n[...truncated...]"
            traces_block.append(f"\n--- Trace {tid} ({label}, iteration {t['iter']}) ---\n{text}\n")

    traces_str = "\n".join(traces_block)

    prompt = f"""You are analyzing mathematical reasoning traces from archive-guided generation.
Models were shown a list of known strategies and asked to find FUNDAMENTALLY DIFFERENT approaches.

Your task: determine if any new traces use a strategy NOT in the baseline list.

IMPORTANT:
- A "strategy" is a fundamentally different MATHEMATICAL APPROACH
- Minor variations, different verbosity, or rephrasing = SAME strategy
- Only flag as "novel" if the core mathematical method is genuinely different
- If a trace claims to use a different approach but actually uses the same math = NOT novel

Problem: {problem_text}

Correct answer: {answer}

BASELINE STRATEGIES (from 64 rollouts per model):
{strat_block}

NEW ARCHIVE-GUIDED TRACES:
{traces_str}

Respond in this exact JSON format:
{{
  "classifications": {{
    "<trace_id>": {{"strategy_id": <int or "novel">, "strategy_name": "<name>"}},
    ...
  }},
  "novel_strategies": [
    {{"name": "<short name>", "description": "<1-2 sentence description>"}},
    ...
  ],
  "n_novel": <int>,
  "notes": "<observations about whether archive prompting produced genuinely different approaches>"
}}

If a trace uses a novel strategy, set strategy_id to "novel" and give it a descriptive name.
If all traces use existing baseline strategies, set novel_strategies to [] and n_novel to 0."""

    return prompt, trace_ids


# ── Main analysis ──────────────────────────────────────────────


def run_analysis():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load data
    with open(ARCHIVE_PATH) as f:
        archives = json.load(f)
    with open(JUDGE_PATH) as f:
        baseline_judge = json.load(f)

    baseline_by_pid = {r["problem_id"]: r for r in baseline_judge}
    archive_traces = load_archive_traces()

    if not archive_traces:
        print("ERROR: No archive traces found. Run modal_archive_pilot.py first.")
        sys.exit(1)

    pilot_pids = sorted(int(k) for k in archives.keys())

    results = []
    total_input = 0
    total_output = 0

    print(f"\n{'='*60}")
    print(f"ARCHIVE PILOT ANALYSIS")
    print(f"  {len(pilot_pids)} problems, {len(archive_traces)} models")
    print(f"{'='*60}")

    for pid in pilot_pids:
        arch = archives[str(pid)]
        baseline = baseline_by_pid.get(pid, {})
        baseline_resp = baseline.get("llm_response", {})
        baseline_strats = baseline_resp.get("strategies", [])
        n_baseline = baseline_resp.get("n_strategies", 0)

        print(f"\nP{pid:2d} [{arch['tier']:6s}]: {n_baseline} baseline strategies")

        # Collect correct archive-guided traces per model
        model_traces = {}
        for m in MODELS:
            if m not in archive_traces:
                continue
            # Find this problem in the model's results
            prob_data = None
            for p in archive_traces[m]["problems"]:
                if p["problem_id"] == pid:
                    prob_data = p
                    break
            if prob_data is None:
                continue
            correct = collect_correct_traces(prob_data)
            if correct:
                model_traces[m] = correct
                print(f"  {MODEL_LABELS[m]}: {len(correct)} correct archive traces")

        if not model_traces:
            print(f"  No correct archive traces — skipping")
            results.append({
                "problem_id": pid,
                "tier": arch["tier"],
                "n_baseline_strategies": n_baseline,
                "n_novel": 0,
                "skipped": True,
                "reason": "no correct archive traces",
            })
            continue

        # Ask LLM judge
        prompt, trace_ids = build_judge_prompt(
            arch["problem_text"], arch["answer"],
            baseline_strats, model_traces
        )

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            text = resp.choices[0].message.content
            usage = resp.usage
            total_input += usage.prompt_tokens
            total_output += usage.completion_tokens

            parsed = json.loads(text)
            n_novel = parsed.get("n_novel", 0)
            novel_strats = parsed.get("novel_strategies", [])
            notes = parsed.get("notes", "")

            # Per-model strategy analysis
            classifications = parsed.get("classifications", {})
            per_model = {}
            for m in MODELS:
                label_prefix = MODEL_LABELS[m][:2]
                model_strat_ids = set()
                model_novel = 0
                for tid, cls in classifications.items():
                    if tid.startswith(label_prefix):
                        sid = cls.get("strategy_id", "?")
                        if sid == "novel":
                            model_novel += 1
                        else:
                            model_strat_ids.add(sid)
                per_model[m] = {
                    "baseline_strategies_used": sorted(model_strat_ids),
                    "n_novel": model_novel,
                }

            print(f"  => {n_novel} novel strategies")
            for ns in novel_strats:
                print(f"     NOVEL: {ns['name']} — {ns['description']}")
            for m in MODELS:
                pm = per_model[m]
                print(f"     {MODEL_LABELS[m]}: baseline={pm['baseline_strategies_used']} novel={pm['n_novel']}")
            if notes:
                print(f"  Notes: {notes[:200]}")

            results.append({
                "problem_id": pid,
                "tier": arch["tier"],
                "n_baseline_strategies": n_baseline,
                "n_novel": n_novel,
                "novel_strategies": novel_strats,
                "per_model": per_model,
                "classifications": classifications,
                "notes": notes,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "problem_id": pid,
                "tier": arch["tier"],
                "n_baseline_strategies": n_baseline,
                "error": str(e),
            })

        time.sleep(0.5)

    # ── Verdict ────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Problem':>8} {'Tier':>7} {'Base':>5} {'R1+Arch':>8} {'v2+Arch':>8} {'Novel':>6}")
    print("-" * 50)

    total_novel_problems = 0
    model_diff_problems = 0
    any_novel_over_baseline = 0

    for r in results:
        pid = r["problem_id"]
        tier = r["tier"]
        n_base = r["n_baseline_strategies"]
        n_novel = r.get("n_novel", 0)

        pm = r.get("per_model", {})
        r1_info = pm.get("r1-distill", {})
        v2_info = pm.get("nemotron-v2", {})

        r1_base = len(r1_info.get("baseline_strategies_used", []))
        r1_novel = r1_info.get("n_novel", 0)
        v2_base = len(v2_info.get("baseline_strategies_used", []))
        v2_novel = v2_info.get("n_novel", 0)

        r1_str = f"{r1_base}+{r1_novel}n" if r1_novel else str(r1_base)
        v2_str = f"{v2_base}+{v2_novel}n" if v2_novel else str(v2_base)

        if r.get("skipped"):
            r1_str = v2_str = "-"

        print(f"  P{pid:<5d} {tier:>7} {n_base:>5} {r1_str:>8} {v2_str:>8} {n_novel:>6}")

        if n_novel > 0:
            total_novel_problems += 1
            any_novel_over_baseline += 1

        # Check if models find different strategies
        r1_all = set(r1_info.get("baseline_strategies_used", []))
        v2_all = set(v2_info.get("baseline_strategies_used", []))
        if r1_novel:
            r1_all.add("novel_r1")
        if v2_novel:
            v2_all.add("novel_v2")
        if r1_all != v2_all and r1_all and v2_all:
            model_diff_problems += 1

    # Criteria
    c1 = total_novel_problems >= 3
    c2 = model_diff_problems >= 2
    c3 = any_novel_over_baseline >= 1

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"  Criterion 1: ≥3/10 problems with novel strategy via archive")
    print(f"    Result: {total_novel_problems}/10 — {'PASS' if c1 else 'FAIL'}")
    print(f"  Criterion 2: R1 and v2 find different strategies on ≥2 problems")
    print(f"    Result: {model_diff_problems}/10 — {'PASS' if c2 else 'FAIL'}")
    print(f"  Criterion 3: Archive finds ≥1 strategy 64 baseline rollouts missed")
    print(f"    Result: {any_novel_over_baseline} — {'PASS' if c3 else 'FAIL'}")

    verdict = "GO" if (c1 and c3) else "NO-GO"
    print(f"\n  {'='*20}")
    print(f"  VERDICT: {verdict}")
    print(f"  {'='*20}")

    # Token usage
    cost = total_input * 0.05 / 1e6 + total_output * 0.40 / 1e6
    print(f"\nAPI usage: {total_input} input + {total_output} output tokens")
    print(f"Estimated cost: ${cost:.4f}")

    # Save
    output = {
        "results": results,
        "summary": {
            "total_novel_problems": total_novel_problems,
            "model_diff_problems": model_diff_problems,
            "any_novel_over_baseline": any_novel_over_baseline,
            "criteria": {"c1": c1, "c2": c2, "c3": c3},
            "verdict": verdict,
        },
        "api_usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": round(cost, 4),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    run_analysis()
