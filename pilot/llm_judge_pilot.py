#!/usr/bin/env python3
"""
LLM Judge Pilot: Strategy classification on 10 selected problems.
Uses GPT-5-nano to determine if genuine strategy diversity exists at 1.5B.

Usage:
  OPENAI_API_KEY=sk-... python3 pilot/llm_judge_pilot.py

Selected problems (10):
  Easy:  1 (both solve, high CV), 5 (R1 answer divergence)
  Medium: 23 (R1 divergent), 37 (R1 much better than v1)
  Hard jointly-solved: 40 (v2>>R1), 43 (both 59+ correct), 59 (R1>>v2!)
  Hard v2-only: 42 (R1=0, v2=9), 51 (R1=0, v2=29)
  Hard R1-only: 57 (R1=2, v2=0)
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

DATA_DIR = Path("data/modal_runs/gen_traces_full")
OUT_DIR = Path("data/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["r1-distill", "nemotron-v1", "nemotron-v2", "nemotron-brorl"]
MODEL_LABELS = {
    "r1-distill": "A (R1-Distill, base)",
    "nemotron-v1": "B (v1, 2K RL steps)",
    "nemotron-v2": "C (v2, 3K RL steps)",
    "nemotron-brorl": "D (BroRL, breadth RL)",
}
SAMPLE_PER_MODEL = 8  # max correct traces to send per model
SELECTED_PROBLEMS = list(range(60))  # All 60 problems

MODEL_NAME = "gpt-5-nano"


def load_data():
    with open(DATA_DIR / "problems.json") as f:
        problems = json.load(f)
    all_data = {}
    for m in MODELS:
        with open(DATA_DIR / m / "traces.json") as f:
            all_data[m] = json.load(f)
    return problems, all_data


def build_prompt(problem, all_data, pid):
    """Build the classification prompt for a single problem."""
    prob_text = problem["problem"]
    answer = problem.get("answer", "unknown")

    # Collect correct traces per model (sample up to SAMPLE_PER_MODEL)
    traces_block = []
    trace_labels = []
    for m in MODELS:
        label_prefix = MODEL_LABELS[m].split(" ")[0]  # A, B, C, D
        p_data = all_data[m]["problems"][pid]
        correct_rollouts = [r for r in p_data["rollouts"] if r["is_correct"]]

        if not correct_rollouts:
            traces_block.append(f"\n[Model {MODEL_LABELS[m]}]: No correct solutions.\n")
            continue

        # Sample evenly
        n = min(SAMPLE_PER_MODEL, len(correct_rollouts))
        step = max(1, len(correct_rollouts) // n)
        sampled = correct_rollouts[::step][:n]

        for i, r in enumerate(sampled):
            tid = f"{label_prefix}{i+1}"
            trace_labels.append(tid)
            # Truncate very long traces to save tokens
            text = r["response"][:3000]
            if len(r["response"]) > 3000:
                text += "\n[...truncated...]"
            traces_block.append(f"\n--- Trace {tid} (Model {MODEL_LABELS[m]}) ---\n{text}\n")

    traces_str = "\n".join(traces_block)

    prompt = f"""You are analyzing mathematical reasoning traces from 4 AI model checkpoints on the SAME training trajectory.
Your task: identify how many DISTINCT high-level solution strategies are used across all traces.

IMPORTANT:
- A "strategy" is a fundamentally different MATHEMATICAL APPROACH (e.g., algebraic vs geometric, direct computation vs proof by contradiction, substitution vs factoring)
- Differences in verbosity, formatting, step ordering, or phrasing are NOT different strategies
- Two traces that use the same approach but one is more detailed or rambles more = SAME strategy
- Focus on the CORE mathematical method, not surface presentation

Problem: {prob_text}

Correct answer: {answer}

{traces_str}

Respond in this exact JSON format:
{{
  "n_strategies": <int>,
  "strategies": [
    {{"id": 1, "name": "<short name>", "description": "<1-2 sentence description of the mathematical approach>"}},
    ...
  ],
  "classifications": {{
    "<trace_id>": <strategy_id>,
    ...
  }},
  "confidence": "<high|medium|low>",
  "notes": "<any observations about strategy diversity or lack thereof>"
}}"""

    return prompt, trace_labels


def run_pilot():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    problems, all_data = load_data()

    results = []
    total_input = 0
    total_output = 0

    for pid in SELECTED_PROBLEMS:
        prob = problems[pid]
        print(f"\n{'='*60}")
        print(f"Problem {pid} [{prob['tier']}/{prob['subject']}]")

        # Show per-model correct counts
        for m in MODELS:
            nc = all_data[m]["problems"][pid]["n_correct"]
            print(f"  {MODEL_LABELS[m]}: {nc}/64 correct")

        prompt, trace_labels = build_prompt(prob, all_data, pid)
        prompt_tokens = len(prompt) // 4  # rough estimate
        print(f"  Prompt ~{prompt_tokens} tokens, {len(trace_labels)} traces")

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
            n_strat = parsed.get("n_strategies", "?")
            confidence = parsed.get("confidence", "?")
            notes = parsed.get("notes", "")

            print(f"  => {n_strat} strategies (confidence: {confidence})")
            for s in parsed.get("strategies", []):
                print(f"     Strategy {s['id']}: {s['name']} — {s['description']}")

            # Count per model
            classifications = parsed.get("classifications", {})
            for m in MODELS:
                prefix = MODEL_LABELS[m].split(" ")[0]
                model_strats = set()
                for tid, sid in classifications.items():
                    if tid.startswith(prefix):
                        model_strats.add(sid)
                if model_strats:
                    print(f"     {MODEL_LABELS[m]}: uses strategies {sorted(model_strats)}")

            if notes:
                print(f"  Notes: {notes}")

            results.append({
                "problem_id": pid,
                "tier": prob["tier"],
                "subject": prob["subject"],
                "llm_response": parsed,
                "trace_labels": trace_labels,
                "per_model_correct": {
                    m: all_data[m]["problems"][pid]["n_correct"] for m in MODELS
                },
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "problem_id": pid,
                "error": str(e),
            })

        time.sleep(0.5)  # rate limit courtesy

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total API usage: {total_input} input + {total_output} output tokens")
    cost = total_input * 0.05 / 1e6 + total_output * 0.40 / 1e6
    print(f"Estimated cost: ${cost:.4f}")

    # Aggregate
    strat_counts = []
    for r in results:
        if "llm_response" in r:
            n = r["llm_response"].get("n_strategies", 0)
            strat_counts.append(n)
            print(f"  P{r['problem_id']:2d} [{r['tier']:6s}]: {n} strategies")

    if strat_counts:
        print(f"\nMean strategies per problem: {sum(strat_counts)/len(strat_counts):.1f}")
        print(f"Problems with >1 strategy: {sum(1 for s in strat_counts if s > 1)}/{len(strat_counts)}")

    # Save
    with open(OUT_DIR / "llm_judge_pilot.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR / 'llm_judge_pilot.json'}")


if __name__ == "__main__":
    run_pilot()
