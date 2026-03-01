#!/usr/bin/env python3
"""
Run 21 Analysis: What Survives Distillation?
=============================================
Compares Student-R1 vs Student-v2 on:
  1. Accuracy (pass@k) — against original teachers and each other
  2. Strategy diversity (LLM judge) — GPT-5-nano classification
  3. Decision table — traces vs model state

Uses the same LLM judge protocol as Run 16 (llm_judge_pilot.py).

Usage:
  OPENAI_API_KEY=sk-... python3 pilot/analyze_distillation.py
  OPENAI_API_KEY=sk-... python3 pilot/analyze_distillation.py --skip-judge  # skip LLM judge, just compare
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────

STUDENT_DIR = Path("data/modal_runs/distill_pilot/full/traces")
TEACHER_DIR = Path("data/modal_runs/gen_traces_full")
BASELINE_JUDGE = Path("data/analysis/llm_judge_pilot.json")
OUT_DIR = Path("data/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDENTS = ["student-r1", "student-v2"]
TEACHERS = ["r1-distill", "nemotron-v2"]  # for comparison
TEACHER_LABELS = {
    "r1-distill": "R1-Distill (teacher)",
    "nemotron-v2": "Nemotron-v2 (teacher)",
}
STUDENT_LABELS = {
    "student-r1": "E (Student-R1, SFT on R1 traces)",
    "student-v2": "F (Student-v2, SFT on v2 traces)",
}

MODEL_NAME = "gpt-5-nano"
SAMPLE_PER_MODEL = 8


# ── Load data ──────────────────────────────────────────────────


def load_student_data():
    problems_path = STUDENT_DIR / "problems.json"
    with open(problems_path) as f:
        problems = json.load(f)
    data = {}
    for s in STUDENTS:
        with open(STUDENT_DIR / s / "traces.json") as f:
            data[s] = json.load(f)
    return problems, data


def load_teacher_data():
    data = {}
    for t in TEACHERS:
        with open(TEACHER_DIR / t / "traces.json") as f:
            data[t] = json.load(f)
    return data


def load_baseline_judge():
    if BASELINE_JUDGE.exists():
        with open(BASELINE_JUDGE) as f:
            return json.load(f)
    return None


# ── LLM Judge ─────────────────────────────────────────────────


def build_judge_prompt(problem: dict, student_data: dict, pid: int) -> tuple:
    """Build classification prompt for student traces."""
    prob_text = problem["problem"]
    answer = problem.get("answer", "unknown")

    traces_block = []
    trace_labels = []

    for s in STUDENTS:
        label_prefix = STUDENT_LABELS[s].split(" ")[0]  # E, F
        p_data = student_data[s]["problems"][pid]
        correct_rollouts = [r for r in p_data["rollouts"] if r["is_correct"]]

        if not correct_rollouts:
            traces_block.append(f"\n[Model {STUDENT_LABELS[s]}]: No correct solutions.\n")
            continue

        n = min(SAMPLE_PER_MODEL, len(correct_rollouts))
        step = max(1, len(correct_rollouts) // n)
        sampled = correct_rollouts[::step][:n]

        for i, r in enumerate(sampled):
            tid = f"{label_prefix}{i+1}"
            trace_labels.append(tid)
            text = r["response"][:3000]
            if len(r["response"]) > 3000:
                text += "\n[...truncated...]"
            traces_block.append(f"\n--- Trace {tid} (Model {STUDENT_LABELS[s]}) ---\n{text}\n")

    traces_str = "\n".join(traces_block)

    prompt = f"""You are analyzing mathematical reasoning traces from 2 AI models.
Both are Qwen2.5-1.5B fine-tuned on different sets of reasoning traces.
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


def run_judge(client: OpenAI, problems: list[dict], student_data: dict) -> list[dict]:
    """Run LLM judge on all problems with correct student traces."""
    results = []
    total_input = 0
    total_output = 0

    for pid, prob in enumerate(problems):
        # Check if either student has correct traces
        any_correct = False
        for s in STUDENTS:
            if student_data[s]["problems"][pid]["n_correct"] > 0:
                any_correct = True
                break

        if not any_correct:
            print(f"  P{pid}: no correct traces from either student, skipping judge")
            results.append({"problem_id": pid, "skipped": True, "reason": "no_correct"})
            continue

        prompt, trace_labels = build_judge_prompt(prob, student_data, pid)

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

            # Count per student
            classifications = parsed.get("classifications", {})
            student_strats = {}
            for s in STUDENTS:
                prefix = STUDENT_LABELS[s].split(" ")[0]
                strats = set()
                for tid, sid in classifications.items():
                    if tid.startswith(prefix):
                        strats.add(sid)
                student_strats[s] = sorted(strats)

            print(f"  P{pid:2d} [{prob['tier']:6s}]: {n_strat} strats | "
                  f"S-R1={student_strats.get('student-r1', [])} "
                  f"S-v2={student_strats.get('student-v2', [])}")

            results.append({
                "problem_id": pid,
                "tier": prob["tier"],
                "subject": prob["subject"],
                "llm_response": parsed,
                "trace_labels": trace_labels,
                "student_strats": student_strats,
            })

        except Exception as e:
            print(f"  P{pid}: ERROR — {e}")
            results.append({"problem_id": pid, "error": str(e)})

        time.sleep(0.3)

    cost = total_input * 0.05 / 1e6 + total_output * 0.40 / 1e6
    print(f"\nJudge API: {total_input} input + {total_output} output tokens = ${cost:.4f}")
    return results


# ── Comparison Analysis ────────────────────────────────────────


def compare_accuracy(problems: list[dict], student_data: dict, teacher_data: dict):
    """Compare pass@k curves between students and teachers."""
    print(f"\n{'='*70}")
    print("ACCURACY COMPARISON: Students vs Teachers")
    print(f"{'='*70}")

    models = {
        "R1-Distill": ("teacher", "r1-distill"),
        "Nemotron-v2": ("teacher", "nemotron-v2"),
        "Student-R1": ("student", "student-r1"),
        "Student-v2": ("student", "student-v2"),
    }

    ks = [1, 4, 8, 16, 32, 64]
    hdr = f"{'Model':<16}" + "".join(f" {'p@'+str(k):>7}" for k in ks) + f" {'uniq':>6}"
    print(hdr)
    print("-" * len(hdr))

    for label, (source, name) in models.items():
        data = teacher_data if source == "teacher" else student_data
        prob_list = data[name]["problems"]
        n = len(prob_list)
        K = 64

        agg = {}
        for k in ks:
            from math import comb
            vals = []
            for p in prob_list:
                nc = p["n_correct"]
                if K - nc < k:
                    vals.append(1.0)
                else:
                    vals.append(1.0 - comb(K - nc, k) / comb(K, k))
            agg[k] = sum(vals) / len(vals)

        mean_uniq = sum(p["n_unique_answers"] for p in prob_list) / n
        row = f"{label:<16}"
        for k in ks:
            row += f" {agg[k]:>7.3f}"
        row += f" {mean_uniq:>6.1f}"
        print(row)

    # Per-tier
    for tier in ["easy", "medium", "hard"]:
        tier_ids = {p["problem_id"] for p in problems if p["tier"] == tier}
        print(f"\n  {tier.upper()} (n={len(tier_ids)}):")
        for label, (source, name) in models.items():
            data = teacher_data if source == "teacher" else student_data
            tier_probs = [p for p in data[name]["problems"] if p["problem_id"] in tier_ids]
            if not tier_probs:
                continue
            K = 64
            pk1 = sum(
                1.0 if K - p["n_correct"] < 1 else 1.0 - (K - p["n_correct"]) / K
                for p in tier_probs
            ) / len(tier_probs)
            mc = sum(p["n_correct"] for p in tier_probs) / len(tier_probs)
            print(f"    {label:<16} pass@1={pk1:.3f}  mean_correct={mc:.1f}/64")


def compare_strategies(judge_results: list[dict], baseline_judge: list[dict]):
    """Compare strategy counts between students and teachers."""
    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON: Students vs Teachers (from Run 16)")
    print(f"{'='*70}")

    if not baseline_judge:
        print("  No baseline judge data available.")
        return

    # Build baseline lookup
    baseline = {}
    for r in baseline_judge:
        if "llm_response" in r:
            pid = r["problem_id"]
            resp = r["llm_response"]
            classifications = resp.get("classifications", {})

            teacher_strats = {}
            for prefix, model in [("A", "r1-distill"), ("B", "nemotron-v1"),
                                   ("C", "nemotron-v2"), ("D", "nemotron-brorl")]:
                strats = set()
                for tid, sid in classifications.items():
                    if tid.startswith(prefix):
                        strats.add(sid)
                teacher_strats[model] = sorted(strats)

            baseline[pid] = {
                "n_strategies": resp.get("n_strategies", 0),
                "teacher_strats": teacher_strats,
                "strategies": resp.get("strategies", []),
            }

    # Compare
    print(f"{'Prob':>5} {'Tier':>7} {'T:R1':>6} {'T:v2':>6} {'S:R1':>6} {'S:v2':>6} {'T_n':>5} {'S_n':>5} {'Delta':>7}")
    print("-" * 60)

    t_r1_total = 0
    t_v2_total = 0
    s_r1_total = 0
    s_v2_total = 0
    t_n_total = 0
    s_n_total = 0
    counted = 0

    for r in judge_results:
        if r.get("skipped") or r.get("error"):
            continue
        pid = r["problem_id"]
        tier = r.get("tier", "?")
        s_strats = r.get("student_strats", {})
        s_n = r["llm_response"].get("n_strategies", 0)

        s_r1_count = len(s_strats.get("student-r1", []))
        s_v2_count = len(s_strats.get("student-v2", []))

        # Teacher data from baseline
        b = baseline.get(pid, {})
        t_n = b.get("n_strategies", 0)
        t_r1_count = len(b.get("teacher_strats", {}).get("r1-distill", []))
        t_v2_count = len(b.get("teacher_strats", {}).get("nemotron-v2", []))

        delta = s_n - t_n

        print(f"  P{pid:<3d} {tier:>7} {t_r1_count:>6} {t_v2_count:>6} "
              f"{s_r1_count:>6} {s_v2_count:>6} {t_n:>5} {s_n:>5} {delta:>+7d}")

        t_r1_total += t_r1_count
        t_v2_total += t_v2_count
        s_r1_total += s_r1_count
        s_v2_total += s_v2_count
        t_n_total += t_n
        s_n_total += s_n
        counted += 1

    if counted:
        print("-" * 60)
        print(f"  {'AVG':>5} {'':>7} "
              f"{t_r1_total/counted:>6.1f} {t_v2_total/counted:>6.1f} "
              f"{s_r1_total/counted:>6.1f} {s_v2_total/counted:>6.1f} "
              f"{t_n_total/counted:>5.1f} {s_n_total/counted:>5.1f} "
              f"{(s_n_total-t_n_total)/counted:>+7.1f}")


def print_decision_table(problems: list[dict], student_data: dict,
                         teacher_data: dict, judge_results: list[dict]):
    """Print the decision table from the pilot design."""
    print(f"\n{'='*70}")
    print("DECISION TABLE")
    print(f"{'='*70}")

    # Compute aggregate metrics
    def get_pass1(data, name, prob_ids=None):
        probs = data[name]["problems"]
        if prob_ids:
            probs = [p for p in probs if p["problem_id"] in prob_ids]
        K = 64
        return sum(
            1.0 if K - p["n_correct"] < 1 else 1.0 - (K - p["n_correct"]) / K
            for p in probs
        ) / len(probs) if probs else 0

    sr1_p1 = get_pass1(student_data, "student-r1")
    sv2_p1 = get_pass1(student_data, "student-v2")
    tr1_p1 = get_pass1(teacher_data, "r1-distill")
    tv2_p1 = get_pass1(teacher_data, "nemotron-v2")

    # Strategy counts from judge
    s_r1_strats = []
    s_v2_strats = []
    for r in judge_results:
        if r.get("skipped") or r.get("error"):
            continue
        s = r.get("student_strats", {})
        s_r1_strats.append(len(s.get("student-r1", [])))
        s_v2_strats.append(len(s.get("student-v2", [])))

    avg_s_r1 = sum(s_r1_strats) / len(s_r1_strats) if s_r1_strats else 0
    avg_s_v2 = sum(s_v2_strats) / len(s_v2_strats) if s_v2_strats else 0

    print(f"\n  pass@1: Teacher-R1={tr1_p1:.3f}, Teacher-v2={tv2_p1:.3f}")
    print(f"  pass@1: Student-R1={sr1_p1:.3f}, Student-v2={sv2_p1:.3f}")
    print(f"  Avg strategies: Student-R1={avg_s_r1:.2f}, Student-v2={avg_s_v2:.2f}")

    print(f"\n  Findings:")

    # Finding 1: Student-R1 accuracy vs Teacher-R1
    r1_ratio = sr1_p1 / tr1_p1 if tr1_p1 > 0 else 0
    print(f"  1. Student-R1 retains {r1_ratio:.0%} of teacher accuracy")

    # Finding 2: Student-v2 accuracy vs Teacher-v2
    v2_ratio = sv2_p1 / tv2_p1 if tv2_p1 > 0 else 0
    print(f"  2. Student-v2 retains {v2_ratio:.0%} of teacher accuracy")

    # Finding 3: Which student is better?
    if abs(sr1_p1 - sv2_p1) < 0.02:
        print(f"  3. Students are ~equal on accuracy → base model dominates")
    elif sv2_p1 > sr1_p1:
        print(f"  3. Student-v2 > Student-R1 on accuracy → v2 traces carry better signal")
    else:
        print(f"  3. Student-R1 > Student-v2 on accuracy → R1 diversity helps")

    # Finding 4: Strategy comparison
    if avg_s_r1 > avg_s_v2 + 0.2:
        print(f"  4. Student-R1 more diverse ({avg_s_r1:.1f} vs {avg_s_v2:.1f}) → diversity IS in traces")
    elif avg_s_v2 > avg_s_r1 + 0.2:
        print(f"  4. Student-v2 more diverse ({avg_s_v2:.1f} vs {avg_s_r1:.1f}) → accuracy enables diversity")
    else:
        print(f"  4. Similar diversity → base model determines strategy repertoire")

    # Interpretation
    print(f"\n  INTERPRETATION:")
    # The key finding categories from our design
    if abs(sr1_p1 - sv2_p1) < 0.02 and abs(avg_s_r1 - avg_s_v2) < 0.2:
        print(f"  → Base model determines everything; teacher traces don't matter much")
        print(f"  → Explorability is a MODEL property (supports thesis)")
    elif avg_s_r1 > avg_s_v2 + 0.2 and sv2_p1 > sr1_p1:
        print(f"  → Diversity IS in traces (transferable), accuracy requires RL")
        print(f"  → Mixed evidence: traces carry strategy diversity but not robustness")
    elif sv2_p1 > sr1_p1 + 0.02 and avg_s_v2 >= avg_s_r1:
        print(f"  → v2 traces carry compressed, higher-quality signal")
        print(f"  → RL's benefit DOES transfer through traces → challenge to 'model state' thesis")
    elif sr1_p1 < tr1_p1 * 0.5 and sv2_p1 < tv2_p1 * 0.5:
        print(f"  → Both students much worse than teachers")
        print(f"  → Distillation is heavily lossy; traces alone insufficient")
        print(f"  → Supports: explorability requires model state, not just training data")
    else:
        print(f"  → Nuanced result — examine per-tier and per-problem patterns")


# ── Main ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge, only do accuracy comparison")
    args = parser.parse_args()

    # Check for student data
    if not STUDENT_DIR.exists():
        print(f"ERROR: Student trace data not found at {STUDENT_DIR}")
        print(f"Run the distillation pipeline first, then download results:")
        print(f"  modal volume get spark-pilot-results distill_pilot/full/ data/modal_runs/distill_pilot/full/")
        sys.exit(1)

    print("Loading student traces...")
    problems, student_data = load_student_data()

    print("Loading teacher traces...")
    teacher_data = load_teacher_data()

    print("Loading baseline judge results (Run 16)...")
    baseline_judge = load_baseline_judge()

    # Phase 1: Accuracy comparison (always)
    compare_accuracy(problems, student_data, teacher_data)

    # Phase 2: LLM judge (unless skipped)
    judge_results = []
    if not args.skip_judge:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("\nWARNING: No OPENAI_API_KEY set. Skipping LLM judge.")
            print("  Set it to run strategy classification.")
        else:
            client = OpenAI(api_key=api_key)
            print(f"\nRunning LLM judge on student traces ({MODEL_NAME})...")
            judge_results = run_judge(client, problems, student_data)
    else:
        # Try loading existing judge results
        judge_path = OUT_DIR / "distillation_judge.json"
        if judge_path.exists():
            print(f"\nLoading existing judge results from {judge_path}")
            with open(judge_path) as f:
                judge_results = json.load(f)

    # Phase 3: Strategy comparison
    if judge_results:
        compare_strategies(judge_results, baseline_judge)
        print_decision_table(problems, student_data, teacher_data, judge_results)

        # Save
        with open(OUT_DIR / "distillation_judge.json", "w") as f:
            json.dump(judge_results, f, indent=2)
        print(f"\nJudge results saved to {OUT_DIR / 'distillation_judge.json'}")
    else:
        print("\nNo judge results — run without --skip-judge to get full analysis")

    # Save summary
    summary = {
        "students": STUDENTS,
        "teachers": TEACHERS,
        "n_problems": len(problems),
        "student_summaries": {},
        "teacher_summaries": {},
    }
    for s in STUDENTS:
        p1 = sum(
            1.0 if 64 - p["n_correct"] < 1 else 1.0 - (64 - p["n_correct"]) / 64
            for p in student_data[s]["problems"]
        ) / len(student_data[s]["problems"])
        summary["student_summaries"][s] = {"pass_at_1": round(p1, 4)}
    for t in TEACHERS:
        p1 = sum(
            1.0 if 64 - p["n_correct"] < 1 else 1.0 - (64 - p["n_correct"]) / 64
            for p in teacher_data[t]["problems"]
        ) / len(teacher_data[t]["problems"])
        summary["teacher_summaries"][t] = {"pass_at_1": round(p1, 4)}

    with open(OUT_DIR / "distillation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {OUT_DIR / 'distillation_summary.json'}")


if __name__ == "__main__":
    main()
