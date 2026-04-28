"""
run_evaluation.py
-----------------
Run all questions in eval/test_questions.csv through both the metric-aware
system and the direct-LLM baseline, then save side-by-side results to
eval/results.csv.

Usage:
    python run_evaluation.py

The script prints a structured summary to the console. Open eval/results.csv
afterward to review full answers, then fill in eval/evaluation_summary.md
with your qualitative analysis.

CSV columns expected in test_questions.csv:
  question        — the user question
  question_type   — canonical | variation | edge_case | unsupported
  expected_metric — the correct metric key, or empty for unsupported questions
  expected_value  — the expected numeric result (float string), or "N/A"
"""
from __future__ import annotations

import argparse
import csv
import os
import time

from dotenv import load_dotenv

load_dotenv()

from retriever import build_or_load_index, retrieve
from llm_utils import classify_metric, extract_values, format_answer
from metric_engine import compute_metric, get_metric_info
from baseline import baseline_answer

TEST_CSV = "eval/test_questions.csv"
RESULTS_CSV = "eval/results.csv"

# Seconds to pause between questions to stay under free-tier rate limits.
# Each question makes ~3-4 API calls (classify + extract + format + baseline).
RATE_LIMIT_SLEEP = 2

# Relative tolerance for declaring a computed value "correct".
# 0.005 = 0.5% — tight enough to catch extraction errors but loose enough
# for minor floating-point differences in how the LLM reads table values.
NUMERIC_TOLERANCE = 0.005


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classification_correct(got: str | None, expected: str) -> bool:
    """Return True if the classification result matches the expectation.

    Handles unsupported questions (empty expected_metric) where the correct
    behavior is for the system to return None (not classify anything).
    """
    if expected == "":
        # Unsupported question: correct only if system declined to classify
        return got is None
    return got == expected


def _numeric_match(computed: float | None, expected_str: str) -> bool | None:
    """Return True/False if both values are present and within tolerance.

    Returns None when no numeric comparison is possible (N/A expected, or
    computed is None because extraction failed).
    """
    if expected_str in ("N/A", "", None) or computed is None:
        return None
    try:
        expected = float(expected_str)
    except ValueError:
        return None
    if expected == 0:
        return abs(computed) < 1e-9
    return abs(computed - expected) / abs(expected) <= NUMERIC_TOLERANCE


# ---------------------------------------------------------------------------
# Per-system runners
# ---------------------------------------------------------------------------

def run_metric_system(
    question: str,
    question_type: str,
    chunks: list,
    vectorizer,
    matrix,
) -> dict:
    """Run the full metric-aware pipeline on one question."""
    record: dict = {
        "system": "metric_aware",
        "question_type": question_type,
        "question": question,
        "metric_key": None,
        "extracted_values": None,
        "computed_value": None,
        "answer": None,
        "error": None,
    }
    try:
        print(f"       [metric-aware] classifying...", flush=True)
        metric_key = classify_metric(question)
        record["metric_key"] = metric_key
        print(f"       [metric-aware] metric_key={metric_key!r}", flush=True)

        if not metric_key:
            # Correctly declined (unsupported question) — not an error
            record["error"] = None
            print(f"       [metric-aware] unsupported question — done", flush=True)
            return record

        metric_info = get_metric_info(metric_key)
        query = " ".join([
            question,
            metric_info["name"],
            " ".join(metric_info.get("search_terms", [])),
        ])
        evidence = retrieve(query, chunks, vectorizer, matrix, top_k=6)
        print(f"       [metric-aware] retrieved {len(evidence)} chunks", flush=True)

        print(f"       [metric-aware] extracting values...", flush=True)
        extracted = extract_values(metric_key, metric_info["required_fields"], evidence)
        record["extracted_values"] = str(extracted)
        print(f"       [metric-aware] extracted={extracted}", flush=True)

        computed = compute_metric(metric_key, extracted)
        record["computed_value"] = computed
        print(f"       [metric-aware] computed={computed}", flush=True)

        if computed is not None:
            print(f"       [metric-aware] formatting answer...", flush=True)
            record["answer"] = format_answer(
                question=question,
                metric_name=metric_info["name"],
                computed_value=computed,
                unit=metric_info["unit"],
                extracted_values=extracted,
                evidence_chunks=evidence,
                formula=metric_info["formula"],
            )
            print(f"       [metric-aware] done", flush=True)
        else:
            record["error"] = "computation_failed_missing_values"
            print(f"       [metric-aware] ERROR: computation_failed_missing_values", flush=True)

    except Exception as exc:
        record["error"] = str(exc)
        print(f"       [metric-aware] EXCEPTION: {exc}", flush=True)

    return record


def run_baseline_system(question: str, question_type: str) -> dict:
    """Run the baseline (direct LLM, no retrieval) on one question."""
    record: dict = {
        "system": "baseline",
        "question_type": question_type,
        "question": question,
        "metric_key": None,
        "extracted_values": None,
        "computed_value": None,
        "answer": None,
        "error": None,
    }
    try:
        print(f"       [baseline] querying LLM...", flush=True)
        record["answer"] = baseline_answer(question)
        print(f"       [baseline] done", flush=True)
    except Exception as exc:
        record["error"] = str(exc)
        print(f"       [baseline] EXCEPTION: {exc}", flush=True)
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SEC 10-K metric evaluation")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Only evaluate the first N questions (useful for quick smoke tests)",
    )
    args = parser.parse_args()

    if not os.path.exists(TEST_CSV):
        print(f"[eval] Test CSV not found at '{TEST_CSV}'. Exiting.")
        return

    print("[eval] Loading TF-IDF index...")
    chunks, vectorizer, matrix = build_or_load_index()

    with open(TEST_CSV, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    if args.limit is not None:
        questions = questions[: args.limit]
        print(f"[eval] --limit {args.limit}: running first {len(questions)} question(s)")

    print(f"[eval] Running {len(questions)} questions × 2 systems "
          f"(~{len(questions) * RATE_LIMIT_SLEEP}s minimum)...\n")

    all_records: list[dict] = []

    for i, row in enumerate(questions, 1):
        question = row["question"]
        question_type = row.get("question_type", "")
        expected_metric = row.get("expected_metric", "")
        expected_value = row.get("expected_value", "N/A")

        print(f"[{i:2d}/{len(questions)}] [{question_type:12s}] {question}")

        # ---- Metric-aware system ----
        m_record = run_metric_system(question, question_type, chunks, vectorizer, matrix)
        m_record["expected_metric"] = expected_metric
        m_record["expected_value"] = expected_value
        m_record["classification_correct"] = _classification_correct(
            m_record["metric_key"], expected_metric
        )
        m_record["numeric_match"] = _numeric_match(
            m_record["computed_value"], expected_value
        )
        all_records.append(m_record)
        time.sleep(RATE_LIMIT_SLEEP)

        # ---- Baseline system ----
        b_record = run_baseline_system(question, question_type)
        b_record["expected_metric"] = expected_metric
        b_record["expected_value"] = expected_value
        b_record["classification_correct"] = None  # baseline doesn't classify
        b_record["numeric_match"] = None           # baseline doesn't compute
        all_records.append(b_record)
        time.sleep(RATE_LIMIT_SLEEP)

    # ---- Save results CSV ----
    fieldnames = [
        "system", "question_type", "question",
        "expected_metric", "expected_value",
        "metric_key", "classification_correct",
        "extracted_values", "computed_value", "numeric_match",
        "answer", "error",
    ]
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n[eval] Results saved to '{RESULTS_CSV}'")

    # ---- Console summary ----
    metric_rows = [r for r in all_records if r["system"] == "metric_aware"]
    baseline_rows = [r for r in all_records if r["system"] == "baseline"]

    # Split by question type
    supported = [r for r in metric_rows if r.get("expected_metric", "") != ""]
    unsupported = [r for r in metric_rows if r.get("expected_metric", "") == ""]

    correct_class_supported = sum(
        1 for r in supported if r["classification_correct"]
    )
    correct_class_unsupported = sum(
        1 for r in unsupported if r["classification_correct"]
    )

    computed_ok = sum(1 for r in supported if r["computed_value"] is not None)

    numeric_rows = [r for r in supported if r["numeric_match"] is not None]
    numeric_ok = sum(1 for r in numeric_rows if r["numeric_match"] is True)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTest set: {len(questions)} questions")
    print(f"  Supported metric questions:   {len(supported)}")
    print(f"  Unsupported/out-of-scope:     {len(unsupported)}")

    print(f"\n--- Metric-aware system ---")
    print(f"  Classification accuracy (supported):   "
          f"{correct_class_supported}/{len(supported)}"
          f"  ({_pct(correct_class_supported, len(supported))})")
    print(f"  Unsupported correctly declined:        "
          f"{correct_class_unsupported}/{len(unsupported)}"
          f"  ({_pct(correct_class_unsupported, len(unsupported))})")
    print(f"  Computation success rate:              "
          f"{computed_ok}/{len(supported)}"
          f"  ({_pct(computed_ok, len(supported))})")
    print(f"  Numeric accuracy (within 0.5%):        "
          f"{numeric_ok}/{len(numeric_rows)}"
          f"  ({_pct(numeric_ok, len(numeric_rows))})")

    print(f"\n--- Baseline system ---")
    baseline_answered = sum(1 for r in baseline_rows if r["answer"])
    print(f"  Questions answered:   {baseline_answered}/{len(baseline_rows)}")
    print(f"  Python computation:   0/{len(baseline_rows)}  (baseline never computes)")
    print(f"  Cited source:         0/{len(baseline_rows)}  (baseline has no retrieval)")

    print(f"\n--- Per-question detail (metric-aware) ---")
    for r in metric_rows:
        q_short = r["question"][:52] + ("…" if len(r["question"]) > 52 else "")
        got = r["metric_key"] or "none"
        exp = r["expected_metric"] or "—"
        val = f"{r['computed_value']:.4f}" if r["computed_value"] is not None else "—"
        nm = {True: "Y", False: "N", None: "-"}[r["numeric_match"]]
        cc = "Y" if r["classification_correct"] else "N"
        print(f"  [{cc}] [{nm}] {q_short}")
        print(f"       classified={got:20s} expected={exp:20s} computed={val}")

    print("\nNext steps:")
    print(f"  - Open {RESULTS_CSV} to review full answers side by side.")
    print("  - Fill in eval/evaluation_summary.md with qualitative notes.")
    print("=" * 60)


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "N/A"
    return f"{100 * num // den}%"


if __name__ == "__main__":
    main()
