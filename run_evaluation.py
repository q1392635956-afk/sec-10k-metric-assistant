"""
run_evaluation.py
-----------------
Run all questions in eval/test_questions.csv through both systems and save
side-by-side results to eval/results.csv.

Usage:
    python run_evaluation.py

The script prints a summary to the console. Open eval/results.csv afterward
to review answers, compare computed values to expected values, and fill in
eval/evaluation_summary.md with your qualitative analysis.
"""
from __future__ import annotations

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

# Pause between API calls to stay under rate limits
RATE_LIMIT_SLEEP = 2  # seconds


# ---------------------------------------------------------------------------
# Per-system runners
# ---------------------------------------------------------------------------

def run_metric_system(question: str, chunks: list, embeddings) -> dict:
    """Run the full metric-aware pipeline on one question."""
    record: dict = {
        "system": "metric_aware",
        "question": question,
        "metric_key": None,
        "extracted_values": None,
        "computed_value": None,
        "answer": None,
        "error": None,
    }
    try:
        metric_key = classify_metric(question)
        record["metric_key"] = metric_key
        if not metric_key:
            record["error"] = "metric_not_classified"
            return record

        metric_info = get_metric_info(metric_key)
        query = (
            f"{question} {metric_info['name']} "
            f"{' '.join(metric_info['required_fields'])}"
        )
        evidence = retrieve(query, chunks, embeddings, top_k=6)

        extracted = extract_values(metric_key, metric_info["required_fields"], evidence)
        record["extracted_values"] = str(extracted)

        computed = compute_metric(metric_key, extracted)
        record["computed_value"] = computed

        if computed is not None:
            record["answer"] = format_answer(
                question=question,
                metric_name=metric_info["name"],
                computed_value=computed,
                unit=metric_info["unit"],
                extracted_values=extracted,
                evidence_chunks=evidence,
                formula=metric_info["formula"],
            )
        else:
            record["error"] = "computation_failed_missing_values"

    except Exception as exc:
        record["error"] = str(exc)

    return record


def run_baseline_system(question: str) -> dict:
    """Run the baseline (direct LLM) on one question."""
    record: dict = {
        "system": "baseline",
        "question": question,
        "metric_key": None,
        "extracted_values": None,
        "computed_value": None,
        "answer": None,
        "error": None,
    }
    try:
        record["answer"] = baseline_answer(question)
    except Exception as exc:
        record["error"] = str(exc)
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.exists(TEST_CSV):
        print(f"[eval] Test CSV not found at '{TEST_CSV}'. Exiting.")
        return

    # Load the embedding index once (expensive, cache is reused)
    print("[eval] Loading embedding index...")
    chunks, embeddings = build_or_load_index()

    with open(TEST_CSV, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    print(f"[eval] Running {len(questions)} questions through both systems...\n")

    all_records: list[dict] = []

    for i, row in enumerate(questions, 1):
        question = row["question"]
        expected_metric = row.get("expected_metric", "")
        expected_value = row.get("expected_value", "")

        print(f"[{i}/{len(questions)}] {question}")

        # Metric-aware
        m_record = run_metric_system(question, chunks, embeddings)
        m_record["expected_metric"] = expected_metric
        m_record["expected_value"] = expected_value
        all_records.append(m_record)
        time.sleep(RATE_LIMIT_SLEEP)

        # Baseline
        b_record = run_baseline_system(question)
        b_record["expected_metric"] = expected_metric
        b_record["expected_value"] = expected_value
        all_records.append(b_record)
        time.sleep(RATE_LIMIT_SLEEP)

    # Save results CSV
    fieldnames = [
        "system", "question", "expected_metric", "expected_value",
        "metric_key", "extracted_values", "computed_value", "answer", "error",
    ]
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n[eval] Results saved to '{RESULTS_CSV}'")

    # Console summary
    metric_rows = [r for r in all_records if r["system"] == "metric_aware"]
    computed_ok = sum(1 for r in metric_rows if r["computed_value"] is not None)
    correct_class = sum(
        1 for r in metric_rows
        if r.get("metric_key") == r.get("expected_metric")
    )

    print("\n=== Evaluation Summary ===")
    print(f"Total questions:            {len(questions)}")
    print(f"Metric classification acc:  {correct_class}/{len(metric_rows)} "
          f"({100 * correct_class // max(len(metric_rows), 1)}%)")
    print(f"Computation success rate:   {computed_ok}/{len(metric_rows)} "
          f"({100 * computed_ok // max(len(metric_rows), 1)}%)")
    print("\nOpen eval/results.csv to review full answers and compare systems.")
    print("Fill in eval/evaluation_summary.md with your qualitative analysis.")


if __name__ == "__main__":
    main()
