# SEC 10-K Metric Assistant: Cited Financial Metric Computation from Annual Filings

**Graduate GenAI Systems Final Project**

A retrieval-augmented pipeline that computes derived financial metrics from Apple's FY2025 Form 10-K filing. The system retrieves evidence from the actual filing, extracts values with a Gemini LLM, calculates the metric using Python, and returns a cited, auditable answer — compared against a direct-LLM baseline that does none of that.

> **Disclaimer:** This tool is for educational and research purposes only. It does not constitute financial or investment advice. All computed values should be independently verified against the original SEC filing before use in any professional or analytical context.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [User, Workflow, and Business Problem](#2-user-workflow-and-business-problem)
3. [What the App Does](#3-what-the-app-does)
4. [Why GenAI Is Useful Here](#4-why-genai-is-useful-here)
5. [System Design and Architecture](#5-system-design-and-architecture)
6. [Baseline Comparison](#6-baseline-comparison)
7. [Evaluation and Results](#7-evaluation-and-results)
8. [What Worked and What Failed](#8-what-worked-and-what-failed)
9. [Human Review and Governance Boundaries](#9-human-review-and-governance-boundaries)
10. [Setup Instructions](#10-setup-instructions)
11. [Usage Instructions](#11-usage-instructions)
12. [File Structure](#12-file-structure)
13. [Example Questions](#13-example-questions)
14. [Limitations and Future Improvements](#14-limitations-and-future-improvements)

---

## 1. Project Overview

Most GenAI filing tools work as general-purpose Q&A chatbots: ask a question, get a text answer. This project takes a different approach. The central thesis is that **derived financial metrics** — ratios and percentages computed from multiple line items across the filing — are where a structured, grounded pipeline adds the most value over a raw LLM.

The system supports five metrics that require locating specific numbers in different sections of the 10-K, then applying a formula:

| Metric | Formula | Fields Required |
|--------|---------|----------------|
| Gross Margin | `(gross_profit / revenue) × 100` | Gross profit, Net sales |
| Operating Margin | `(operating_income / revenue) × 100` | Operating income, Net sales |
| Net Profit Margin | `(net_income / revenue) × 100` | Net income, Net sales |
| Current Ratio | `current_assets / current_liabilities` | Current assets, Current liabilities |
| R&D Expense YoY Growth | `((rd_current − rd_prior) / rd_prior) × 100` | R&D FY2025, R&D FY2024 |

Every result is grounded in the actual filing, computed with Python, and returned with the source passages so the analyst can verify it.

The Apple FY2025 Form 10-K (`data/apple_2025_10k.txt`) is included in this repository. It is a plain-text extraction of the public iXBRL filing from SEC EDGAR and is included for reproducibility. The original filing is available at no cost from the SEC EDGAR system.

---

## 2. User, Workflow, and Business Problem

**Target user:** A financial analyst reviewing Apple's FY2025 annual report.

**The problem:** A Form 10-K is 100–200 pages. The numbers needed to compute a ratio are often in different sections — the income statement, balance sheet, and notes are all separate. Finding the right line item, reading it correctly from a multi-year comparison table, and then computing the ratio by hand is time-consuming and error-prone.

**The workflow this system replaces:**

1. Analyst opens the 10-K PDF, searches for "gross profit" and "net sales"
2. Finds the Consolidated Statements of Operations — a dense two-column table
3. Reads off the FY2025 column values, hoping not to mix them up with FY2024
4. Opens a spreadsheet and divides the numbers
5. Repeats for each metric they need

**What this system does instead:**

1. Analyst types a natural language question
2. The system retrieves the relevant passage from the filing, extracts the values, computes the metric, and returns the answer with the source text

The output is designed to be auditable, not just fast. The system shows the formula used, the extracted input values, and the passages from the 10-K so the analyst can verify every number before using it.

---

## 3. What the App Does

The Streamlit app exposes the full pipeline in a five-step display:

**Step 1 — Classify:** Gemini reads the question and maps it to one of the five supported metric keys (or declines if it is out of scope).

**Step 2 — Retrieve:** A local TF-IDF search over 43 chunks of the 10-K returns the top-6 most relevant passages. No API call is needed for retrieval.

**Step 3 — Extract:** Gemini reads the retrieved passages and returns a structured JSON object with the specific numeric values needed for the formula (e.g., `{"revenue": 416161.0, "gross_profit": 195201.0}`).

**Step 4 — Compute:** Python applies the formula from `metrics.json` to the extracted values. No LLM is involved in the arithmetic.

**Step 5 — Format:** Gemini writes a concise 3–4 sentence explanation of the result, citing the line items and formula used.

The final output includes the computed value, the formula, the extracted inputs, and the source passages — enough for an analyst to verify the answer independently.

---

## 4. Why GenAI Is Useful Here

GenAI is used for the three subtasks where language understanding is genuinely required:

**Routing (classify_metric):** A user asking "How efficiently did Apple convert revenue into net income?" is asking for the net profit margin, but those exact words do not appear in the filing. The LLM understands that the question maps to `net_profit_margin`. A keyword lookup would fail here.

**Extraction (extract_values):** The Consolidated Statements of Operations presents three years of data in a single table. The LLM reads the passage and extracts the correct FY2025 value for a specific line item, filtering out the FY2024 and FY2023 columns. This is a targeted reading task that is difficult to do reliably with pattern matching.

**Explanation (format_answer):** The LLM writes a professional summary of the computed result, explaining what the metric means in context and citing the source line items. This is a generation task where fluency matters.

**Where GenAI is explicitly not used:** The arithmetic. All formulas are implemented as Python functions in `metric_engine.py`. This eliminates arithmetic hallucination entirely. The LLM never multiplies, divides, or subtracts — it only reads and writes.

---

## 5. System Design and Architecture

### Pipeline flow

```
User question
      |
      v
[1] llm_utils.classify_metric()     LLM: one-word routing to a metric key
      |
      v
[2] retriever.retrieve()            TF-IDF: top-6 chunks from the 10-K
      |
      v
[3] llm_utils.extract_values()      LLM: structured JSON of numeric values
      |
      v
[4] metric_engine.compute_metric()  Python: deterministic arithmetic
      |
      v
[5] llm_utils.format_answer()       LLM: cited explanation paragraph
      |
      v
    Streamlit UI
```

For unsupported questions, the pipeline exits cleanly at step 1 with a message explaining that the question is out of scope.

### Key design decisions

**Character-based chunking** (`ingest.py`): 1500-character chunks with 150-character overlap, split at newline boundaries. This keeps financial table rows intact and puts each major statement (income statement, balance sheet) in its own dedicated chunk, which is important for TF-IDF retrieval precision.

**Local TF-IDF retrieval** (`retriever.py`): `TfidfVectorizer` with unigrams and bigrams, cached to `data/tfidf_cache.pkl` after the first build. No API calls are needed for retrieval. Each metric in `metrics.json` includes `search_terms` — exact phrases from the document vocabulary (e.g., "Total current assets", "Gross margin") — appended to the query at retrieval time. This bridges the gap between natural language queries and financial document vocabulary.

**Structured JSON extraction** (`llm_utils.py`): `extract_values` uses `response_mime_type="application/json"` so the LLM always returns parseable JSON. Prompt includes per-field hints that specify exactly which year and which line item to find, reducing extraction errors from multi-year tables.

**Python-only arithmetic** (`metric_engine.py`): Each metric has an explicit Python function. The formula is displayed to the user alongside the result. This makes computation fully auditable and eliminates any risk of LLM hallucination in the math.

**Thinking disabled** (`thinking_budget=0`): All Gemini calls disable chain-of-thought reasoning. Classification, extraction, and explanation are narrow language tasks that do not benefit from extended reasoning — and enabling thinking on `gemini-2.5-flash` exhaust the token budget before any visible text is emitted.

**Retry with timeout** (`_call_with_retry` in `llm_utils.py`): Every Gemini API call is wrapped with a 90-second timeout (via `ThreadPoolExecutor`) and exponential backoff at 2s, 4s, and 8s for transient errors (429, 503, connection errors). This prevents the evaluation script from hanging indefinitely on a slow API response.

### LLM model

Default: `gemini-2.5-flash-lite` (via Google AI Studio free tier). Flash-Lite is sufficient because:
- Classification is a one-word output from a fixed list of five options
- Extraction reads a short passage and returns a small JSON object
- Explanation writes 3–4 sentences around a pre-computed number

Because Python handles all arithmetic, the model never needs to reason about calculations.

---

## 6. Baseline Comparison

The baseline (`baseline.py`) sends the question directly to Gemini with no retrieval, no formula, and no Python computation. This represents the naive approach: just ask the LLM.

| Capability | Metric-aware system | Baseline |
|-----------|--------------------|---------:|
| Retrieves from the actual filing | Yes | No |
| Uses a verified formula | Yes | No |
| Computes with Python | Yes | No |
| Shows source passages | Yes | No |
| Answer is auditable | Yes | No |

### Concrete examples

**FY2025 Gross Margin:**
- Baseline response: stated that Apple's FY2025 10-K had not yet been released, and offered estimates based on analyst projections.
- Metric-aware system: retrieved the Consolidated Statements of Operations from the FY2025 filing, extracted gross profit of $195,201M and net sales of $416,161M, and computed gross margin as **46.91%**.

**Current Ratio:**
- Baseline response: cited Apple's FY2023 balance sheet and gave a current ratio of approximately 1.10.
- Metric-aware system: retrieved the FY2025 balance sheet (as of September 27, 2025), extracted current assets of $147,957M and current liabilities of $165,631M, and computed a current ratio of **0.8933x** — a materially different result that reflects Apple's actual FY2025 balance sheet structure.

These examples illustrate why retrieval-grounded computation matters: the baseline's training data did not include the FY2025 filing, so it either declined or used stale figures. The metric-aware system uses the actual document.

---

## 7. Evaluation and Results

The evaluation script (`run_evaluation.py`) runs 10 questions through both systems and compares computed values against expected results from the FY2025 filing.

### Test set

| Type | Count | Description |
|------|-------|-------------|
| Canonical | 5 | One direct question per supported metric |
| Variation | 2 | Paraphrased wording for gross margin and net profit margin |
| Edge case | 1 | Current ratio phrased as a yes/no liquidity question |
| Unsupported | 2 | Total revenue (no formula), EPS (not a supported metric) |

### Results — Metric-aware system

| Dimension | Result |
|-----------|--------|
| Supported questions | 8 of 10 |
| Unsupported questions | 2 of 10 |
| Classification accuracy (supported) | **8 / 8 (100%)** |
| Unsupported questions correctly declined | **2 / 2 (100%)** |
| Computation success rate | **8 / 8 (100%)** |
| Numeric accuracy (within 0.5% tolerance) | **8 / 8 (100%)** |

Expected vs. computed values (FY2025):

| Metric | Expected | Computed | Match |
|--------|----------|----------|-------|
| Gross Margin | 46.9052% | 46.9052% | Yes |
| Operating Margin | 31.9708% | 31.9708% | Yes |
| Net Profit Margin | 26.9151% | 26.9151% | Yes |
| Current Ratio | 0.8933x | 0.8933x | Yes |
| R&D YoY Growth | 10.1371% | 10.1371% | Yes |

### Results — Baseline system

| Dimension | Result |
|-----------|--------|
| Questions answered | 10 / 10 |
| Python computation | 0 / 10 |
| Cited source / retrieval | 0 / 10 |

The baseline answers fluently but without grounding. It cannot verify values against the filing, cannot compute formulas, and for FY2025 data may rely on outdated or projected figures from its training data.

### Running the evaluation

```bash
# Full evaluation (10 questions)
python run_evaluation.py

# Quick smoke test (first N questions)
python run_evaluation.py --limit 2
```

Results are saved to `eval/results.csv`. Open it to review full answers side by side. After reviewing, fill in `eval/evaluation_summary.md` with qualitative analysis.

---

## 8. What Worked and What Failed

### What worked

**TF-IDF retrieval with search_terms:** Early testing used 3000-character chunks and plain text queries. TF-IDF would retrieve sections about Apple's business overview instead of the income statement, because the table-of-contents entries for those sections had higher overlap with the query words. Two fixes solved this: reducing chunk size to 1500 characters (so each financial statement occupies its own chunk) and adding `search_terms` to `metrics.json` (exact phrases from the document like "Total net sales" and "Gross margin") appended to retrieval queries. After these changes, all five metrics' required values appeared in the top-6 retrieved chunks.

**Structured JSON extraction with field hints:** The first extraction prompt asked for field names only. The model would sometimes extract the FY2024 value instead of FY2025 because the multi-year table lists both in adjacent columns. Adding per-field hints to the prompt (e.g., "Total current assets from the balance sheet as of September 27, 2025") made year-specific extraction reliable.

**`thinking_budget=0`:** Disabling Gemini's thinking tokens was necessary, not optional. Without it, `response.text` was `None` for classification calls because thinking tokens exhausted the budget before any visible text was emitted. Setting `thinking_budget=0` resolved this across all three LLM calls.

**Retry wrapper with thread timeout:** The evaluation script previously hung indefinitely on the first API call. Wrapping every `generate_content` call in a `ThreadPoolExecutor.result(timeout=90)` with exponential backoff gives the script a clean timeout path and makes the evaluation runnable end-to-end.

### What failed (and how it was addressed)

**229KB HTML-to-text extraction:** The initial approach stripped HTML tags directly. Each table cell became its own line, so a financial table row like `Net sales  391,035  383,285` became three separate lines with no structure. TF-IDF retrieved these fragments rather than coherent table rows. Fixed by a table-row-aware extractor that splits on `<tr>` boundaries and joins cell content per row, producing 54KB of clean, readable text.

**SEC EDGAR download failures:** SEC EDGAR returns 403 if the User-Agent header does not identify the requester. Fixed by using a SEC-compliant User-Agent header.

**`'NoneType' object has no attribute 'strip'`:** The original `classify_metric` called `response.text.strip()` directly. When Gemini returns a non-STOP finish reason (safety filter, token exhaustion), `response.text` is `None`. Fixed by adding `_extract_text(response)`, which falls back to inspecting `candidates[0].content.parts` and raises a descriptive error if no text is found anywhere in the response.

---

## 9. Human Review and Governance Boundaries

This system is a decision-support tool, not an autonomous agent. Human review is required at the following points:

**Verify extracted values before use.** The Computation Details panel in the app shows what the LLM extracted (e.g., `revenue = 416,161.00`, `gross_profit = 195,201.00`). Before using the result, confirm these values against the actual 10-K. A mis-read table cell will produce a wrong answer without any error signal.

**Interpret edge cases in context.** The current ratio of 0.8933x is factually correct for Apple's FY2025 balance sheet, but whether this is a concern requires understanding Apple's specific capital structure — large amounts of short-term marketable securities are not classified as current assets, and Apple carries significant short-term commercial paper. A number without context can mislead.

**Do not use outputs for investment decisions.** This system extracts and computes from a single filing. It does not account for adjustments, restatements, segment breakdowns, or the broader industry context that professional analysis requires.

**Approve before citing in reports.** Any metric used in a report, model, or client-facing document should be verified by a human analyst who has reviewed the source filing directly.

**Update for new filings.** When Apple's FY2026 10-K is released, delete `data/tfidf_cache.pkl` to force an index rebuild, replace `data/apple_2025_10k.txt` with the new filing text, and update expected values in `eval/test_questions.csv`.

---

## 10. Setup Instructions

**Requirements:** Python 3.10 or later. A free Google AI Studio API key.

```bash
# 1. Clone and enter the project directory
git clone <repo-url>
cd sec-10k-metric-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the environment template and add your API key
cp .env.example .env
```

Open `.env` and set your key:

```
GEMINI_API_KEY=your_google_ai_studio_api_key_here
GEMINI_CHAT_MODEL=gemini-2.5-flash-lite
```

Get a free key at Google AI Studio. Keep `GEMINI_CHAT_MODEL=gemini-2.5-flash-lite` — this is the model the project was built and evaluated on.

The 10-K text file (`data/apple_2025_10k.txt`) is included in the repository. No download step is needed.

On first run, the TF-IDF index is built from the text file and cached to `data/tfidf_cache.pkl` (under one second). Subsequent runs load from cache instantly.

---

## 11. Usage Instructions

### Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Type a question in the text box and click **Compute Metric**. The app displays each pipeline step — classification, retrieval, extraction, computation, and the formatted answer — so you can inspect the intermediate results.

For unsupported questions (e.g., "What is Apple's EPS?"), the system declines gracefully at the classification step.

### Evaluation script

```bash
# Full evaluation — all 10 test questions
python run_evaluation.py

# Quick test — first 2 questions only
python run_evaluation.py --limit 2
```

Console output shows per-step progress for each question. Results are saved to `eval/results.csv` with columns for both systems side by side.

### Screenshots

**Metric-aware system — gross margin computed successfully:**

![Metric-aware gross margin](screenshots/metric_aware_gross_margin.png)

The app retrieves relevant passages from the FY2025 10-K, extracts gross profit ($195,201M) and net sales ($416,161M) via Gemini, and computes gross margin as 46.91% using Python.

**Baseline — gross margin query fails to ground in FY2025 filing:**

![Baseline gross margin failure](screenshots/baseline_gross_margin_failure.png)

Without retrieval, the baseline LLM does not have access to the FY2025 10-K and either declines or returns an estimate based on stale training data.

**Full evaluation run — console summary:**

![Evaluation summary](screenshots/evaluation_summary.png)

Output of `python run_evaluation.py` across all 10 test questions: 8/8 classification accuracy on supported metrics, 2/2 unsupported questions correctly declined, 8/8 computation success rate, and 8/8 numeric accuracy within the 0.5% tolerance.

---

## 12. File Structure

```
sec-10k-metric-assistant/
|
|- app.py                  # Streamlit UI — 5-step pipeline display
|- ingest.py               # Load and chunk 10-K text (1500 chars, 150 overlap)
|- retriever.py            # TF-IDF index, cache, retrieve top-k chunks
|- llm_utils.py            # All Gemini API calls: classify, extract, format
|                          #   includes _call_with_retry (timeout + backoff)
|- metric_engine.py        # Python metric computation — no LLM math
|- baseline.py             # Direct-LLM baseline (no retrieval, no formula)
|- run_evaluation.py       # Batch evaluation: both systems, 10 questions
|                          #   supports --limit N for quick testing
|- metrics.json            # Metric definitions: formulas, fields, search_terms
|- requirements.txt
|- .env.example            # Template: GEMINI_API_KEY, GEMINI_CHAT_MODEL
|- .gitignore
|
|- data/
|   |- README.md           # Notes on the 10-K source file
|   |- apple_2025_10k.txt  # Plain-text Apple FY2025 10-K (public SEC filing)
|   `- tfidf_cache.pkl     # Auto-generated on first run — git-ignored, safe to delete
|
|- eval/
|   |- test_questions.csv  # 10 test questions with expected values
|   |- results.csv         # Generated by run_evaluation.py
|   `- evaluation_summary.md  # Fill in with qualitative analysis after running
|
`- screenshots/
    |- README.md                          # Screenshot guidance
    |- metric_aware_gross_margin.png      # App computing gross margin end-to-end
    |- baseline_gross_margin_failure.png  # Baseline failing without retrieval
    `- evaluation_summary.png             # Full 10-question evaluation console output
```

---

## 13. Example Questions

Questions the system can answer (supported metrics):

```
What is Apple's gross margin for FY2025?
What was Apple's operating margin in fiscal year 2025?
What is Apple's net profit margin for FY2025?
What is Apple's current ratio?
How much did Apple's R&D expenses grow year over year in FY2025?

# Paraphrased — also handled correctly
What percentage of Apple's revenue was retained as gross profit in FY2025?
How efficiently did Apple convert revenue into net income in FY2025?
Is Apple's current ratio above 1.0 — can it cover all short-term obligations?
```

Questions that are out of scope (system will decline):

```
What is Apple's total revenue for FY2025?
What is Apple's earnings per share for FY2025?
What did Tim Cook say about AI in the 10-K?
```

---

## 14. Limitations and Future Improvements

### Current limitations

**One company, one filing, five metrics.** The system is scoped to Apple's FY2025 10-K. Extending to other companies requires adding a new text file and, for new metric types, adding entries to `metrics.json`.

**TF-IDF retrieval does not generalize across documents.** TF-IDF works well when the query vocabulary matches the document vocabulary. For filings with unusual table formatting or non-standard terminology, retrieval quality may degrade. Embedding-based retrieval would be more robust.

**Extraction can fail on complex multi-column tables.** If the LLM reads the wrong column in a three-year comparison table (FY2025 vs. FY2024 vs. FY2023), it will extract the wrong value. The current mitigation is per-field hints in the extraction prompt specifying the exact date; manual verification is still recommended.

**No cross-metric analysis.** The system answers one metric per question. A question like "Compare Apple's gross margin trend over the last three years" is out of scope.

**Rate limits on the free tier.** The evaluation script sleeps 2 seconds between questions to avoid 429 errors from the Gemini free tier. Running the full 10-question evaluation takes approximately 60–90 seconds including API call time.

### Potential improvements

- **Embedding-based retrieval** (e.g., text-embedding-004) would handle vocabulary variation better than TF-IDF
- **Multi-company support** by parameterizing the document path and updating `metrics.json` with company-agnostic field descriptions
- **Additional metrics** such as debt-to-equity ratio, return on equity, or free cash flow
- **Confidence scoring** on extracted values — flag when the retrieved chunk does not contain the expected line item
- **Multi-year analysis** by extracting both current and prior year values and supporting trend questions
- **PDF ingestion** directly from SEC EDGAR instead of pre-converted plain text

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `google-genai` | Gemini API for classification, extraction, and explanation |
| `streamlit` | Web UI |
| `scikit-learn` | TF-IDF vectorizer and cosine similarity for retrieval |
| `numpy` | Array operations for retrieval scoring |
| `python-dotenv` | Load `.env` in development |
| `pandas` | CSV I/O in evaluation script |

---

## Data Source

`data/apple_2025_10k.txt` is a plain-text extraction of Apple's FY2025 Annual Report on Form 10-K, filed with the U.S. Securities and Exchange Commission on October 31, 2025. The original iXBRL filing is publicly available at no cost from SEC EDGAR. This file is included in the repository for reproducibility. It contains no proprietary information beyond what Apple disclosed publicly.

---

## License

This project is for educational purposes as part of a graduate Generative AI Systems course. Do not include API keys or non-public data in any public repository.
