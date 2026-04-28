"""
llm_utils.py
------------
All LLM API calls using Google Gemini via the google-genai SDK:

  classify_metric()  — routes a user question to a metric key
  extract_values()   — pulls required numbers out of evidence text as JSON
  format_answer()    — writes a cited explanation around a computed result

The actual math lives in metric_engine.py, not here.
"""
from __future__ import annotations

import json
import os

from google import genai
from google.genai import types

MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

SUPPORTED_METRICS = [
    "gross_margin",
    "operating_margin",
    "net_profit_margin",
    "current_ratio",
    "rd_growth",
]


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your Google AI Studio key."
        )
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# 1. Metric classification
# ---------------------------------------------------------------------------

def classify_metric(question: str) -> str | None:
    """Map a free-text question to one of the supported metric keys.

    Returns the key string (e.g. 'gross_margin') or None if no match.
    """
    client = _get_client()

    prompt = (
        "You are a financial analyst assistant. Classify the following question "
        "into exactly one of these metric categories:\n\n"
        "- gross_margin: gross margin or gross profit percentage\n"
        "- operating_margin: operating margin or operating income percentage\n"
        "- net_profit_margin: net profit margin or net income percentage\n"
        "- current_ratio: current ratio or short-term liquidity\n"
        "- rd_growth: R&D expense growth or research and development spending change\n\n"
        "If the question does not fit any category, respond with 'none'.\n\n"
        "Respond with ONLY the metric key or 'none'. No explanation, no punctuation.\n\n"
        f"Question: {question}"
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=20,
        ),
    )
    result = response.text.strip().lower()
    return result if result in SUPPORTED_METRICS else None


# ---------------------------------------------------------------------------
# 2. Value extraction
# ---------------------------------------------------------------------------

def extract_values(
    metric_key: str,
    required_fields: list[str],
    evidence_chunks: list[str],
) -> dict[str, float | None]:
    """Extract required numeric values from retrieved evidence chunks.

    Returns a dict mapping each required field name to a float (or None).
    Values are expected in millions of USD unless noted otherwise.
    """
    client = _get_client()

    evidence_text = "\n\n---\n\n".join(evidence_chunks)
    fields_list = ", ".join(required_fields)

    # Field-specific hints so the model knows what to look for
    field_hints = {
        "revenue": "Total net sales for FY2025 (fiscal year ended September 27, 2025)",
        "gross_profit": "Gross margin dollar amount for FY2025",
        "operating_income": "Operating income for FY2025",
        "net_income": "Net income for FY2025",
        "current_assets": "Total current assets from the balance sheet as of September 27, 2025",
        "current_liabilities": "Total current liabilities from the balance sheet as of September 27, 2025",
        "rd_current": "Research and development expense for FY2025 (most recent year)",
        "rd_prior": "Research and development expense for FY2024 (prior year, ended September 28, 2024)",
    }
    hints = "\n".join(
        f"  - {f}: {field_hints.get(f, f)}" for f in required_fields
    )

    prompt = (
        "You are extracting financial data from Apple's FY2025 Form 10-K filing.\n\n"
        f"Metric to compute: {metric_key}\n"
        f"Fields needed: {fields_list}\n\n"
        "Field descriptions:\n"
        f"{hints}\n\n"
        "Rules:\n"
        "- All values are in millions of USD (e.g. 391035.0 for $391,035 million).\n"
        "- Return numbers only — no units, commas, or dollar signs.\n"
        "- If a value cannot be found in the evidence, use null.\n"
        "- Use the most recent fiscal year (FY2025) values unless the field says 'prior'.\n\n"
        "Return ONLY a valid JSON object with the required field names as keys.\n"
        'Example: {"revenue": 391035.0, "gross_profit": 180683.0}\n\n'
        "Evidence from the 10-K:\n"
        f"{evidence_text}"
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )

    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, AttributeError):
        # Return all nulls so the caller can surface a clean error
        return {f: None for f in required_fields}


# ---------------------------------------------------------------------------
# 3. Answer formatting
# ---------------------------------------------------------------------------

def format_answer(
    question: str,
    metric_name: str,
    computed_value: float,
    unit: str,
    extracted_values: dict,
    evidence_chunks: list[str],
    formula: str,
) -> str:
    """Generate a concise, cited explanation of the computed metric result."""
    client = _get_client()

    values_str = ", ".join(
        f"{k} = {v:,.2f}" for k, v in extracted_values.items() if v is not None
    )

    prompt = (
        "You are a financial analyst explaining a computed metric to a colleague.\n\n"
        f"User question: {question}\n"
        f"Metric: {metric_name}\n"
        f"Formula used: {formula}\n"
        f"Inputs (millions USD where applicable): {values_str}\n"
        f"Computed result: {computed_value:.4f} {unit}\n\n"
        "Write a concise 3–4 sentence response that:\n"
        "1. States the computed value clearly.\n"
        "2. Explains what this metric means for Apple's financial health.\n"
        "3. Notes which line items from the 10-K were used.\n"
        "4. Adds one brief interpretive comment (healthy range, trend, or context).\n\n"
        "Be professional and concise. Do not invent numbers beyond what is provided above."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=400,
        ),
    )
    return response.text.strip()
