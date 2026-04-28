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

import concurrent.futures
import json
import os
import time

from google import genai
from google.genai import types

# Per-call timeout and backoff schedule for Gemini API calls.
API_TIMEOUT_SECONDS = 90
RETRY_DELAYS = [2, 4, 8]


def _is_retryable_error(exc: Exception) -> bool:
    retryable_keywords = (
        "503", "429", "unavailable", "resource_exhausted",
        "rate limit", "rate_limit", "timeout", "timed out",
        "connection", "temporarily", "overloaded", "network",
    )
    err_str = str(exc).lower()
    return (
        any(kw in err_str for kw in retryable_keywords)
        or isinstance(exc, (ConnectionError, TimeoutError, OSError))
    )


def _call_with_retry(fn):
    """Call fn() with timeout and exponential-backoff retry.

    fn must be a zero-argument callable that makes one Gemini API call.
    Raises the last exception after all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn)
        executor.shutdown(wait=False)
        try:
            return future.result(timeout=API_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            last_exc = TimeoutError(
                f"Gemini API call timed out after {API_TIMEOUT_SECONDS}s"
            )
        except Exception as exc:
            last_exc = exc
        if attempt == len(RETRY_DELAYS) or not _is_retryable_error(last_exc):
            raise last_exc
        wait = RETRY_DELAYS[attempt]
        print(f"  [retry {attempt + 1}/{len(RETRY_DELAYS)}] "
              f"{type(last_exc).__name__}: {str(last_exc)[:80]}")
        print(f"  Retrying in {wait}s...")
        time.sleep(wait)
    raise last_exc

MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")

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


def _extract_text(response) -> str:
    """Safely extract text from a Gemini GenerateContentResponse.

    Tries response.text first. If that is None (e.g. safety filter triggered,
    finish_reason != STOP), falls back to reading candidates[0].content.parts
    directly. Raises a clear ValueError if no usable text is found anywhere.
    """
    # Fast path: the SDK shortcut works for the vast majority of responses
    if response.text is not None:
        return response.text

    # Slow path: inspect candidates and parts manually
    try:
        for candidate in response.candidates or []:
            parts = getattr(getattr(candidate, "content", None), "parts", None) or []
            assembled = "".join(
                p.text for p in parts if getattr(p, "text", None)
            )
            if assembled:
                return assembled
    except Exception:
        pass

    # Build a useful error message that includes the finish reason
    finish_reason = "unknown"
    try:
        finish_reason = str(response.candidates[0].finish_reason)
    except Exception:
        pass

    raise ValueError(
        f"Gemini returned no text content (finish_reason={finish_reason}). "
        "Possible causes: safety filter triggered, response blocked, or empty output. "
        "Check your GEMINI_API_KEY, try a different question, or increase max_output_tokens."
    )


# ---------------------------------------------------------------------------
# 1. Metric classification
# ---------------------------------------------------------------------------

def classify_metric(question: str) -> str | None:
    """Map a free-text question to one of the supported metric keys.

    Returns the key string (e.g. 'gross_margin') or None if no match.
    Never raises on a None response — returns None instead.
    """
    client = _get_client()

    valid_keys = ", ".join(SUPPORTED_METRICS)

    prompt = (
        "You are a financial analyst assistant. Classify the question below into "
        "exactly one of these metric keys:\n\n"
        "  gross_margin        — gross margin or gross profit percentage\n"
        "  operating_margin    — operating margin or operating income percentage\n"
        "  net_profit_margin   — net profit margin or net income percentage\n"
        "  current_ratio       — current ratio or short-term liquidity\n"
        "  rd_growth           — R&D expense growth or R&D spending year-over-year change\n"
        "  unknown             — does not match any of the above\n\n"
        f"Valid outputs: {valid_keys}, unknown\n\n"
        "Rules:\n"
        "- Output ONLY the metric key — one word, no punctuation, no explanation.\n"
        "- You must choose from the valid outputs list above.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    response = _call_with_retry(
        lambda: client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=50,
                # Disable thinking for this simple classification task.
                # gemini-2.5-flash uses thinking tokens by default; without this,
                # thinking can exhaust the token budget before any text is emitted.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    )

    raw = _extract_text(response).strip().lower()

    # Exact match first
    if raw in SUPPORTED_METRICS:
        return raw
    if "unknown" in raw:
        return None

    # Fallback: model returned extra words — scan for the first valid key
    for key in SUPPORTED_METRICS:
        if key in raw:
            return key

    return None


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
    Values are in millions of USD unless noted otherwise.
    """
    client = _get_client()

    evidence_text = "\n\n---\n\n".join(evidence_chunks)
    fields_list = ", ".join(required_fields)

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
        "- Use FY2025 values unless the field description says 'prior' or 'FY2024'.\n\n"
        "Return ONLY a valid JSON object with the required field names as keys.\n"
        'Example: {"revenue": 391035.0, "gross_profit": 180683.0}\n\n'
        "Evidence from the 10-K:\n"
        f"{evidence_text}"
    )

    response = _call_with_retry(
        lambda: client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
                max_output_tokens=512,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    )

    try:
        raw = _extract_text(response)
        return json.loads(raw)
    except (ValueError, json.JSONDecodeError):
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

    response = _call_with_retry(
        lambda: client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    )
    return _extract_text(response).strip()
