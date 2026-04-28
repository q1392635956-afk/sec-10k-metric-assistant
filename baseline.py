"""
baseline.py
-----------
Simple baseline: send the user's question directly to Gemini with no
retrieval, no formula lookup, and no Python computation.

This is the comparison system — it shows what you get when you just ask
a language model a financial question versus the full metric-aware pipeline.
"""
from __future__ import annotations

import os

from google import genai
from google.genai import types

from llm_utils import _extract_text, _call_with_retry

MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your Google AI Studio key."
        )
    return genai.Client(api_key=api_key)


def baseline_answer(question: str) -> str:
    """Answer the question using Gemini's training knowledge only.

    No retrieval, no formula, no Python math — just the model's parametric
    knowledge of Apple's recent financials.
    """
    client = _get_client()

    prompt = (
        "You are a financial analyst assistant. A user is asking about Apple's "
        "FY2025 financial metrics from the annual Form 10-K filing.\n\n"
        "Answer the question as best you can based on your knowledge. "
        "Be specific with numbers if you know them. "
        "If you are uncertain about a specific figure, say so clearly rather than guessing.\n\n"
        f"Question: {question}"
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
