"""
baseline.py
-----------
Simple baseline: send the user's question directly to the LLM with no
retrieval, no formula lookup, and no Python computation.

This is the "comparison system" — it shows what you get when you just ask
a language model a financial question versus the full metric-aware pipeline.
The baseline may hallucinate numbers or give vague answers; that contrast
is the point of the evaluation.
"""
from __future__ import annotations

import os

from openai import OpenAI

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key)


def baseline_answer(question: str) -> str:
    """Answer the question using LLM knowledge only.

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

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()
