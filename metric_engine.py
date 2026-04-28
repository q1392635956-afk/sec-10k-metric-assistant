"""
metric_engine.py
----------------
Load metric definitions from metrics.json and compute metric values using
explicit Python arithmetic — the LLM is never used for the math itself.

This separation is intentional: Python gives reproducible, auditable numbers;
the LLM is used only for language tasks (classification, extraction, formatting).
"""
from __future__ import annotations

import json
import os

METRICS_PATH = "metrics.json"


def load_metrics() -> dict:
    """Load all metric definitions from metrics.json."""
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"Metrics config not found at '{METRICS_PATH}'. "
            "Make sure you are running from the project root directory."
        )
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def get_metric_info(metric_key: str) -> dict:
    """Return the full definition dict for a single metric key."""
    metrics = load_metrics()
    if metric_key not in metrics:
        supported = list(metrics.keys())
        raise ValueError(f"Unknown metric '{metric_key}'. Supported: {supported}")
    return metrics[metric_key]


def compute_metric(metric_key: str, values: dict) -> float | None:
    """Compute a metric from extracted values using Python arithmetic.

    Returns the computed float, or None if any required field is missing / null.
    Raises ValueError for unknown metric keys.

    All formulas match metrics.json exactly — no eval(), no dynamic dispatch,
    so the logic is transparent and easy to audit.
    """
    metric = get_metric_info(metric_key)
    required = metric["required_fields"]

    # Validate all required fields are present and non-null
    for field in required:
        if values.get(field) is None:
            return None  # caller will surface a helpful error

    if metric_key == "gross_margin":
        # (gross_profit / revenue) * 100
        return (values["gross_profit"] / values["revenue"]) * 100

    elif metric_key == "operating_margin":
        # (operating_income / revenue) * 100
        return (values["operating_income"] / values["revenue"]) * 100

    elif metric_key == "net_profit_margin":
        # (net_income / revenue) * 100
        return (values["net_income"] / values["revenue"]) * 100

    elif metric_key == "current_ratio":
        # current_assets / current_liabilities
        if values["current_liabilities"] == 0:
            raise ZeroDivisionError("current_liabilities is zero — cannot compute current ratio.")
        return values["current_assets"] / values["current_liabilities"]

    elif metric_key == "rd_growth":
        # ((rd_current - rd_prior) / rd_prior) * 100
        if values["rd_prior"] == 0:
            raise ZeroDivisionError("rd_prior is zero — cannot compute YoY growth.")
        return ((values["rd_current"] - values["rd_prior"]) / values["rd_prior"]) * 100

    else:
        raise ValueError(f"No compute function defined for metric '{metric_key}'")
