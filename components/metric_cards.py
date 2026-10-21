"""Dashboard metric cards backed by the local SQLite inventory database.

Values are real database records, never sample or illustrative data.
"""

from __future__ import annotations

import streamlit as st

from services.inventory_service import get_inventory_metrics


def _format_inr(value) -> str:
    try:
        return f"\u20B9{float(value):,.2f}"
    except (TypeError, ValueError):
        return "\u20B90.00"


def render_metric_cards(metrics=None, db_path=None) -> None:
    """Render the four metric cards using real inventory database values."""
    if metrics is None:
        metrics = get_inventory_metrics(db_path=db_path)

    METRICS = [
        {
            "label": "Products Tracked",
            "value": str(metrics["total_products"]),
            "delta": "\u2014",
            "delta_kind": "emerald",
            "accent": "emerald",
            "help": "Total products in the local inventory database.",
        },
        {
            "label": "High-Risk Products",
            "value": str(metrics["high_risk_products"]),
            "delta": "\u2014",
            "delta_kind": "red",
            "accent": "red",
            "help": "Products with High Risk expiry status (expiring soon). "
            "Not a prediction of unsold stock.",
        },
        {
            "label": "Total Units in Stock",
            "value": str(metrics["total_units"]),
            "delta": "\u2014",
            "delta_kind": "green",
            "accent": "green",
            "help": "Sum of quantities across all tracked products.",
        },
        {
            "label": "Potential Cost Exposure",
            "value": _format_inr(metrics["cost_exposure"]),
            "delta": "\u2014",
            "delta_kind": "amber",
            "accent": "amber",
            "help": "Current value (qty x unit cost) of high-expiry-risk stock. "
            "Expiry status alone does not establish stock will remain unsold.",
        },
    ]

    cols = st.columns(len(METRICS), gap="medium")
    for col, metric in zip(cols, METRICS):
        with col:
            st.markdown(
                f"""
                <div class="metric-card metric-card-{metric['accent']}">
                  <div class="metric-accent {metric['accent']}"></div>
                  <div class="metric-label">{metric["label"]}</div>
                  <div class="metric-value">{metric["value"]}</div>
                  <div class="metric-delta {metric['delta_kind']}">{metric["delta"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(metric["help"])
