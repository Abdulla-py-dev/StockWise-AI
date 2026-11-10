"""StockWise AI — Predictions page (Phase 5 Step 6 Part 2B).

Demand forecasting pipeline infrastructure that does NOT claim unreliable
predictions. Shows forecasting eligibility, dataset sufficiency, model
training status, evaluation metrics (MAE/RMSE), and actual forecasts — each
clearly distinguished. Fabricated predictions are never shown as real results.
"""

from __future__ import annotations

import streamlit as st

from services.ml_forecast_service import run_forecast_pipeline


def _render_eligibility_card(eligibility: dict) -> None:
    eligible = bool(eligibility.get("eligible", False))
    criteria = eligibility.get("criteria", {})
    current = eligibility.get("current", {})

    if eligible:
        st.success(eligibility.get("message", "Forecasting eligible."))
    else:
        st.warning(eligibility.get("message", "Forecasting not eligible."))
        for reason in eligibility.get("reasons", []):
            st.caption(f"\u2022 {reason}")

    st.caption(
        "Criteria: "
        f"\u2265{criteria.get('min_transactions', '?')} transactions, "
        f"\u2265{criteria.get('min_days_coverage', '?')} days coverage, "
        f"\u2265{criteria.get('min_products_with_sales', '?')} products with sales, "
        "data quality OK."
    )
    st.caption(
        f"Current: {current.get('transactions', 0)} transactions, "
        f"{current.get('days_coverage', 0)} days coverage, "
        f"{current.get('products_with_sales', 0)} products with sales."
    )


def _render_sufficiency(sufficiency: dict) -> None:
    calendar_obs = int(sufficiency.get("calendar_observations", 0))
    engineered = int(sufficiency.get("engineered_rows", 0))
    supervised = int(sufficiency.get("supervised_rows", 0))
    minimum = int(sufficiency.get("min_required", 0))
    sufficient = bool(sufficiency.get("sufficient", False))

    st.caption(
        f"Calendar observations: {calendar_obs} "
        f"(all contiguous product-day observations)"
    )
    st.caption(
        f"Engineered feature rows: {engineered} "
        f"(after unavailable lag/rolling features excluded)"
    )
    st.caption(
        f"Supervised training rows: {supervised} "
        f"(after one-day-ahead target created, unavailable targets excluded)"
    )
    st.caption(f"Minimum required supervised rows: {minimum}.")

    if sufficient:
        st.success(
            f"Dataset sufficient: {supervised} supervised training row(s) "
            f"(minimum {minimum} required)."
        )
    else:
        st.warning(
            f"Dataset insufficient: {supervised} supervised training row(s), "
            f"minimum {minimum} required. "
            f"({calendar_obs} calendar observations → {engineered} engineered "
            f"rows → {supervised} supervised rows.)"
        )


def _render_training_status(result) -> None:
    status = result.training_status
    message = result.message

    if status == "completed":
        st.success(message)
    elif status in ("skipped_insufficient_data", "skipped_insufficient_split"):
        st.warning(message)
    elif status == "skipped_not_eligible":
        st.info(message)
    else:
        st.info(message)


def _render_metrics(result) -> None:
    st.markdown("**Evaluation metrics (chronological test set)**")

    cols = st.columns(3)
    with cols[0]:
        b = result.baseline
        if b.trained:
            st.metric("Naive baseline MAE", f"{b.mae:.3f}")
            st.metric("Naive baseline RMSE", f"{b.rmse:.3f}")
            st.caption(f"Test rows: {b.n_test}")
        else:
            st.metric("Naive baseline MAE", "\u2014")
            st.metric("Naive baseline RMSE", "\u2014")
            st.caption(b.skip_reason or "Not trained")

    with cols[1]:
        lr = next((m for m in result.models if m.name == "linear_regression"), None)
        if lr and lr.trained:
            st.metric("Linear Regression MAE", f"{lr.mae:.3f}")
            st.metric("Linear Regression RMSE", f"{lr.rmse:.3f}")
            st.caption(f"Train: {lr.n_train}, Test: {lr.n_test}")
        else:
            st.metric("Linear Regression MAE", "\u2014")
            st.metric("Linear Regression RMSE", "\u2014")
            st.caption(lr.skip_reason if lr else "Not available")

    with cols[2]:
        rf = next((m for m in result.models if m.name == "random_forest"), None)
        if rf and rf.trained:
            st.metric("Random Forest MAE", f"{rf.mae:.3f}")
            st.metric("Random Forest RMSE", f"{rf.rmse:.3f}")
            st.caption(f"Train: {rf.n_train}, Test: {rf.n_test}")
        else:
            st.metric("Random Forest MAE", "\u2014")
            st.metric("Random Forest RMSE", "\u2014")
            st.caption(rf.skip_reason if rf else "Not available")

    st.caption(
        "MAE = average absolute error in units. RMSE = root mean squared "
        "error (penalises larger misses). Lower is better. A model is only "
        "useful if it beats the naive last-observed-demand baseline."
    )


def _render_forecasts(result) -> None:
    st.markdown("**One-day-ahead forecasts**")

    if not result.forecasts:
        st.info(
            "No forecasts available. Forecasts are generated only after a "
            "model is trained, which requires sufficient genuine sales data."
        )
        return

    for f in result.forecasts:
        with st.container():
            st.markdown(
                f"**Product {f['product_id']} — {f.get('name') or f.get('sku') or 'Unknown'}**"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Forecast date", f["forecast_date"])
            with col2:
                st.metric("Last observed date", f["last_observed_date"])
            with col3:
                st.metric("Last observed units", f"{f['last_observed_units']:.0f}")
            st.metric(
                "Naive baseline forecast (next day)",
                f"{f['naive_last_observed_forecast']:.0f} units",
            )
            st.caption(
                "Forecast uses only history up to the last observed date. "
                "Model-based forecasts appear here once a model is trained."
            )


def render() -> None:
    st.title("Demand Forecasting")
    st.markdown(
        '<div class="section-subtitle">'
        "Safe forecasting pipeline: eligibility, sufficiency, training status, "
        "and evaluation — no unreliable predictions."
        "</div>",
        unsafe_allow_html=True,
    )

    result = run_forecast_pipeline()

    # 1. Forecasting eligibility
    st.markdown('<div class="section-title">Forecasting eligibility</div>', unsafe_allow_html=True)
    _render_eligibility_card(result.eligibility)

    # 2. Dataset sufficiency
    st.markdown('<div class="section-title">Dataset sufficiency</div>', unsafe_allow_html=True)
    _render_sufficiency(result.sufficiency)

    # 3. Model training status
    st.markdown('<div class="section-title">Model training status</div>', unsafe_allow_html=True)
    _render_training_status(result)

    # 4. Evaluation metrics
    if result.training_status == "completed":
        st.markdown('<div class="section-title">Evaluation metrics</div>', unsafe_allow_html=True)
        _render_metrics(result)

    # 5. Actual forecasts
    st.markdown('<div class="section-title">One-day-ahead forecasts</div>', unsafe_allow_html=True)
    _render_forecasts(result)

    # Transparency note
    st.markdown("---")
    st.caption(
        "Pipeline is read-only: no sales data is fabricated, backdated, or "
        "modified. No future sales are used as input features. Models are "
        "never trained on insufficient data. Metrics are computed on a "
        "strictly chronological train/test split (no random splitting)."
    )
