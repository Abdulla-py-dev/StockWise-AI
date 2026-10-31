"""StockWise AI — Alerts & Recommendations page (Phase 5).

Displays real inventory records prioritized by expiry risk, with financial
cost-value exposure and heuristic action recommendations.
"""

from __future__ import annotations

import streamlit as st

from services.expiry_service import prioritize_products, financial_analysis, stock_cost_value
from services.inventory_service import list_products
from services.recommendation_service import recommend


def render() -> None:
    st.title("Alerts & Recommendations")
    st.markdown(
        '<div class="section-subtitle">Expiry intelligence based on real inventory records. '
        'Cost values are cost exposure, not confirmed losses. Discount suggestions are heuristic.</div>',
        unsafe_allow_html=True,
    )

    products = prioritize_products(list_products())

    if not products:
        st.info("No products in inventory. Add products on the Inventory page.")
        return

    fin = financial_analysis(products)

    # Financial summary
    st.markdown('<div class="section-title">Financial Cost Exposure</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost Value", f"\u20B9{fin['total_cost_value']:,.2f}")
    c2.metric("Expired Value", f"\u20B9{fin['expired_value']:,.2f}")
    c3.metric("High-Risk Value", f"\u20B9{fin['high_risk_value']:,.2f}")
    c4.metric("Exposed (Expired+HR+MR)", f"\u20B9{fin['exposed_value']:,.2f}")
    st.caption(fin["potential_loss_note"])

    # Filters
    st.markdown('<div class="section-title">Prioritized Inventory</div>', unsafe_allow_html=True)
    col_risk, col_sort = st.columns([2, 2])
    with col_risk:
        risk_filter = st.selectbox(
            "Filter by risk", ["All", "Expired", "High Risk", "Medium Risk", "Low Risk"],
            key="alerts_risk_filter",
        )
    with col_sort:
        sort_by = st.selectbox(
            "Sort by", ["Priority (default)", "Stock Value (desc)", "Days Remaining (asc)"],
            key="alerts_sort",
        )

    filtered = products
    if risk_filter != "All":
        filtered = [p for p in filtered if p.expiry_status == risk_filter]

    if sort_by == "Stock Value (desc)":
        filtered = sorted(filtered, key=lambda p: stock_cost_value(p), reverse=True)
    elif sort_by == "Days Remaining (asc)":
        filtered = sorted(filtered, key=lambda p: (p.days_until_expiry if p.days_until_expiry is not None else 9999))

    rows = []
    for p in filtered:
        rec = recommend(p)
        rows.append({
            "Product": p.name,
            "SKU": p.sku,
            "Category": p.category,
            "Qty": p.quantity,
            "Unit Cost": f"\u20B9{p.unit_cost:,.2f}",
            "Stock Value": f"\u20B9{stock_cost_value(p):,.2f}",
            "Expiry Date": p.expiry_date,
            "Days Remaining": p.days_until_expiry,
            "Risk": p.expiry_status,
            "Suggested Action": rec["action"],
            "Discount % (heuristic)": rec["discount_percent"],
            "Explanation": rec["explanation"],
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(f"Showing {len(filtered)} of {len(products)} products. Discount % is a heuristic suggestion, not a guaranteed optimal price.")
