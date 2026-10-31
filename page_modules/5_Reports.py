"""StockWise AI — Reports page (Phase 5 Step 4).

Sales analytics built on real database records. Empty states are shown
honestly when no sales exist; no illustrative data is presented as analytics.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from services.analytics_service import (
    best_selling_products,
    daily_revenue,
    monthly_revenue,
    product_sales_velocity,
    revenue_and_margin,
    slow_moving_products,
    transaction_counts,
    weekly_revenue,
)


def render() -> None:
    st.title("Reports & Analytics")
    st.markdown(
        '<div class="section-subtitle">Sales analytics computed from real transaction records.</div>',
        unsafe_allow_html=True,
    )

    tc = transaction_counts()
    if tc["total_transactions"] == 0:
        st.info("No sales records yet. Record sales on the Sales History page to see analytics.")
        return

    # Summary metrics
    rm = revenue_and_margin()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", tc["total_transactions"])
    c2.metric("Units Sold", tc["total_units_sold"])
    c3.metric("Revenue", f"\u20B9{rm['revenue_inr']:,.2f}")
    c4.metric("Gross Margin", f"\u20B9{rm['gross_margin_inr']:,.2f} ({rm['margin_percent']}%)")

    # Revenue trend (daily)
    st.markdown('<div class="section-title">Daily Revenue</div>', unsafe_allow_html=True)
    dr = daily_revenue()
    if dr:
        import plotly.express as px
        import pandas as pd
        df = pd.DataFrame(dr)
        df["label"] = pd.to_datetime(df["date"]).dt.strftime("%d %b %Y")
        df = df.sort_values("date")
        fig = px.bar(df, x="label", y="revenue_inr", text="revenue_inr",
                     labels={"label": "Date", "revenue_inr": "Revenue (INR)"},
                     title="Daily Revenue (INR)")
        fig.update_traces(marker_color="#10B981", texttemplate="₹%{y:,.2f}", textposition="outside",
                          width=0.35)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Date", yaxis_title="INR", height=360, bargap=0.6)
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No daily revenue data.")

    # Monthly revenue
    st.markdown('<div class="section-title">Monthly Revenue</div>', unsafe_allow_html=True)
    mr = monthly_revenue()
    if mr:
        import plotly.express as px
        import pandas as pd
        df = pd.DataFrame(mr)
        df["label"] = pd.to_datetime(df["month"] + "-01").dt.strftime("%b %Y")
        df = df.sort_values("month")
        fig = px.bar(df, x="label", y="revenue_inr", text="revenue_inr",
                     labels={"label": "Month", "revenue_inr": "Revenue (INR)"},
                     title="Monthly Revenue (INR)")
        fig.update_traces(marker_color="#10B981", texttemplate="₹%{y:,.2f}", textposition="outside",
                          width=0.35)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Month", yaxis_title="INR", height=360, bargap=0.6)
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No monthly revenue data.")

    # Best selling
    st.markdown('<div class="section-title">Best-Selling Products</div>', unsafe_allow_html=True)
    bs = best_selling_products()
    if bs:
        st.dataframe(__import__("pandas").DataFrame(bs), width="stretch", hide_index=True)

    # Slow moving
    st.markdown('<div class="section-title">Slow-Moving Products (last 30 days)</div>', unsafe_allow_html=True)
    sm = slow_moving_products(days=30)
    if sm:
        st.dataframe(__import__("pandas").DataFrame(sm), width="stretch", hide_index=True)
    else:
        st.info("No slow-moving products identified.")

    # Velocity
    st.markdown('<div class="section-title">Product Sales Velocity (last 30 days)</div>', unsafe_allow_html=True)
    vel = product_sales_velocity(days=30)
    if vel:
        st.dataframe(__import__("pandas").DataFrame(vel), width="stretch", hide_index=True)
    else:
        st.info("No sales in the last 30 days.")

    # ML readiness & data quality
    st.markdown('<div class="section-title">ML Readiness & Data Quality</div>', unsafe_allow_html=True)
    from services.ml_readiness_service import data_quality, ml_readiness, build_timeseries
    q = data_quality()
    r = ml_readiness()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", q["transactions"])
    c2.metric("Products with Sales", q["distinct_products_with_sales"])
    c3.metric("Days Coverage", q["days_coverage"])
    c4.metric("Quality OK", "Yes" if q["quality_ok"] else "No")

    st.caption(
        f"Historical sale items without a current product reference: "
        f"{q.get('historical_items_without_product', 0)}. "
        f"{q.get('historical_product_note', '')}"
    )

    if r["eligible"]:
        st.success(r["message"])
    else:
        st.warning(r["message"])
        for reason in r["reasons"]:
            st.caption(f"• {reason}")

    st.caption(
        f"Criteria: ≥{r['criteria']['min_transactions']} transactions, "
        f"≥{r['criteria']['min_days_coverage']} days coverage, "
        f"≥{r['criteria']['min_products_with_sales']} products with sales, quality OK."
    )

    if q["issues"]:
        st.markdown("**Data quality issues:**")
        for issue in q["issues"]:
            st.caption(f"• {issue}")

    st.markdown("**Historical time-series dataset (daily product-level units):**")
    ts = build_timeseries()
    if ts:
        import pandas as pd
        st.dataframe(pd.DataFrame(ts), width="stretch", hide_index=True)
    else:
        st.info("No time-series data available (no sales records).")
