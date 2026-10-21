import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_sales_forecast_chart() -> None:
    """Render an illustrative Sales vs Forecast line chart."""
    st.subheader("Sales vs Forecast")
    st.caption(
        "Illustrative sample data for UI testing only - not live application data "
        "and not a model prediction."
    )

    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    actual = [1240, 1380, 1520, 1480, 1610, 1750, 1690, 1820, 1900, 1850, 1720, 1680]
    forecast = [1200, 1350, 1500, 1550, 1650, 1780, 1720, 1850, 1880, 1820, 1750, 1700]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=months,
            y=actual,
            name="Actual Sales",
            mode="lines+markers",
            line=dict(color="#172B2A", width=3),
            marker=dict(size=7, symbol="circle"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=forecast,
            name="Illustrative Forecast",
            mode="lines+markers",
            line=dict(color="#10B981", width=3, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Monthly Sales vs Illustrative Forecast",
            x=0.02,
            xanchor="left",
            font=dict(size=15, color="#1F2937", family="Inter, system-ui, -apple-system, sans-serif"),
        ),
        xaxis_title="Month",
        yaxis_title="Units Sold",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=60, b=50),
        height=360,
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", color="#1F2937", size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D1E7DB", linecolor="#B5D9C6", tickformat=",d")
    fig.update_yaxes(showgrid=True, gridcolor="#D1E7DB", linecolor="#B5D9C6", tickformat=",d")

    st.plotly_chart(fig, width="stretch", key="sales_forecast_chart")


def render_expiry_risk_chart() -> None:
    """Render an illustrative Expiry Risk Distribution donut chart."""
    st.subheader("Expiry Risk Distribution")
    st.caption(
        "Illustrative sample data for UI testing only - not live application data "
        "and not a model prediction."
    )

    labels = ["High Risk", "Medium Risk", "Low Risk"]
    values = [12, 38, 200]
    colors = ["#EF4444", "#F59E0B", "#16A34A"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                textinfo="label+value",
                hoverinfo="label+value+percent",
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Illustrative Product Risk Distribution",
            x=0.02,
            xanchor="left",
            font=dict(size=15, color="#1F2937", family="Inter, system-ui, -apple-system, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=40),
        height=360,
        showlegend=True,
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", color="#1F2937", size=12),
    )

    st.plotly_chart(fig, width="stretch", key="expiry_risk_chart")


def render_sample_products_table() -> None:
    """Render the 'Needs Attention' sample products table with search and risk filter."""
    st.subheader("Needs Attention")
    st.caption(
        "Illustrative sample data for UI testing only - not live inventory records "
        "and not real business data."
    )

    products = [
        {"Product": "Milk", "Stock Quantity": 100, "Expiry / Days Remaining": "2 days", "Risk Level": "High", "Suggested Next Step": "Discount and prioritize for immediate sale"},
        {"Product": "Bread", "Stock Quantity": 60, "Expiry / Days Remaining": "1 day", "Risk Level": "High", "Suggested Next Step": "Mark down heavily; donate or discard unsold units"},
        {"Product": "Yogurt", "Stock Quantity": 80, "Expiry / Days Remaining": "4 days", "Risk Level": "Medium", "Suggested Next Step": "Run a short-term promotion this week"},
        {"Product": "Biscuits", "Stock Quantity": 200, "Expiry / Days Remaining": "180 days", "Risk Level": "Low", "Suggested Next Step": "Monitor routinely; no action needed"},
    ]

    df = pd.DataFrame(products)

    col_search, col_filter = st.columns([3, 2])
    with col_search:
        search = st.text_input(
            "Filter sample products",
            placeholder="Filter by product name...",
            key="sample_products_search",
        )
    with col_filter:
        risk_options = ["All"] + sorted(df["Risk Level"].unique().tolist())
        risk_filter = st.selectbox("Filter by risk", risk_options, index=0)

    filtered = df
    if search:
        mask = df["Product"].astype(str).str.contains(search, case=False, na=False)
        filtered = df[mask]
    if risk_filter != "All":
        filtered = filtered[filtered["Risk Level"] == risk_filter]

    badge_map = {
        "High": 'risk-high',
        "Medium": 'risk-medium',
        "Low": 'risk-low',
    }

    rows = ""
    for _, row in filtered.iterrows():
        badge_cls = badge_map.get(row["Risk Level"], "risk-low")
        rows += (
            "<tr>"
            f"<td>{row['Product']}</td>"
            f"<td>{row['Stock Quantity']}</td>"
            f"<td>{row['Expiry / Days Remaining']}</td>"
            f"<td><span class='risk-badge {badge_cls}'>{row['Risk Level']}</span></td>"
            f"<td>{row['Suggested Next Step']}</td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <table class="attention">
          <thead>
            <tr>
              <th>Product</th>
              <th>Stock Quantity</th>
              <th>Expiry / Days Remaining</th>
              <th>Risk Level</th>
              <th>Suggested Next Step</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Showing {len(filtered)} of {len(df)} illustrative products.")
