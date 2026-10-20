import streamlit as st

METRICS = [
    {
        "label": "Products Tracked",
        "value": "250",
        "delta": "0",
        "delta_kind": "navy",
        "help": "Illustrative sample value for UI testing only.",
    },
    {
        "label": "High-Risk Products",
        "value": "12",
        "delta": "0",
        "delta_kind": "red",
        "help": "Illustrative sample value for UI testing only.",
    },
    {
        "label": "Potentially Unsold Units",
        "value": "340",
        "delta": "0",
        "delta_kind": "amber",
        "help": "Illustrative sample value for UI testing only.",
    },
    {
        "label": "Potential Cost Exposure",
        "value": "\u20B98,500",
        "delta": "0",
        "delta_kind": "navy",
        "help": "Illustrative sample value for UI testing only.",
    },
]


def render_metric_cards() -> None:
    """Render the four illustrative metric cards on the dashboard."""
    cols = st.columns(len(METRICS), gap="medium")
    for col, metric in zip(cols, METRICS):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">{metric["label"]}</div>
                  <div class="metric-value">{metric["value"]}</div>
                  <div class="metric-delta {metric['delta_kind']}">{metric['delta']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(metric["help"])
