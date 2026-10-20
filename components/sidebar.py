import streamlit as st

NAV_ITEMS = [
    ("Overview", "overview"),
    ("Inventory", "inventory"),
    ("Sales History", "sales_history"),
    ("Predictions", "predictions"),
    ("Alerts & Recommendations", "alerts"),
    ("Reports", "reports"),
    ("Settings", "settings"),
]


def render_sidebar() -> str:
    """Render the StockWise AI sidebar and return the selected page key."""
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-mark">
                <div class="brand-icon">S</div>
                <div>
                  <div class="brand-name">StockWise AI</div>
                  <div class="brand-tagline">Predict Smarter. Waste Less. Sell Better.</div>
                </div>
              </div>
            </div>
            <hr class="nav-divider">
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)

        labels = [label for label, _ in NAV_ITEMS]
        default_index = 0

        selection = st.radio(
            "Navigation",
            labels,
            index=default_index,
            label_visibility="collapsed",
        )

        for label, key in NAV_ITEMS:
            if selection == label:
                return key

        return "overview"


def render_page_placeholder(page_title: str, detail: str) -> None:
    """Render a minimal placeholder for pages reserved for later phases."""
    st.title(page_title)
    st.info(detail)
    st.caption("This page is reserved for a later phase of StockWise AI.")
