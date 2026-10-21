import streamlit as st

NAV_ITEMS = [
    ("\U0001F3E0 Overview", "overview"),
    ("\U0001F4E6 Inventory", "inventory"),
    ("\U0001F4CA Sales History", "sales_history"),
    ("\U0001F4C8 Predictions", "predictions"),
    ("\U0001F514 Alerts & Recommendations", "alerts"),
    ("\U0001F4C4 Reports", "reports"),
    ("\U00002699 Settings", "settings"),
]


def render_sidebar() -> str:
    """Render the StockWise AI sidebar and return the selected page key."""
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "overview"

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

        selected = st.session_state.selected_page
        for label, key in NAV_ITEMS:
            is_active = key == selected
            if st.button(
                label,
                key=f"nav_{key}",
                width="stretch",
                type="primary" if is_active else "secondary",
            ):
                st.session_state.selected_page = key
                return key

        st.markdown('<hr class="profile-divider">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="user-profile">
              <div class="profile-avatar">U</div>
              <div class="profile-info">
                <div class="profile-name">Store Manager</div>
                <div class="profile-role">Inventory Administrator</div>
              </div>
              <button class="profile-logout" title="Logout">
                <span class="logout-icon">&#x274C;</span>
              </button>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return selected


def render_page_placeholder(page_title: str, detail: str) -> None:
    """Render a minimal placeholder for pages reserved for later phases."""
    st.title(page_title)
    st.info(detail)
    st.caption("This page is reserved for a later phase of StockWise AI.")
