from html import escape

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


def _select_page(page_key: str) -> None:
    """Set the active page. Registered as the navigation buttons' on_click callback.

    A callback runs before the script re-executes, so on the render that
    follows a click the active navigation item is already correct and every
    navigation button is rendered. Setting the state inline after the button
    returned True would instead leave the previous item highlighted and skip
    the remaining buttons for that run.
    """
    st.session_state.selected_page = page_key


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
            st.button(
                label,
                key=f"nav_{key}",
                width="stretch",
                type="primary" if key == selected else "secondary",
                on_click=_select_page,
                args=(key,),
            )

    return st.session_state.selected_page


def render_page_placeholder(page_title: str, detail: str) -> None:
    """Render a themed empty state for pages reserved for later phases."""
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="empty-state-icon">&#9683;</div>
          <h2 class="empty-state-title">{escape(page_title)}</h2>
          <div class="empty-state-text">{escape(detail)}</div>
          <div class="empty-state-note">Reserved for a later phase</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
