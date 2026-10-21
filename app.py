import importlib.util
from html import escape
from pathlib import Path

import streamlit as st

from components.sidebar import render_sidebar
from components.metric_cards import render_metric_cards
from components.charts import (
    render_sales_forecast_chart,
    render_expiry_risk_chart,
    render_sample_products_table,
)
from database.db import init_database
from services.inventory_service import get_inventory_metrics, list_products

st.set_page_config(
    page_title="StockWise AI \u2014 Inventory Overview",
    page_icon="\U0001F4E6",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES_DIR = Path(__file__).parent / "page_modules"

PAGE_FILES = {
    "inventory": "1_Inventory.py",
    "sales_history": "2_Sales_History.py",
    "predictions": "3_Predictions.py",
    "alerts": "4_Alerts_Recommendations.py",
    "reports": "5_Reports.py",
    "settings": "6_Settings.py",
}


def load_css() -> None:
    css_path = Path(__file__).parent / "assets" / "styles.css"
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_page_module(page_key: str):
    """Dynamically load a page module from the page_modules/ directory."""
    file_name = PAGE_FILES.get(page_key)
    if not file_name:
        return None
    module_path = PAGES_DIR / file_name
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"pages_{page_key}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_search_results(query: str) -> None:
    """Render live search results read from the real inventory database.

    Uses actual records from the local SQLite database via the inventory
    service. No sample or illustrative data is used.
    """
    products = list_products(search=query)

    st.markdown('<div class="section-title">Search Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">'
        f'{len(products)} inventory record(s) matching "{escape(query)}".'
        '</div>',
        unsafe_allow_html=True,
    )

    if not products:
        st.info(f'No products match "{query}". Search by SKU or product name.')
        return

    badge_map = {
        "High Risk": "risk-high",
        "Expired": "risk-high",
        "Medium Risk": "risk-medium",
        "Low Risk": "risk-low",
    }

    rows = ""
    for product in products:
        badge_cls = badge_map.get(product.expiry_status, "risk-low")
        rows += (
            "<tr>"
            f"<td>{escape(product.sku)}</td>"
            f"<td>{escape(product.name)}</td>"
            f"<td>{escape(product.category)}</td>"
            f"<td>{product.quantity}</td>"
            f"<td>{escape(product.expiry_date)}</td>"
            f"<td><span class='risk-badge {badge_cls}'>"
            f"{escape(product.expiry_status)}</span></td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <table class="attention">
          <thead>
            <tr>
              <th>SKU</th>
              <th>Product</th>
              <th>Category</th>
              <th>Quantity</th>
              <th>Expiry Date</th>
              <th>Expiry Status</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    """Render the StockWise AI dashboard overview page."""
    st.title("Inventory Overview")
    st.markdown(
        '<div class="section-subtitle">Know what needs attention before it expires.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclosure">
          <b>Illustrative sample data \u2014 UI testing only.</b>
          The values and charts below are placeholder data used to demonstrate the
          dashboard layout. They are not real predictions, not actual business data,
          and not produced by any machine learning model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top bar: live search across real inventory records
    with st.container(key="top_bar"):
        top_col1, top_col2 = st.columns([3, 1], gap="medium", vertical_alignment="center")
        with top_col1:
            query = st.text_input(
                "Search inventory",
                placeholder="Search SKU or product name...",
                label_visibility="collapsed",
                key="top_search",
            )
        with top_col2:
            st.markdown(
                '<div class="top-bar-hint">Searches real inventory records</div>',
                unsafe_allow_html=True,
            )

    if query:
        render_search_results(query)

    metrics = get_inventory_metrics()
    render_metric_cards(metrics=metrics)

    if metrics["total_products"] == 0:
        st.info(
            "Your inventory is empty. Go to the **Inventory** page to add products, "
            "then return here to see live dashboard metrics."
        )

    st.markdown('<div class="section-title">Sales vs Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Illustrative historical sales and forecast values.</div>',
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(2, gap="medium")
    with chart_col1:
        render_sales_forecast_chart()
    with chart_col2:
        render_expiry_risk_chart()

    st.markdown('<div class="section-title">Needs Attention</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Sample products that may require action based on illustrative expiry data.</div>',
        unsafe_allow_html=True,
    )
    render_sample_products_table()


def main() -> None:
    load_css()
    init_database()
    selection_key = render_sidebar()

    if selection_key == "overview":
        render_overview()
        return

    module = load_page_module(selection_key)
    if module is not None and hasattr(module, "render"):
        module.render()
    else:
        st.error(f"Page module for '{selection_key}' could not be loaded.")


if __name__ == "__main__":
    main()
