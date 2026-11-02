"""StockWise AI — Sales History page (Phase 5 Step 3).

Records sales transactions with multiple line items, atomically reducing
inventory stock. Displays history with search and date filtering.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import streamlit as st

from services.inventory_service import list_products
from services.sales_service import (
    INR_TO_PAISE,
    from_paise,
    get_sale,
    get_sales_metrics,
    list_sales,
    record_sale,
    to_paise,
)
from utils.validators import ValidationError


def _init_state():
    defaults = {
        "sale_cart": [],
        "sale_success": None,
        "sale_error": None,
        "sale_last_product_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render() -> None:
    _init_state()
    st.title("Sales History")
    st.markdown(
        '<div class="section-subtitle">Record sales and review transaction history. '
        'Stock is reduced atomically. Amounts stored as integer paise.</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Record Sale", "Sales History", "Metrics"])

    # ---------------------------------------------------------------- Record Sale
    with tabs[0]:
        st.subheader("Record a Sale")
        if st.session_state.sale_success:
            st.success(st.session_state.sale_success)
            st.session_state.sale_success = None
        if st.session_state.sale_error:
            st.error(st.session_state.sale_error)
            st.session_state.sale_error = None

        products = list_products()
        if not products:
            st.info("No products in inventory. Add products first.")
            return

        # Product selector
        options = {f"{p.sku} — {p.name} (stock: {p.quantity})": p for p in products}
        selected_label = st.selectbox("Select product", list(options.keys()), key="sale_prod_select")
        selected = options[selected_label]

        # Sync price when product changes, but preserve manual edits for same product
        if st.session_state.get("sale_last_product_id") != selected.id:
            default_price = round(float(selected.unit_cost), 2)
            st.session_state["sale_price"] = default_price
            st.session_state["sale_last_product_id"] = selected.id

        col_qty, col_price = st.columns(2)
        with col_qty:
            qty = st.number_input("Quantity", min_value=1, step=1, value=1, key="sale_qty")
        with col_price:
            price = st.number_input(
                "Unit selling price (INR)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="sale_price",
            )

        if st.button("Add to Cart", key="sale_add_cart"):
            st.session_state.sale_cart.append({
                "product_id": selected.id,
                "sku": selected.sku,
                "name": selected.name,
                "quantity": int(qty),
                "unit_selling_price": float(price),
                "stock": selected.quantity,
                "expiry_status": selected.expiry_status,
            })
            st.rerun()

        # Cart
        if st.session_state.sale_cart:
            st.markdown("#### Cart")
            cart_rows = []
            for c in st.session_state.sale_cart:
                cart_rows.append({
                    "SKU": c["sku"],
                    "Product": c["name"],
                    "Qty": c["quantity"],
                    "Unit Price (INR)": f"\u20B9{c['unit_selling_price']:,.2f}",
                    "Line Total (INR)": f"\u20B9{c['quantity'] * c['unit_selling_price']:,.2f}",
                    "Stock": c["stock"],
                    "Expiry Status": c["expiry_status"],
                })
            st.dataframe(pd.DataFrame(cart_rows), width="stretch", hide_index=True)
            total = sum(c["quantity"] * c["unit_selling_price"] for c in st.session_state.sale_cart)
            st.markdown(f"**Cart Total: \u20B9{total:,.2f}**")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear Cart", key="sale_clear"):
                    st.session_state.sale_cart = []
                    st.rerun()
            with c2:
                if st.button("Complete Sale", type="primary", key="sale_complete"):
                    try:
                        from services.sales_service import SaleItemInput
                        items = [
                            SaleItemInput(
                                product_id=c["product_id"],
                                quantity=c["quantity"],
                                unit_selling_price=c["unit_selling_price"],
                            )
                            for c in st.session_state.sale_cart
                        ]
                        sale = record_sale(items)
                        st.session_state.sale_success = f"Sale #{sale.id} recorded. Total \u20B9{from_paise(sale.total_amount):,.2f}"
                        st.session_state.sale_cart = []
                        st.rerun()
                    except ValidationError as exc:
                        st.session_state.sale_error = str(exc)
                        st.rerun()
                    except Exception as exc:
                        st.session_state.sale_error = f"Sale failed: {exc}"
                        st.rerun()

    # ---------------------------------------------------------------- History
    with tabs[1]:
        st.subheader("Transaction History")
        col_search, col_start, col_end = st.columns([2, 1, 1])
        with col_search:
            search = st.text_input("Search by product name/SKU", key="sale_search")
        with col_start:
            start = st.date_input("From", value=None, key="sale_start")
        with col_end:
            end = st.date_input("To", value=None, key="sale_end")

        sales = list_sales(
            search=search or None,
            start_date=start.isoformat() if start else None,
            end_date=end.isoformat() if end else None,
        )

        if not sales:
            st.info("No sales recorded yet.")
            return

        rows = []
        for s in sales:
            rows.append({
                "Sale ID": s.id,
                "Date": s.transaction_date,
                "Items": len(s.items),
                "Total (INR)": f"\u20B9{from_paise(s.total_amount):,.2f}",
                "UUID": s.transaction_uuid[:8] + "…",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # Transaction details
        st.markdown("#### Transaction Details")
        sale_ids = [s.id for s in sales]
        selected_id = st.selectbox("Select sale", sale_ids, key="sale_detail_select")
        sale = get_sale(selected_id)
        if sale:
            detail_rows = []
            for it in sale.items:
                detail_rows.append({
                    "SKU": it.product_sku_snapshot,
                    "Product": it.product_name_snapshot,
                    "Qty": it.quantity_sold,
                    "Unit Price (INR)": f"\u20B9{from_paise(it.unit_selling_price):,.2f}",
                    "Unit Cost (INR)": f"\u20B9{from_paise(it.unit_cost_at_sale):,.2f}",
                    "Line Total (INR)": f"\u20B9{from_paise(it.line_total):,.2f}",
                })
            st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)
            st.caption(f"Sale #{sale.id} — Total \u20B9{from_paise(sale.total_amount):,.2f}")

    # ---------------------------------------------------------------- Metrics
    with tabs[2]:
        st.subheader("Sales Metrics")
        m = get_sales_metrics()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", m["total_sales"])
        c2.metric("Revenue", f"\u20B9{m['total_revenue_inr']:,.2f}")
        c3.metric("Items Sold", m["total_items_sold"])
        c4.metric("Gross Profit", f"\u20B9{from_paise(m['gross_profit_paise']):,.2f}")
