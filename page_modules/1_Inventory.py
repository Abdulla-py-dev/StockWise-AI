"""StockWise AI Inventory page (Phase 2).

Provides a complete local inventory management interface backed by SQLite:

* Add product (validated form, duplicate-submission resistant)
* View products (search, category / expiry-status / expiring-soon filters,
  sorting, pagination)
* Edit product
* Delete product (with confirmation)
* CSV import with validation, preview, row-level errors, and a transactional import

All database access goes through services.inventory_service; UI code contains
no SQL.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from services.inventory_service import (
    HIGH_RISK_DAYS,
    MEDIUM_RISK_DAYS,
    add_product,
    classify_expiry,
    delete_product,
    get_categories,
    get_inventory_metrics,
    get_product,
    import_products_from_rows,
    list_products,
    update_product,
)
from utils.validators import ValidationError, parse_and_validate_csv

PAGE_SIZE_OPTIONS = [10, 25, 50, 100]
SORT_OPTIONS = ["name", "sku", "category", "quantity", "unit_cost", "expiry_date"]
STATUS_OPTIONS = ["All", "Expired", "High Risk", "Medium Risk", "Low Risk"]

STATUS_BADGE_CLASS = {
    "Expired": "risk-high",
    "High Risk": "risk-high",
    "Medium Risk": "risk-medium",
    "Low Risk": "risk-low",
}


def _init_state() -> None:
    defaults = {
        "inv_edit_id": None,
        "inv_delete_id": None,
        "inv_delete_confirmed": False,
        "inv_page": 1,
        "inv_page_size": 25,
        "inv_add_success": None,
        "inv_add_error": None,
        "inv_import_preview": None,
        "inv_import_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Add product
# ---------------------------------------------------------------------------

def render_add_product() -> None:
    st.subheader("Add Product")
    st.caption("Add a new product to the local inventory database.")

    if st.session_state.inv_add_success:
        st.success(st.session_state.inv_add_success)
        st.session_state.inv_add_success = None
    if st.session_state.inv_add_error:
        st.error(st.session_state.inv_add_error)
        st.session_state.inv_add_error = None

    # st.form ensures submission only happens on explicit button click,
    # which prevents accidental duplicate submissions.
    with st.form("add_product_form", clear_on_submit=False):
        sku = st.text_input("SKU", placeholder="e.g. MILK-001")
        name = st.text_input("Product name", placeholder="e.g. Toned Milk")
        category = st.text_input("Category", placeholder="e.g. Dairy")
        quantity = st.number_input("Quantity", min_value=0, step=1, value=0)
        unit_cost = st.number_input(
            "Unit cost (INR)", min_value=0.0, step=0.01, format="%.2f", value=0.0
        )
        expiry_date = st.date_input("Expiry date", value=None)
        submitted = st.form_submit_button("Add Product", type="primary")

    if submitted:
        try:
            if not sku or not name or not category:
                raise ValidationError("SKU, product name, and category are required.")
            if expiry_date is None:
                raise ValidationError("Expiry date is required.")
            add_product(
                sku=sku,
                name=name,
                category=category,
                quantity=int(quantity),
                unit_cost=float(unit_cost),
                expiry_date=expiry_date.isoformat(),
            )
            st.session_state.inv_add_success = f"Product '{name}' added successfully."
            st.rerun()
        except ValidationError as exc:
            st.session_state.inv_add_error = str(exc)
            st.rerun()
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.session_state.inv_add_error = f"Failed to add product: {exc}"
            st.rerun()


# ---------------------------------------------------------------------------
# View products
# ---------------------------------------------------------------------------

def _status_badge_html(status: str) -> str:
    css_class = STATUS_BADGE_CLASS.get(status, "risk-low")
    return f"<span class='risk-badge {css_class}'>{status}</span>"


def render_view_products() -> None:
    st.subheader("View Products")

    products = list_products()

    if not products:
        st.info(
            "No products in the inventory database yet. Use the **Add Product** tab "
            "to add your first product."
        )
        return

    col_search, col_category, col_status, col_soon = st.columns([3, 2, 2, 2])
    with col_search:
        search = st.text_input(
            "Search by SKU or product name",
            key="inv_search",
            placeholder="e.g. milk or SKU-100",
        )
    with col_category:
        category_options = ["All"] + get_categories()
        category_filter = st.selectbox(
            "Filter by category", category_options, key="inv_category_filter"
        )
    with col_status:
        status_filter = st.selectbox(
            "Filter by expiry status", STATUS_OPTIONS, key="inv_status_filter"
        )
    with col_soon:
        expiring_soon = st.checkbox(
            "Expiring soon only",
            key="inv_expiring_soon",
            help=f"Products expiring within {HIGH_RISK_DAYS} days (includes expired).",
        )

    sort_col, order_col = st.columns([2, 1])
    with sort_col:
        sort_by = st.selectbox(
            "Sort by", SORT_OPTIONS, key="inv_sort_by",
            index=SORT_OPTIONS.index("name"),
        )
    with order_col:
        sort_order = st.selectbox(
            "Order", ["Ascending", "Descending"], key="inv_sort_order", index=0
        )

    products = list_products(
        search=search or None,
        category=category_filter if category_filter != "All" else None,
        expiry_status=status_filter if status_filter != "All" else None,
        sort_by=sort_by,
        sort_order="asc" if sort_order == "Ascending" else "desc",
    )

    if expiring_soon:
        products = [
            p
            for p in products
            if p.days_until_expiry is not None and p.days_until_expiry <= HIGH_RISK_DAYS
        ]

    total = len(products)
    page_size = st.session_state.inv_page_size
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.session_state.inv_page
    if page > total_pages:
        page = total_pages
        st.session_state.inv_page = page

    start = (page - 1) * page_size
    page_products = products[start : start + page_size]

    rows = [
        {
            "SKU": p.sku,
            "Product": p.name,
            "Category": p.category,
            "Quantity": p.quantity,
            "Unit Cost (INR)": f"\u20B9{p.unit_cost:,.2f}",
            "Expiry Date": p.expiry_date,
            "Days Until Expiry": (
                p.days_until_expiry if p.days_until_expiry is not None else "\u2014"
            ),
            "Expiry Status": p.expiry_status,
        }
        for p in page_products
    ]
    df = pd.DataFrame(rows)

    def _status_color(val):
        colors = {
            "Expired": "color: #e63946; font-weight: 700;",
            "High Risk": "color: #e63946; font-weight: 700;",
            "Medium Risk": "color: #f4a261; font-weight: 700;",
            "Low Risk": "color: #2e7d32; font-weight: 700;",
        }
        return colors.get(val, "")

    styled = df.style.map(_status_color, subset=["Expiry Status"])
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        f"Showing {start + 1} to {min(start + len(page_products), total)} of {total} products."
    )

    col_prev, col_pagesize, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("Previous", disabled=(page <= 1), key="inv_prev"):
            st.session_state.inv_page = page - 1
            st.rerun()
    with col_pagesize:
        st.selectbox(
            "Page size", PAGE_SIZE_OPTIONS, key="inv_page_size"
        )
    with col_next:
        if st.button("Next", disabled=(page >= total_pages), key="inv_next"):
            st.session_state.inv_page = page + 1
            st.rerun()


# ---------------------------------------------------------------------------
# Edit product
# ---------------------------------------------------------------------------

def render_edit_product() -> None:
    st.subheader("Edit Product")
    products = list_products()
    if not products:
        st.info("No products available to edit.")
        return

    options = {f"{p.sku} \u2014 {p.name} ({p.category})": p.id for p in products}
    selected_label = st.selectbox(
        "Select product to edit", options=list(options.keys()), key="inv_edit_select"
    )
    product_id = options[selected_label]
    product = get_product(product_id)

    if product is None:
        st.error("Product not found.")
        return

    with st.form("edit_product_form"):
        sku = st.text_input("SKU", value=product.sku, key="inv_edit_sku")
        name = st.text_input("Product name", value=product.name, key="inv_edit_name")
        category = st.text_input("Category", value=product.category, key="inv_edit_category")
        quantity = st.number_input(
            "Quantity", min_value=0, step=1, value=product.quantity, key="inv_edit_quantity"
        )
        unit_cost = st.number_input(
            "Unit cost (INR)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            value=float(product.unit_cost),
            key="inv_edit_unit_cost",
        )
        expiry_date = st.date_input(
            "Expiry date",
            value=date.fromisoformat(product.expiry_date),
            key="inv_edit_expiry",
        )
        submitted = st.form_submit_button("Save Changes", type="primary")

    if submitted:
        try:
            update_product(
                product_id,
                sku=sku,
                name=name,
                category=category,
                quantity=int(quantity),
                unit_cost=float(unit_cost),
                expiry_date=expiry_date.isoformat(),
            )
            st.success("Product updated successfully.")
            st.session_state.inv_edit_id = None
            st.rerun()
        except ValidationError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.error(f"Failed to update product: {exc}")


# ---------------------------------------------------------------------------
# Delete product
# ---------------------------------------------------------------------------

def render_delete_product() -> None:
    st.subheader("Delete Product")
    products = list_products()
    if not products:
        st.info("No products available to delete.")
        return

    options = {f"{p.sku} \u2014 {p.name} ({p.category})": p.id for p in products}
    selected_label = st.selectbox(
        "Select product to delete", options=list(options.keys()), key="inv_delete_select"
    )
    product_id = options[selected_label]
    product = get_product(product_id)

    if product is None:
        st.error("Product not found.")
        return

    st.warning(
        f"You are about to permanently delete '{product.name}' (SKU: {product.sku}). "
        "This action cannot be undone."
    )
    confirmed = st.checkbox(
        "I confirm I want to delete this product", key="inv_delete_confirm"
    )
    if st.button("Delete Product", type="primary", disabled=not confirmed, key="inv_delete_btn"):
        try:
            delete_product(product_id)
            st.success("Product deleted.")
            st.session_state.inv_delete_confirmed = False
            st.rerun()
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.error(f"Failed to delete product: {exc}")


# ---------------------------------------------------------------------------
# CSV import
# ---------------------------------------------------------------------------

def render_csv_import() -> None:
    st.subheader("Import CSV")
    st.caption("Expected columns: sku, name, category, quantity, unit_cost, expiry_date")
    st.info(
        "Uploaded CSV files are read only. They are parsed with Python's csv module; "
        "no formulas, macros, or arbitrary code from the file are ever executed."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="inv_csv_upload")
    if uploaded is None:
        st.session_state.inv_import_preview = None
        st.session_state.inv_import_result = None
        return

    file_bytes = uploaded.read()
    try:
        header, valid_rows, invalid_rows = parse_and_validate_csv(file_bytes)
    except ValidationError as exc:
        st.error(str(exc))
        st.session_state.inv_import_preview = None
        return
    except Exception as exc:
        st.error(f"Failed to read CSV file: {exc}")
        return

    st.session_state.inv_import_preview = {
        "header": header,
        "valid": valid_rows,
        "invalid": invalid_rows,
    }
    preview = st.session_state.inv_import_preview
    valid = preview["valid"]
    invalid = preview["invalid"]

    st.write(f"Valid rows: **{len(valid)}**  |  Rows with errors: **{len(invalid)}**")

    if invalid:
        st.error(
            "Some rows failed validation and will not be imported. Fix the source file "
            "and re-upload, or import only the valid rows shown below."
        )
        invalid_data = [
            {"Row": r.row_number, **r.data, "Errors": "; ".join(r.errors)}
            for r in invalid
        ]
        st.dataframe(
            pd.DataFrame(invalid_data), width="stretch", hide_index=True
        )

    if valid:
        st.success(f"{len(valid)} row(s) are ready to import.")
        st.dataframe(pd.DataFrame(valid), width="stretch", hide_index=True)
        if st.button("Import Valid Rows", type="primary", key="inv_import_btn"):
            try:
                imported, rejected = import_products_from_rows(valid)
                st.session_state.inv_import_result = {
                    "imported": imported,
                    "rejected": rejected,
                }
                st.rerun()
            except Exception as exc:  # pragma: no cover - defensive UI guard
                st.error(f"Import failed: {exc}")

    if st.session_state.inv_import_result:
        result = st.session_state.inv_import_result
        st.success(f"Imported {result['imported']} row(s) into the database.")
        if result["rejected"]:
            st.warning(
                f"{len(result['rejected'])} row(s) were rejected because their SKU "
                "already exists in the database (or within the import batch)."
            )
            st.dataframe(
                pd.DataFrame(result["rejected"]), width="stretch", hide_index=True
            )
        st.caption(
            "Import behavior: valid rows are inserted in a single database transaction. "
            "Invalid rows and duplicate-SKU rows are rejected and shown above; nothing "
            "is silently skipped. Rejected rows are not inserted."
        )
        st.session_state.inv_import_result = None


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def render() -> None:
    _init_state()
    tabs = st.tabs(
        ["Add Product", "View Products", "Edit Product", "Delete Product", "Import CSV"]
    )
    with tabs[0]:
        render_add_product()
    with tabs[1]:
        render_view_products()
    with tabs[2]:
        render_edit_product()
    with tabs[3]:
        render_delete_product()
    with tabs[4]:
        render_csv_import()
