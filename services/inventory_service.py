"""Inventory service: CRUD operations, queries, metrics, and CSV import.

Database operations live here and out of UI code. All SQL is parameterized.
A temporary database path can be supplied for testing so the real inventory
database is never touched.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Optional

from database.db import DEFAULT_DB_PATH, get_connection
from utils.validators import (
    ValidationError,
    validate_category,
    validate_expiry_date,
    validate_name,
    validate_quantity,
    validate_sku,
    validate_unit_cost,
)


# ---- Expiry risk thresholds (central, documented, configurable) ----
#
# days_until_expiry = (expiry_date - today).days
#   days < 0          -> Expired            (already past expiry)
#   0  <= days <= 7   -> High Risk          (expiring within the threshold)
#   8  <= days <= 30  -> Medium Risk
#   days > 30         -> Low Risk
#
# Expiry status describes how close stock is to expiry. It does NOT establish
# that stock will remain unsold; demand forecasting is a later phase.

HIGH_RISK_DAYS = 7
MEDIUM_RISK_DAYS = 30


@dataclass
class Product:
    id: Optional[int]
    sku: str
    name: str
    category: str
    quantity: int
    unit_cost: float
    expiry_date: str  # ISO YYYY-MM-DD
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    days_until_expiry: Optional[int] = None
    expiry_status: Optional[str] = None


def classify_expiry(days_until_expiry: int) -> str:
    """Classify expiry status from days until expiry.

    Already-expired products (negative days) are treated separately from
    products that are merely approaching expiry.
    """
    if days_until_expiry < 0:
        return "Expired"
    if days_until_expiry <= HIGH_RISK_DAYS:
        return "High Risk"
    if days_until_expiry <= MEDIUM_RISK_DAYS:
        return "Medium Risk"
    return "Low Risk"


def _row_to_product(row) -> Product:
    return Product(
        id=row["id"],
        sku=row["sku"],
        name=row["name"],
        category=row["category"],
        quantity=row["quantity"],
        unit_cost=row["unit_cost"],
        expiry_date=row["expiry_date"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _compute_expiry_fields(product: Product, today: date) -> Product:
    expiry = date.fromisoformat(product.expiry_date)
    product.days_until_expiry = (expiry - today).days
    product.expiry_status = classify_expiry(product.days_until_expiry)
    return product


# ---- CRUD ----

def add_product(sku, name, category, quantity, unit_cost, expiry_date, db_path=DEFAULT_DB_PATH) -> Product:
    """Add a new product. Raises ValidationError on invalid input or duplicate SKU."""
    sku = validate_sku(sku)
    name = validate_name(name)
    category = validate_category(category)
    quantity = validate_quantity(quantity)
    unit_cost = validate_unit_cost(unit_cost)
    expiry_date = validate_expiry_date(expiry_date)

    conn = get_connection(db_path)
    try:
        existing = conn.execute("SELECT id FROM inventory WHERE sku = ?", (sku,)).fetchone()
        if existing is not None:
            raise ValidationError(f"A product with SKU '{sku}' already exists.")
        cur = conn.execute(
            "INSERT INTO inventory (sku, name, category, quantity, unit_cost, expiry_date) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sku, name, category, quantity, unit_cost, expiry_date),
        )
        conn.commit()
        return get_product(cur.lastrowid, db_path=db_path)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise ValidationError(f"A product with SKU '{sku}' already exists.") from exc
    finally:
        conn.close()


def get_product(product_id, db_path=DEFAULT_DB_PATH) -> Optional[Product]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM inventory WHERE id = ?", (product_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    product = _row_to_product(row)
    return _compute_expiry_fields(product, date.today())


def list_products(
    search=None,
    category=None,
    expiry_status=None,
    sort_by="name",
    sort_order="asc",
    db_path=DEFAULT_DB_PATH,
):
    """Return products as Product dataclass instances with computed expiry fields.

    Searching, category/status filtering, and sorting are applied in Python so
    expiry-status classification (which depends on today's date) stays
    consistent everywhere. Suitable for a local single-user inventory.
    """
    today = date.today()
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM inventory").fetchall()
    finally:
        conn.close()

    products = [_compute_expiry_fields(_row_to_product(row), today) for row in rows]

    if search:
        term = search.lower()
        products = [
            p for p in products if term in p.sku.lower() or term in p.name.lower()
        ]

    if category:
        products = [p for p in products if p.category == category]

    if expiry_status and expiry_status != "All":
        products = [p for p in products if p.expiry_status == expiry_status]

    reverse = sort_order == "desc"
    sort_keys = {
        "sku": lambda p: p.sku.lower(),
        "name": lambda p: p.name.lower(),
        "category": lambda p: p.category.lower(),
        "quantity": lambda p: p.quantity,
        "unit_cost": lambda p: p.unit_cost,
        "expiry_date": lambda p: p.expiry_date,
    }
    key = sort_keys.get(sort_by, sort_keys["name"])
    products.sort(key=key, reverse=reverse)

    return products


def get_categories(db_path=DEFAULT_DB_PATH):
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT category FROM inventory ORDER BY category").fetchall()
    finally:
        conn.close()
    return [row["category"] for row in rows]


def update_product(
    product_id,
    sku,
    name,
    category,
    quantity,
    unit_cost,
    expiry_date,
    db_path=DEFAULT_DB_PATH,
) -> Product:
    sku = validate_sku(sku)
    name = validate_name(name)
    category = validate_category(category)
    quantity = validate_quantity(quantity)
    unit_cost = validate_unit_cost(unit_cost)
    expiry_date = validate_expiry_date(expiry_date)

    conn = get_connection(db_path)
    try:
        if conn.execute("SELECT id FROM inventory WHERE id = ?", (product_id,)).fetchone() is None:
            raise ValidationError(f"Product id {product_id} not found.")
        existing = conn.execute(
            "SELECT id FROM inventory WHERE sku = ? AND id != ?", (sku, product_id)
        ).fetchone()
        if existing is not None:
            raise ValidationError(f"A product with SKU '{sku}' already exists.")
        conn.execute(
            "UPDATE inventory SET sku = ?, name = ?, category = ?, quantity = ?, "
            "unit_cost = ?, expiry_date = ? WHERE id = ?",
            (sku, name, category, quantity, unit_cost, expiry_date, product_id),
        )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise ValidationError(f"A product with SKU '{sku}' already exists.") from exc
    finally:
        conn.close()
    return get_product(product_id, db_path=db_path)


def delete_product(product_id, db_path=DEFAULT_DB_PATH) -> bool:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("DELETE FROM inventory WHERE id = ?", (product_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ---- Dashboard metrics ----

def get_inventory_metrics(db_path=DEFAULT_DB_PATH) -> dict:
    """Return real inventory metrics read from the SQLite database.

    No sample or illustrative values are used. When the database is empty all
    numeric values are zero.
    """
    today = date.today()
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM inventory").fetchall()
    finally:
        conn.close()

    products = [_compute_expiry_fields(_row_to_product(row), today) for row in rows]

    total_products = len(products)
    high_risk = [p for p in products if p.expiry_status == "High Risk"]
    expired = [p for p in products if p.expiry_status == "Expired"]
    total_units = sum(p.quantity for p in products)
    # Current value (qty x unit cost) of high-expiry-risk stock. This is the
    # value of stock approaching expiry, not a prediction of unsold units.
    cost_exposure = round(sum(p.quantity * p.unit_cost for p in high_risk), 2)

    return {
        "total_products": total_products,
        "high_risk_products": len(high_risk),
        "expired_products": len(expired),
        "total_units": total_units,
        "cost_exposure": cost_exposure,
    }


# ---- CSV import ----

def import_products_from_rows(rows, db_path=DEFAULT_DB_PATH):
    """Import validated product dicts in a single database transaction.

    Returns ``(imported_count, rejected_rows)`` where each rejected row is a
    dict with an added ``error`` key. Valid rows are inserted; duplicate SKUs
    (within the batch or already in the database) are rejected and reported.
    The transaction is committed only at the end and rolled back on error.
    """
    imported = 0
    rejected = []
    conn = get_connection(db_path)
    try:
        with conn:
            seen_skus = set()
            existing_skus = {
                row["sku"]
                for row in conn.execute("SELECT sku FROM inventory").fetchall()
            }
            for row in rows:
                sku = row["sku"]
                if sku in seen_skus or sku in existing_skus:
                    rejected.append({**row, "error": "Duplicate SKU"})
                    continue
                try:
                    conn.execute(
                        "INSERT INTO inventory (sku, name, category, quantity, unit_cost, expiry_date) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            sku,
                            row["name"],
                            row["category"],
                            row["quantity"],
                            row["unit_cost"],
                            row["expiry_date"],
                        ),
                    )
                    seen_skus.add(sku)
                    imported += 1
                except sqlite3.IntegrityError as exc:
                    rejected.append({**row, "error": f"Database error: {exc}"})
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return imported, rejected
