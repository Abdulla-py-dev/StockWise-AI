"""Sales service: recording sales, querying history, and sales metrics.

All money is stored as integer paise (1 INR = 100 paise) to avoid
floating-point inaccuracies. Conversion from INR (REAL) happens at the
service boundary.

Transactions are atomic: a sale and its stock reductions either both
commit or both roll back.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import List, Optional, Sequence

from database.db import get_connection
from services.inventory_service import get_product, classify_expiry
from utils.validators import ValidationError


INR_TO_PAISE = 100


def to_paise(amount) -> int:
    """Convert an INR amount (int/float/str) to integer paise."""
    try:
        value = Decimal(str(amount).replace(",", ""))
    except (TypeError, ValueError, InvalidOperation):
        raise ValidationError("Amount must be a number.")
    # Round to nearest paise using exact decimal arithmetic.
    return int((value * INR_TO_PAISE).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def from_paise(paise) -> float:
    return round(int(paise) / INR_TO_PAISE, 2)


@dataclass
class SaleItemInput:
    product_id: int
    quantity: int
    unit_selling_price: float  # INR


@dataclass
class SaleItem:
    id: int
    sale_id: int
    product_id: Optional[int]
    product_sku_snapshot: Optional[str]
    product_name_snapshot: Optional[str]
    quantity_sold: int
    unit_selling_price: int  # paise
    unit_cost_at_sale: int   # paise
    line_total: int          # paise


@dataclass
class Sale:
    id: int
    transaction_uuid: str
    transaction_date: str
    total_amount: int  # paise
    items: List[SaleItem]


def _row_to_sale_item(row) -> SaleItem:
    return SaleItem(
        id=row["id"],
        sale_id=row["sale_id"],
        product_id=row["product_id"],
        product_sku_snapshot=row["product_sku_snapshot"],
        product_name_snapshot=row["product_name_snapshot"],
        quantity_sold=row["quantity_sold"],
        unit_selling_price=row["unit_selling_price"],
        unit_cost_at_sale=row["unit_cost_at_sale"],
        line_total=row["line_total"],
    )


def record_sale(
    items: Sequence[SaleItemInput],
    db_path=None,
    transaction_uuid: Optional[str] = None,
    transaction_date: Optional[str] = None,
) -> Sale:
    from database.db import DEFAULT_DB_PATH
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    if not items:
        raise ValidationError("Sale must contain at least one item.")

    if transaction_uuid is None:
        transaction_uuid = str(uuid.uuid4())

    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        # Validate and lock products.
        prepared = []
        for it in items:
            if it.quantity <= 0:
                raise ValidationError("Quantity must be greater than zero.")
            product = get_product(it.product_id, db_path=db_path)
            # get_product opens its own connection; re-fetch within this txn for lock.
            row = conn.execute(
                "SELECT * FROM inventory WHERE id = ?", (it.product_id,)
            ).fetchone()
            if row is None:
                raise ValidationError(f"Product id {it.product_id} not found.")
            if row["quantity"] < it.quantity:
                raise ValidationError(
                    f"Insufficient stock for '{row['name']}': requested {it.quantity}, available {row['quantity']}."
                )
            # Block expired products.
            from services.inventory_service import _compute_expiry_fields
            from datetime import date as _date
            p = None
            try:
                from services.inventory_service import Product as _P
                p = _P(
                    id=row["id"], sku=row["sku"], name=row["name"], category=row["category"],
                    quantity=row["quantity"], unit_cost=row["unit_cost"], expiry_date=row["expiry_date"],
                )
                _compute_expiry_fields(p, _date.today())
            except Exception:
                p = None
            if p is not None and p.expiry_status == "Expired":
                raise ValidationError(f"Cannot sell expired product '{row['name']}'.")
            unit_cost_paise = to_paise(row["unit_cost"])
            selling_paise = to_paise(it.unit_selling_price)
            line_total = selling_paise * it.quantity
            prepared.append({
                "product_id": row["id"],
                "sku": row["sku"],
                "name": row["name"],
                "quantity": it.quantity,
                "unit_selling_price": selling_paise,
                "unit_cost_at_sale": unit_cost_paise,
                "line_total": line_total,
            })

        total_amount = sum(p["line_total"] for p in prepared)

        # Insert sale header.
        try:
            cur = conn.execute(
                "INSERT INTO sales (transaction_uuid, transaction_date, total_amount) VALUES (?, ?, ?)",
                (transaction_uuid, transaction_date or datetime.now().isoformat(), total_amount),
            )
        except sqlite3.IntegrityError as exc:
            raise ValidationError("Duplicate transaction UUID.") from exc
        sale_id = cur.lastrowid

        # Insert items and decrement stock.
        for p in prepared:
            conn.execute(
                """INSERT INTO sale_items
                   (sale_id, product_id, product_sku_snapshot, product_name_snapshot,
                    quantity_sold, unit_selling_price, unit_cost_at_sale, line_total)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (sale_id, p["product_id"], p["sku"], p["name"], p["quantity"],
                 p["unit_selling_price"], p["unit_cost_at_sale"], p["line_total"]),
            )
            conn.execute(
                "UPDATE inventory SET quantity = quantity - ? WHERE id = ?",
                (p["quantity"], p["product_id"]),
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return get_sale(sale_id, db_path=db_path)


def list_sales(db_path=None, search=None, start_date=None, end_date=None) -> List[Sale]:
    from database.db import DEFAULT_DB_PATH
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM sales WHERE 1=1"
        params = []
        if start_date:
            query += " AND transaction_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND transaction_date <= ?"
            params.append(end_date)
        if search:
            query += " AND id IN (SELECT sale_id FROM sale_items WHERE product_name_snapshot LIKE ? OR product_sku_snapshot LIKE ?)"
            term = f"%{search}%"
            params.extend([term, term])
        query += " ORDER BY transaction_date DESC, id DESC"
        rows = conn.execute(query, params).fetchall()
        sales = []
        for r in rows:
            items_rows = conn.execute(
                "SELECT * FROM sale_items WHERE sale_id = ?", (r["id"],)
            ).fetchall()
            sales.append(Sale(
                id=r["id"],
                transaction_uuid=r["transaction_uuid"],
                transaction_date=r["transaction_date"],
                total_amount=r["total_amount"],
                items=[_row_to_sale_item(ir) for ir in items_rows],
            ))
        return sales
    finally:
        conn.close()


def get_sale(sale_id, db_path=None) -> Optional[Sale]:
    from database.db import DEFAULT_DB_PATH
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    conn = get_connection(db_path)
    try:
        r = conn.execute("SELECT * FROM sales WHERE id = ?", (sale_id,)).fetchone()
        if r is None:
            return None
        items_rows = conn.execute(
            "SELECT * FROM sale_items WHERE sale_id = ?", (sale_id,)
        ).fetchall()
        return Sale(
            id=r["id"],
            transaction_uuid=r["transaction_uuid"],
            transaction_date=r["transaction_date"],
            total_amount=r["total_amount"],
            items=[_row_to_sale_item(ir) for ir in items_rows],
        )
    finally:
        conn.close()


def get_sales_metrics(db_path=None) -> dict:
    from database.db import DEFAULT_DB_PATH
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    conn = get_connection(db_path)
    try:
        total_sales = conn.execute("SELECT COUNT(*) AS c FROM sales").fetchone()["c"]
        total_revenue = conn.execute("SELECT COALESCE(SUM(total_amount),0) AS s FROM sales").fetchone()["s"]
        total_items = conn.execute("SELECT COALESCE(SUM(quantity_sold),0) AS s FROM sale_items").fetchone()["s"]
        total_cost = conn.execute(
            "SELECT COALESCE(SUM(unit_cost_at_sale * quantity_sold),0) AS s FROM sale_items"
        ).fetchone()["s"]
        return {
            "total_sales": total_sales,
            "total_revenue_paise": int(total_revenue),
            "total_revenue_inr": from_paise(total_revenue),
            "total_items_sold": int(total_items),
            "total_cost_paise": int(total_cost),
            "gross_profit_paise": int(total_revenue) - int(total_cost),
        }
    finally:
        conn.close()
