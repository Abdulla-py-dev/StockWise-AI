"""Sales analytics service for StockWise AI.

All aggregations run over real ``sales``/``sale_items`` records. Money values
are stored as integer paise; conversions to INR happen at the service boundary.
Empty databases return empty results / zero totals honestly.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

from database.db import get_connection, DEFAULT_DB_PATH


def _conn(db_path):
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    return get_connection(db_path)


def _paise_to_inr(p):
    return round(int(p or 0) / 100, 2)


def daily_revenue(db_path=None, start_date=None, end_date=None) -> List[dict]:
    conn = _conn(db_path)
    try:
        q = ("SELECT substr(transaction_date,1,10) AS d, COUNT(*) AS txns, "
             "COALESCE(SUM(total_amount),0) AS rev FROM sales WHERE 1=1")
        params = []
        if start_date:
            q += " AND transaction_date >= ?"; params.append(start_date)
        if end_date:
            q += " AND transaction_date <= ?"; params.append(end_date)
        q += " GROUP BY d ORDER BY d"
        rows = conn.execute(q, params).fetchall()
        return [{"date": r["d"], "transactions": r["txns"],
                 "revenue_paise": int(r["rev"]), "revenue_inr": _paise_to_inr(r["rev"])}
                for r in rows]
    finally:
        conn.close()


def weekly_revenue(db_path=None) -> List[dict]:
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            "SELECT strftime('%Y-%W', transaction_date) AS wk, COUNT(*) AS txns, "
            "COALESCE(SUM(total_amount),0) AS rev FROM sales GROUP BY wk ORDER BY wk"
        ).fetchall()
        return [{"week": r["wk"], "transactions": r["txns"],
                 "revenue_paise": int(r["rev"]), "revenue_inr": _paise_to_inr(r["rev"])}
                for r in rows]
    finally:
        conn.close()


def monthly_revenue(db_path=None) -> List[dict]:
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            "SELECT strftime('%Y-%m', transaction_date) AS m, COUNT(*) AS txns, "
            "COALESCE(SUM(total_amount),0) AS rev FROM sales GROUP BY m ORDER BY m"
        ).fetchall()
        return [{"month": r["m"], "transactions": r["txns"],
                 "revenue_paise": int(r["rev"]), "revenue_inr": _paise_to_inr(r["rev"])}
                for r in rows]
    finally:
        conn.close()


def transaction_counts(db_path=None) -> dict:
    conn = _conn(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) AS c FROM sales").fetchone()["c"]
        items = conn.execute("SELECT COALESCE(SUM(quantity_sold),0) AS s FROM sale_items").fetchone()["s"]
        return {"total_transactions": int(total), "total_units_sold": int(items)}
    finally:
        conn.close()


def best_selling_products(db_path=None, limit=10) -> List[dict]:
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            """SELECT COALESCE(product_id, -1) AS pid,
                      product_sku_snapshot AS sku, product_name_snapshot AS name,
                      SUM(quantity_sold) AS units,
                      SUM(line_total) AS revenue_paise,
                      SUM((unit_selling_price - unit_cost_at_sale) * quantity_sold) AS margin_paise
               FROM sale_items GROUP BY pid, sku, name
               ORDER BY units DESC LIMIT ?""", (limit,)
        ).fetchall()
        return [{"product_id": (r["pid"] if r["pid"] != -1 else None),
                 "sku": r["sku"], "name": r["name"],
                 "units_sold": int(r["units"]),
                 "revenue_paise": int(r["revenue_paise"]),
                 "revenue_inr": _paise_to_inr(r["revenue_paise"]),
                 "margin_paise": int(r["margin_paise"]),
                 "margin_inr": _paise_to_inr(r["margin_paise"])} for r in rows]
    finally:
        conn.close()


def slow_moving_products(db_path=None, days=30, limit=20) -> List[dict]:
    """Products with sales below a threshold in the last `days` window.

    A product is 'slow-moving' if its sold units in the window are 0 or below
    the median sold units across products that had any sale. Products with no
    sales at all are included.
    """
    conn = _conn(db_path)
    try:
        # units sold per product in window
        rows = conn.execute(
            """SELECT i.id, i.sku, i.name, i.quantity AS stock, i.expiry_date,
                      COALESCE(SUM(si.quantity_sold),0) AS sold
               FROM inventory i
               LEFT JOIN sale_items si ON si.product_id = i.id
               LEFT JOIN sales s ON s.id = si.sale_id
                 AND s.transaction_date >= date('now', ?)
               GROUP BY i.id""", (f"-{int(days)} days",)
        ).fetchall()
        result = []
        for r in rows:
            result.append({
                "product_id": r["id"], "sku": r["sku"], "name": r["name"],
                "stock": r["stock"], "expiry_date": r["expiry_date"],
                "units_sold_in_window": int(r["sold"]),
            })
        # threshold: median of sold among those with sold>0
        sold_vals = sorted([x["units_sold_in_window"] for x in result if x["units_sold_in_window"] > 0])
        if sold_vals:
            mid = len(sold_vals) // 2
            threshold = sold_vals[mid] if len(sold_vals) % 2 else (sold_vals[mid-1] + sold_vals[mid]) / 2
        else:
            threshold = 0
        slow = [x for x in result if x["units_sold_in_window"] <= threshold]
        slow.sort(key=lambda x: x["units_sold_in_window"])
        return slow[:limit]
    finally:
        conn.close()


def product_sales_velocity(db_path=None, days=30) -> List[dict]:
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            """SELECT COALESCE(product_id,-1) AS pid, product_sku_snapshot AS sku,
                      product_name_snapshot AS name, SUM(quantity_sold) AS units
               FROM sale_items si JOIN sales s ON s.id = si.sale_id
               WHERE s.transaction_date >= date('now', ?)
               GROUP BY pid, sku, name""", (f"-{int(days)} days",)
        ).fetchall()
        out = []
        for r in rows:
            units = int(r["units"])
            out.append({
                "product_id": (r["pid"] if r["pid"] != -1 else None),
                "sku": r["sku"], "name": r["name"],
                "units_sold": units,
                "velocity_per_day": round(units / max(days, 1), 3),
            })
        out.sort(key=lambda x: x["velocity_per_day"], reverse=True)
        return out
    finally:
        conn.close()


def revenue_and_margin(db_path=None) -> dict:
    conn = _conn(db_path)
    try:
        row = conn.execute(
            """SELECT COALESCE(SUM(line_total),0) AS rev,
                      COALESCE(SUM((unit_selling_price - unit_cost_at_sale) * quantity_sold),0) AS margin
               FROM sale_items"""
        ).fetchone()
        rev = int(row["rev"])
        margin = int(row["margin"])
        return {
            "revenue_paise": rev, "revenue_inr": _paise_to_inr(rev),
            "gross_margin_paise": margin, "gross_margin_inr": _paise_to_inr(margin),
            "margin_percent": round((margin / rev * 100), 2) if rev else 0.0,
        }
    finally:
        conn.close()
