"""Sales data quality, ML readiness and demand-forecasting preparation.

All reads are read-only. No sales data is fabricated. Forecasting is NOT
performed here; only a readiness verdict and a clean historical time-series
dataset are produced.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

from database.db import get_connection, DEFAULT_DB_PATH


def _conn(db_path):
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    return get_connection(db_path)


# Documented readiness criteria.
MIN_TRANSACTIONS = 30
MIN_DAYS_COVERAGE = 10
MIN_PRODUCTS_WITH_SALES = 3


def data_quality(db_path=None) -> dict:
    conn = _conn(db_path)
    try:
        txns = conn.execute("SELECT COUNT(*) AS c FROM sales").fetchone()["c"]
        items = conn.execute("SELECT COUNT(*) AS c FROM sale_items").fetchone()["c"]
        distinct_products = conn.execute(
            "SELECT COUNT(DISTINCT product_id) FROM sale_items WHERE product_id IS NOT NULL"
        ).fetchone()[0]
        historical_without_product = conn.execute(
            "SELECT COUNT(*) FROM sale_items WHERE product_id IS NULL"
        ).fetchone()[0]
        missing_snapshots = conn.execute(
            "SELECT COUNT(*) FROM sale_items WHERE product_sku_snapshot IS NULL OR product_name_snapshot IS NULL"
        ).fetchone()[0]
        invalid_qty = conn.execute(
            "SELECT COUNT(*) FROM sale_items WHERE quantity_sold <= 0"
        ).fetchone()[0]
        invalid_price = conn.execute(
            "SELECT COUNT(*) FROM sale_items WHERE unit_selling_price < 0 OR unit_cost_at_sale < 0 OR line_total < 0"
        ).fetchone()[0]
        dup_uuids = conn.execute(
            "SELECT COUNT(*) FROM (SELECT transaction_uuid FROM sales GROUP BY transaction_uuid HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        # date range
        row = conn.execute("SELECT MIN(transaction_date), MAX(transaction_date) FROM sales").fetchone()
        first, last = row[0], row[1]
        days_coverage = 0
        if first and last:
            try:
                d0 = date.fromisoformat(first[:10])
                d1 = date.fromisoformat(last[:10])
                days_coverage = (d1 - d0).days + 1
            except ValueError:
                days_coverage = 0
        missing_dates = 0
        if first and last:
            # count distinct sale dates
            distinct_days = conn.execute(
                "SELECT COUNT(DISTINCT substr(transaction_date,1,10)) FROM sales"
            ).fetchone()[0]
            missing_dates = max(0, days_coverage - int(distinct_days))

        issues = []
        if missing_snapshots:
            issues.append(f"{missing_snapshots} sale item(s) missing product snapshots.")
        if invalid_qty:
            issues.append(f"{invalid_qty} sale item(s) with invalid quantity.")
        if invalid_price:
            issues.append(f"{invalid_price} sale item(s) with invalid prices.")
        if dup_uuids:
            issues.append(f"{dup_uuids} duplicate transaction UUID(s).")
        # A calendar day with zero recorded sales is a legitimate observation,
        # not a data-quality defect. build_timeseries() zero-fills such days,
        # so missing_calendar_days is reported for transparency only and does
        # not affect quality_ok or ML eligibility.

        return {
            "transactions": int(txns),
            "sale_items": int(items),
            "distinct_products_with_sales": int(distinct_products),
            "historical_items_without_product": int(historical_without_product),
            "historical_product_note": (
                "Historical sales with a deleted product reference (product_id IS NULL) "
                "are preserved via snapshots but are NOT counted as current distinct products."
            ),
            "missing_snapshots": int(missing_snapshots),
            "invalid_quantities": int(invalid_qty),
            "invalid_prices": int(invalid_price),
            "duplicate_uuids": int(dup_uuids),
            "first_transaction_date": first,
            "last_transaction_date": last,
            "days_coverage": int(days_coverage),
            "missing_calendar_days": int(missing_dates),
            "issues": issues,
            "quality_ok": len(issues) == 0,
        }
    finally:
        conn.close()


def ml_readiness(db_path=None) -> dict:
    q = data_quality(db_path)
    eligible = (
        q["transactions"] >= MIN_TRANSACTIONS
        and q["days_coverage"] >= MIN_DAYS_COVERAGE
        and q["distinct_products_with_sales"] >= MIN_PRODUCTS_WITH_SALES
        and q["quality_ok"]
    )
    reasons = []
    if q["transactions"] < MIN_TRANSACTIONS:
        reasons.append(f"Need at least {MIN_TRANSACTIONS} transactions (have {q['transactions']}).")
    if q["days_coverage"] < MIN_DAYS_COVERAGE:
        reasons.append(f"Need at least {MIN_DAYS_COVERAGE} days of history (have {q['days_coverage']}).")
    if q["distinct_products_with_sales"] < MIN_PRODUCTS_WITH_SALES:
        reasons.append(f"Need at least {MIN_PRODUCTS_WITH_SALES} products with sales (have {q['distinct_products_with_sales']}).")
    if not q["quality_ok"]:
        reasons.append("Data quality issues present.")
    return {
        "eligible": eligible,
        "criteria": {
            "min_transactions": MIN_TRANSACTIONS,
            "min_days_coverage": MIN_DAYS_COVERAGE,
            "min_products_with_sales": MIN_PRODUCTS_WITH_SALES,
        },
        "current": {
            "transactions": q["transactions"],
            "days_coverage": q["days_coverage"],
            "products_with_sales": q["distinct_products_with_sales"],
        },
        "reasons": reasons,
        "message": ("Forecasting eligible." if eligible
                    else "Insufficient genuine historical data for demand forecasting. "
                         "Forecasting disabled until criteria are met."),
    }


def build_timeseries(db_path=None, product_id=None) -> List[dict]:
    """Build a daily product-level units-sold time series from real sales.

    Missing days within the observed range are filled with 0 units so the
    series is contiguous. No fabricated transactions; quantities come from
    actual sale_items.
    """
    conn = _conn(db_path)
    try:
        # aggregate actual sales per product per day
        q = ("SELECT si.product_id, si.product_sku_snapshot, si.product_name_snapshot, "
             "substr(s.transaction_date,1,10) AS d, SUM(si.quantity_sold) AS units "
             "FROM sale_items si JOIN sales s ON s.id = si.sale_id")
        params = []
        if product_id is not None:
            q += " WHERE si.product_id = ?"
            params.append(product_id)
        q += " GROUP BY si.product_id, d ORDER BY d"
        rows = conn.execute(q, params).fetchall()

        # group by product
        by_product = {}
        for r in rows:
            pid = r["product_id"]
            by_product.setdefault(pid, {
                "product_id": pid,
                "sku": r["product_sku_snapshot"],
                "name": r["product_name_snapshot"],
                "days": {},
            })
            by_product[pid]["days"][r["d"]] = int(r["units"])

        result = []
        for pid, info in by_product.items():
            if not info["days"]:
                continue
            dates = sorted(info["days"].keys())
            start = date.fromisoformat(dates[0])
            end = date.fromisoformat(dates[-1])
            cur = start
            while cur <= end:
                ds = cur.isoformat()
                result.append({
                    "product_id": info["product_id"],
                    "sku": info["sku"],
                    "name": info["name"],
                    "date": ds,
                    "units_sold": info["days"].get(ds, 0),
                })
                cur += timedelta(days=1)
        result.sort(key=lambda x: (x["product_id"] or 0, x["date"]))
        return result
    finally:
        conn.close()
