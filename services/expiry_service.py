"""Expiry intelligence service for StockWise AI.

Provides expiry prioritization and financial (cost-value) analysis built on top
of the existing inventory service. No database schema changes are made; all
values come from existing ``inventory`` records (quantity, unit_cost, expiry_date).

Expiry status is reused from ``services.inventory_service`` (``classify_expiry``)
and is never recomputed independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Optional

from services.inventory_service import Product, _compute_expiry_fields, classify_expiry


# Risk ordering for prioritization: expired first, then high, medium, low.
RISK_PRIORITY = {"Expired": 0, "High Risk": 1, "Medium Risk": 2, "Low Risk": 3}


def _today() -> date:
    return date.today()


def _ensure_expiry_fields(product: Product, today: Optional[date] = None) -> Product:
    """Ensure a Product has computed expiry fields without duplicating logic.

    If ``days_until_expiry``/``expiry_status`` are already set they are kept;
    otherwise they are derived via the shared ``_compute_expiry_fields`` helper.
    """
    if today is None:
        today = _today()
    if product.days_until_expiry is None or product.expiry_status is None:
        try:
            _compute_expiry_fields(product, today)
        except (ValueError, TypeError):
            # Leave fields as-is on unparseable expiry date.
            pass
    if product.expiry_status is None:
        product.expiry_status = "Low Risk"
    if product.days_until_expiry is None:
        product.days_until_expiry = 0
    return product


def stock_cost_value(product: Product) -> float:
    """Cost value of a product's stock: quantity * unit_cost (rounded)."""
    try:
        qty = int(product.quantity)
    except (TypeError, ValueError):
        qty = 0
    try:
        cost = float(product.unit_cost)
    except (TypeError, ValueError):
        cost = 0.0
    if qty < 0:
        qty = 0
    if cost < 0:
        cost = 0.0
    return round(qty * cost, 2)


def _risk_rank(product: Product) -> int:
    return RISK_PRIORITY.get(product.expiry_status or "Low Risk", 9)


def prioritize_products(products: Iterable[Product]) -> List[Product]:
    """Return products sorted by expiry priority.

    Expired first, then High/Medium/Low Risk. Within each risk group, higher
    stock cost value comes first. Invalid/missing entries are skipped.
    """
    today = _today()
    valid: List[Product] = []
    for p in products or []:
        if p is None:
            continue
        try:
            _ensure_expiry_fields(p, today)
            valid.append(p)
        except Exception:
            continue
    # stable sort: first by stock value desc within group, then risk asc
    valid.sort(key=lambda p: stock_cost_value(p), reverse=True)
    valid.sort(key=_risk_rank)
    return valid


def financial_analysis(products: Iterable[Product]) -> dict:
    """Compute cost-value financial exposure from inventory records.

    All values are COST values (quantity * unit_cost), not revenue or loss.
    Cost exposure indicates stock at risk of expiring; it is NOT a claim that
    all at-risk stock will become a financial loss.
    """
    total_cost_value = 0.0
    expired_value = 0.0
    high_risk_value = 0.0
    medium_risk_value = 0.0
    low_risk_value = 0.0

    today = _today()
    for p in products or []:
        if p is None:
            continue
        try:
            _ensure_expiry_fields(p, today)
            value = stock_cost_value(p)
        except Exception:
            continue
        total_cost_value += value
        status = p.expiry_status or "Low Risk"
        if status == "Expired":
            expired_value += value
        elif status == "High Risk":
            high_risk_value += value
        elif status == "Medium Risk":
            medium_risk_value += value
        else:
            low_risk_value += value

    exposed = round(expired_value + high_risk_value + medium_risk_value, 2)
    return {
        "total_cost_value": round(total_cost_value, 2),
        "expired_value": round(expired_value, 2),
        "high_risk_value": round(high_risk_value, 2),
        "medium_risk_value": round(medium_risk_value, 2),
        "low_risk_value": round(low_risk_value, 2),
        "exposed_value": exposed,
        # Cost exposure is potential at-risk value, not confirmed loss.
        "potential_loss_note": (
            "Cost exposure is the cost value of stock that is expired or nearing "
            "expiry. It is not a confirmed financial loss; actual loss depends on "
            "whether the stock is sold, discounted, donated, or discarded."
        ),
    }
