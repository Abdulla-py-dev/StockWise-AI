"""Smart action recommendation service for StockWise AI.

Provides product-specific, heuristic (rule-based) recommendations based on
expiry status and days remaining. Discount suggestions are configurable
constants and are clearly identified as heuristic suggestions, not guaranteed
optimal prices.

Expired or unsafe products are never recommended for sale or discount.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

from services.inventory_service import Product
from services.expiry_service import _ensure_expiry_fields, stock_cost_value


# Configurable discount tiers (heuristic suggestions only).
# Format: (max_days_inclusive, discount_percent, action_label)
# Expired products are handled separately and never receive a sell/discount rec.
DISCOUNT_TIERS = [
    (7, 20.0, "Urgent review: consider a promotion or discount (heuristic)"),
    (30, 10.0, "Consider promotional pricing and prioritise sales"),
]
DEFAULT_NO_ACTION_DISCOUNT = 0.0


def _days(product: Product) -> int:
    try:
        return int(product.days_until_expiry)
    except (TypeError, ValueError):
        return 0


def recommend(product: Product) -> Dict[str, Any]:
    """Return a recommendation dict for a single product.

    Keys: action, discount_percent, sell_allowed, explanation, priority.
    """
    if product is None:
        return {
            "action": "No data available",
            "discount_percent": 0.0,
            "sell_allowed": False,
            "explanation": "Product record is missing.",
            "priority": "Low",
        }

    _ensure_expiry_fields(product)
    status = product.expiry_status or "Low Risk"
    days = _days(product)
    value = stock_cost_value(product)

    if status == "Expired" or days < 0:
        return {
            "action": "Do not sell. Review and remove/return expired stock immediately.",
            "discount_percent": 0.0,
            "sell_allowed": False,
            "explanation": (
                "Product is expired. Selling expired goods is not recommended. "
                "Verify expiry date and arrange safe disposal or return."
            ),
            "priority": "Critical",
        }

    if days <= 7:
        discount = DISCOUNT_TIERS[0][1] if DISCOUNT_TIERS else 20.0
        return {
            "action": "Urgent review: promote or discount to accelerate sales.",
            "discount_percent": discount,
            "sell_allowed": True,
            "explanation": (
                f"Expires in {days} day(s). Suggested discount of {discount:.0f}% is a "
                f"heuristic starting point, not a guaranteed optimal price. "
                f"Stock cost value: {value:.2f}."
            ),
            "priority": "High",
        }

    if days <= 30:
        discount = DISCOUNT_TIERS[1][1] if len(DISCOUNT_TIERS) > 1 else 10.0
        return {
            "action": "Consider promotional pricing and prioritise sales.",
            "discount_percent": discount,
            "sell_allowed": True,
            "explanation": (
                f"Expires in {days} day(s). A modest promotional discount "
                f"({discount:.0f}%, heuristic) may help move stock before expiry. "
                f"Stock cost value: {value:.2f}."
            ),
            "priority": "Medium",
        }

    return {
        "action": "Normal inventory monitoring. No immediate action required.",
        "discount_percent": 0.0,
        "sell_allowed": True,
        "explanation": (
            f"Expires in {days} day(s), beyond the risk horizon. Continue routine "
            f"monitoring. Stock cost value: {value:.2f}."
        ),
        "priority": "Low",
    }


def recommend_all(products) -> list:
    return [recommend(p) for p in (products or [])]
