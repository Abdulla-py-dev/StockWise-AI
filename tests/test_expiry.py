"""Tests for Phase 5 expiry intelligence services."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date, timedelta

from database.db import init_database
from services.inventory_service import add_product
from services.expiry_service import (
    prioritize_products,
    financial_analysis,
    stock_cost_value,
    RISK_PRIORITY,
)
from services.recommendation_service import recommend, DISCOUNT_TIERS


def _temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_database(path)
    return path


def _add(db_path, days, qty=10, cost=25.0, sku=None):
    today = date.today()
    expiry = (today + timedelta(days=days)).isoformat()
    return add_product(
        sku=sku or f"SKU{days}",
        name=f"Prod{days}",
        category="Cat",
        quantity=qty,
        unit_cost=cost,
        expiry_date=expiry,
        db_path=db_path,
    )


class ExpiryServiceTests(unittest.TestCase):
    def setUp(self):
        self.db = _temp_db()
        self.today = date.today()

    def test_boundaries_classification(self):
        # expired (-1), 0, 7, 8, 30, 31
        for d in [-1, 0, 7, 8, 30, 31]:
            p = _add(self.db, d, sku=f"B{d}")
            from services.inventory_service import get_product
            pp = get_product(p.id, db_path=self.db)
            expected = None
            if d < 0:
                expected = "Expired"
            elif d <= 7:
                expected = "High Risk"
            elif d <= 30:
                expected = "Medium Risk"
            else:
                expected = "Low Risk"
            self.assertEqual(pp.expiry_status, expected, msg=f"days={d}")

    def test_prioritization_order(self):
        _add(self.db, 40, sku="LOW")
        _add(self.db, 5, sku="HIGH")
        _add(self.db, -2, sku="EXP")
        _add(self.db, 15, sku="MED")
        from services.inventory_service import list_products
        prods = list_products(db_path=self.db)
        ordered = prioritize_products(prods)
        statuses = [p.expiry_status for p in ordered]
        # Expired first
        self.assertEqual(statuses[0], "Expired")
        # then high, medium, low
        self.assertLessEqual(statuses.index("High Risk"), statuses.index("Medium Risk"))
        self.assertLessEqual(statuses.index("Medium Risk"), statuses.index("Low Risk"))

    def test_stock_cost_value(self):
        p = _add(self.db, 5, qty=4, cost=2.5, sku="V1")
        from services.inventory_service import get_product
        pp = get_product(p.id, db_path=self.db)
        self.assertEqual(stock_cost_value(pp), 10.0)

    def test_financial_analysis_empty(self):
        from services.inventory_service import list_products
        fin = financial_analysis(list_products(db_path=self.db))
        self.assertEqual(fin["total_cost_value"], 0.0)
        self.assertEqual(fin["expired_value"], 0.0)
        self.assertEqual(fin["exposed_value"], 0.0)

    def test_financial_analysis_values(self):
        _add(self.db, -1, qty=2, cost=10.0, sku="E")   # expired 20
        _add(self.db, 5, qty=4, cost=5.0, sku="H")     # high 20
        _add(self.db, 15, qty=1, cost=100.0, sku="M")  # medium 100
        _add(self.db, 60, qty=10, cost=1.0, sku="L")   # low 10
        from services.inventory_service import list_products
        fin = financial_analysis(list_products(db_path=self.db))
        self.assertEqual(fin["total_cost_value"], 150.0)
        self.assertEqual(fin["expired_value"], 20.0)
        self.assertEqual(fin["high_risk_value"], 20.0)
        self.assertEqual(fin["medium_risk_value"], 100.0)
        self.assertEqual(fin["low_risk_value"], 10.0)
        self.assertEqual(fin["exposed_value"], 140.0)

    def test_missing_data_safe(self):
        from services.inventory_service import Product
        p = Product(id=None, sku="X", name="X", category="C", quantity=None, unit_cost=None, expiry_date="bad-date")
        # should not raise
        prioritize_products([p])
        fin = financial_analysis([p])
        self.assertIn("total_cost_value", fin)
        rec = recommend(p)
        self.assertIsInstance(rec, dict)
        self.assertIn("action", rec)


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.db = _temp_db()

    def test_expired_no_sell_no_discount(self):
        p = _add(self.db, -3, sku="EXP")
        from services.inventory_service import get_product
        pp = get_product(p.id, db_path=self.db)
        rec = recommend(pp)
        self.assertFalse(rec["sell_allowed"])
        self.assertEqual(rec["discount_percent"], 0.0)
        self.assertIn("Do not sell", rec["action"])

    def test_discount_rules(self):
        for d, expected_disc in [(0, 20.0), (7, 20.0), (8, 10.0), (30, 10.0), (31, 0.0)]:
            p = _add(self.db, d, sku=f"D{d}")
            from services.inventory_service import get_product
            pp = get_product(p.id, db_path=self.db)
            rec = recommend(pp)
            self.assertEqual(rec["discount_percent"], expected_disc, msg=f"days={d}")
            if d >= 0:
                self.assertTrue(rec["sell_allowed"])

    def test_empty_inventory_recommend_all(self):
        from services.inventory_service import list_products
        recs = recommend.__module__ and __import__("services.recommendation_service", fromlist=["recommend_all"]).recommend_all(list_products(db_path=self.db))
        self.assertEqual(recs, [])


if __name__ == "__main__":
    unittest.main()
