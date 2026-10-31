"""Tests for Phase 5 Step 4 analytics service."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date, timedelta

from database.db import init_database
from services.inventory_service import add_product
from services.sales_service import SaleItemInput, record_sale
from services.analytics_service import (
    best_selling_products,
    daily_revenue,
    monthly_revenue,
    product_sales_velocity,
    revenue_and_margin,
    slow_moving_products,
    transaction_counts,
    weekly_revenue,
)


def _temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_database(path)
    return path


def _add(db, sku="A", name="Milk", qty=10, cost=25.0, days=30):
    today = date.today()
    return add_product(sku=sku, name=name, category="Dairy", quantity=qty,
                       unit_cost=cost, expiry_date=(today + timedelta(days=days)).isoformat(),
                       db_path=db)


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.db = _temp_db()

    def test_empty_database(self):
        self.assertEqual(transaction_counts(db_path=self.db)["total_transactions"], 0)
        self.assertEqual(daily_revenue(db_path=self.db), [])
        self.assertEqual(monthly_revenue(db_path=self.db), [])
        self.assertEqual(weekly_revenue(db_path=self.db), [])
        self.assertEqual(best_selling_products(db_path=self.db), [])
        self.assertEqual(revenue_and_margin(db_path=self.db)["revenue_paise"], 0)

    def test_daily_monthly_revenue(self):
        p = _add(self.db)
        record_sale([SaleItemInput(product_id=p.id, quantity=2, unit_selling_price=30.0)], db_path=self.db)
        dr = daily_revenue(db_path=self.db)
        self.assertEqual(len(dr), 1)
        self.assertEqual(dr[0]["revenue_paise"], 6000)  # 30.00 INR * 2 = 6000 paise
        mr = monthly_revenue(db_path=self.db)
        self.assertEqual(len(mr), 1)

    def test_best_selling_and_margin(self):
        p = _add(self.db, cost=25.0)
        record_sale([SaleItemInput(product_id=p.id, quantity=3, unit_selling_price=40.0)], db_path=self.db)
        bs = best_selling_products(db_path=self.db)
        self.assertEqual(bs[0]["units_sold"], 3)
        rm = revenue_and_margin(db_path=self.db)
        self.assertEqual(rm["gross_margin_paise"], (4000 - 2500) * 3)

    def test_slow_moving_and_velocity(self):
        p = _add(self.db)
        record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)], db_path=self.db)
        vel = product_sales_velocity(db_path=self.db, days=30)
        self.assertEqual(len(vel), 1)
        sm = slow_moving_products(db_path=self.db, days=30)
        self.assertIsInstance(sm, list)


if __name__ == "__main__":
    unittest.main()
