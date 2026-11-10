"""Tests for Phase 5 Step 5 ML readiness & data quality."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date, timedelta

from database.db import get_connection, init_database
from services.inventory_service import add_product
from services.sales_service import SaleItemInput, record_sale
from services.ml_readiness_service import data_quality, ml_readiness, build_timeseries


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


class MLReadinessTests(unittest.TestCase):
    def setUp(self):
        self.db = _temp_db()

    def test_empty_database(self):
        q = data_quality(db_path=self.db)
        self.assertEqual(q["transactions"], 0)
        self.assertTrue(q["quality_ok"])
        r = ml_readiness(db_path=self.db)
        self.assertFalse(r["eligible"])
        self.assertEqual(build_timeseries(db_path=self.db), [])

    def test_one_transaction(self):
        p = _add(self.db)
        record_sale([SaleItemInput(product_id=p.id, quantity=2, unit_selling_price=30.0)], db_path=self.db)
        q = data_quality(db_path=self.db)
        self.assertEqual(q["transactions"], 1)
        self.assertEqual(q["distinct_products_with_sales"], 1)
        ts = build_timeseries(db_path=self.db)
        self.assertTrue(len(ts) >= 1)
        self.assertEqual(ts[0]["units_sold"], 2)

    def test_multiple_transactions_missing_dates(self):
        p = _add(self.db)
        record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)], db_path=self.db)
        # second sale on a different day impossible to set date easily; use transaction_date param
        record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)], db_path=self.db,
                    transaction_date="2020-01-01T00:00:00")
        q = data_quality(db_path=self.db)
        self.assertEqual(q["transactions"], 2)
        self.assertGreaterEqual(q["missing_calendar_days"], 0)
        ts = build_timeseries(db_path=self.db)
        # contiguous fill includes zeros for missing days
        units = [x["units_sold"] for x in ts]
        self.assertIn(0, units)

    def test_coverage_boundary(self):
        # 30 transactions spread over 9 days -> not eligible (coverage)
        p = _add(self.db, qty=60)
        base = date.today()
        for i in range(30):
            d = (base + timedelta(days=i % 9)).isoformat() + "T00:00:00"
            record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)],
                        db_path=self.db, transaction_date=d, transaction_uuid=str(__import__("uuid").uuid4()))
        # distinct products only 1 -> still not eligible, but check coverage reason not present for 9? We just check not eligible.
        r = ml_readiness(db_path=self.db)
        self.assertFalse(r["eligible"])
        # 10 days coverage with 3 products and 30 txns would be needed; coverage alone boundary tested via days_coverage
        q = data_quality(db_path=self.db)
        self.assertLessEqual(q["days_coverage"], 9)


    def test_null_product_handling(self):
        p = _add(self.db)
        record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)], db_path=self.db)
        from services.inventory_service import delete_product
        delete_product(p.id, db_path=self.db)
        q = data_quality(db_path=self.db)
        self.assertEqual(q["distinct_products_with_sales"], 0)
        self.assertEqual(q["historical_items_without_product"], 1)
        ts = build_timeseries(db_path=self.db)
        self.assertTrue(any(x["sku"] == p.sku for x in ts))


    def test_threshold_is_ten_days(self):
        r = ml_readiness(db_path=self.db)
        self.assertEqual(r["criteria"]["min_days_coverage"], 10)
        self.assertTrue(any("10 days" in reason for reason in r["reasons"]))

    def test_zero_sales_day_does_not_affect_quality_ok(self):
        """A legitimate zero-sales calendar day does not make quality_ok false."""
        p = _add(self.db)
        base = date.today()
        # Sale on day 1 and day 3 — day 2 is a legitimate zero-sales day.
        record_sale(
            [SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)],
            db_path=self.db,
            transaction_uuid=str(__import__("uuid").uuid4()),
            transaction_date=base.isoformat() + "T00:00:00",
        )
        record_sale(
            [SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)],
            db_path=self.db,
            transaction_uuid=str(__import__("uuid").uuid4()),
            transaction_date=(base + timedelta(days=2)).isoformat() + "T00:00:00",
        )
        q = data_quality(db_path=self.db)
        self.assertEqual(q["missing_calendar_days"], 1)
        self.assertTrue(q["quality_ok"])

    def test_missing_calendar_days_still_reported(self):
        """missing_calendar_days value is still returned even though it does not fail quality."""
        p = _add(self.db)
        base = date.today()
        record_sale(
            [SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)],
            db_path=self.db,
            transaction_uuid=str(__import__("uuid").uuid4()),
            transaction_date=base.isoformat() + "T00:00:00",
        )
        record_sale(
            [SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)],
            db_path=self.db,
            transaction_uuid=str(__import__("uuid").uuid4()),
            transaction_date=(base + timedelta(days=2)).isoformat() + "T00:00:00",
        )
        q = data_quality(db_path=self.db)
        self.assertEqual(q["missing_calendar_days"], 1)

    def test_genuine_quality_issues_still_fail_quality_ok(self):
        """Genuine data-quality issues (missing snapshot) still make quality_ok false."""
        p = _add(self.db)
        sale = record_sale(
            [SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)],
            db_path=self.db,
        )
        # Directly insert a sale item with missing product snapshots to trigger a real defect.
        # The product_sku_snapshot column is nullable TEXT, so this insert is allowed.
        conn = get_connection(self.db)
        try:
            conn.execute(
                "INSERT INTO sale_items (sale_id, product_id, product_sku_snapshot, product_name_snapshot, "
                "quantity_sold, unit_selling_price, unit_cost_at_sale, line_total) "
                "VALUES (?, ?, NULL, NULL, 1, 100, 100, 100)",
                (sale.id, p.id),
            )
            conn.commit()
        finally:
            conn.close()
        q = data_quality(db_path=self.db)
        self.assertFalse(q["quality_ok"])
        self.assertTrue(
            any("missing product snapshots" in issue for issue in q["issues"])
        )

    def test_ml_readiness_eligible_when_criteria_met(self):
        """ML readiness is eligible when numerical criteria and genuine quality checks pass."""
        products = []
        for i in range(3):
            p = _add(self.db, sku=f"P{i}", name=f"Product {i}", qty=100)
            products.append(p)
        base = date.today()
        # 10 transactions per product across 10 days = 30 transactions, 10 days coverage, 3 products.
        for p in products:
            for d in range(10):
                record_sale(
                    [SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)],
                    db_path=self.db,
                    transaction_uuid=str(__import__("uuid").uuid4()),
                    transaction_date=(base + timedelta(days=d)).isoformat() + "T00:00:00",
                )
        r = ml_readiness(db_path=self.db)
        self.assertTrue(r["eligible"])
        self.assertEqual(r["current"]["transactions"], 30)
        self.assertEqual(r["current"]["days_coverage"], 10)
        self.assertEqual(r["current"]["products_with_sales"], 3)

    def test_null_product_reference_handling_unchanged(self):
        """NULL product references are preserved but not counted as distinct products."""
        p = _add(self.db)
        record_sale(
            [SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)],
            db_path=self.db,
        )
        from services.inventory_service import delete_product
        delete_product(p.id, db_path=self.db)
        q = data_quality(db_path=self.db)
        self.assertEqual(q["distinct_products_with_sales"], 0)
        self.assertEqual(q["historical_items_without_product"], 1)
        self.assertTrue(q["quality_ok"])


if __name__ == "__main__":
    unittest.main()
