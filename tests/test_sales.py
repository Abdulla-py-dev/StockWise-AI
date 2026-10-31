"""Tests for Phase 5 Step 3 sales tracking."""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from datetime import date, timedelta

from database.db import init_database, get_connection
from services.inventory_service import add_product
from services.sales_service import (
    SaleItemInput,
    from_paise,
    get_sale,
    get_sales_metrics,
    list_sales,
    record_sale,
    to_paise,
)


def _temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_database(path)
    return path


def _add(db_path, sku="S1", name="Milk", qty=10, cost=25.0, days=30):
    today = date.today()
    return add_product(
        sku=sku, name=name, category="Dairy", quantity=qty,
        unit_cost=cost, expiry_date=(today + timedelta(days=days)).isoformat(),
        db_path=db_path,
    )


class SalesServiceTests(unittest.TestCase):
    def setUp(self):
        self.db = _temp_db()

    def test_single_product_sale(self):
        p = _add(self.db)
        sale = record_sale([SaleItemInput(product_id=p.id, quantity=2, unit_selling_price=30.0)], db_path=self.db)
        self.assertEqual(len(sale.items), 1)
        self.assertEqual(sale.total_amount, to_paise(30.0) * 2)

    def test_stock_reduction(self):
        p = _add(self.db, qty=10)
        record_sale([SaleItemInput(product_id=p.id, quantity=3, unit_selling_price=30.0)], db_path=self.db)
        from services.inventory_service import get_product
        pp = get_product(p.id, db_path=self.db)
        self.assertEqual(pp.quantity, 7)

    def test_multiple_product_sale(self):
        p1 = _add(self.db, sku="A", qty=5)
        p2 = _add(self.db, sku="B", qty=5, name="Bread")
        sale = record_sale([
            SaleItemInput(product_id=p1.id, quantity=1, unit_selling_price=10.0),
            SaleItemInput(product_id=p2.id, quantity=2, unit_selling_price=20.0),
        ], db_path=self.db)
        self.assertEqual(len(sale.items), 2)
        self.assertEqual(sale.total_amount, to_paise(10.0) + to_paise(20.0) * 2)

    def test_insufficient_stock(self):
        p = _add(self.db, qty=2)
        with self.assertRaises(Exception):
            record_sale([SaleItemInput(product_id=p.id, quantity=5, unit_selling_price=10.0)], db_path=self.db)
        from services.inventory_service import get_product
        pp = get_product(p.id, db_path=self.db)
        self.assertEqual(pp.quantity, 2)  # unchanged

    def test_expired_product_rejected(self):
        p = _add(self.db, days=-1)
        with self.assertRaises(Exception):
            record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)], db_path=self.db)

    def test_empty_cart(self):
        with self.assertRaises(Exception):
            record_sale([], db_path=self.db)

    def test_duplicate_uuid(self):
        p = _add(self.db)
        u = str(uuid.uuid4())
        record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)], db_path=self.db, transaction_uuid=u)
        with self.assertRaises(Exception):
            record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)], db_path=self.db, transaction_uuid=u)

    def test_atomic_rollback(self):
        p = _add(self.db, qty=5)
        # second item insufficient -> whole sale rolls back
        p2 = _add(self.db, sku="B2", qty=1)
        try:
            record_sale([
                SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0),
                SaleItemInput(product_id=p2.id, quantity=5, unit_selling_price=10.0),
            ], db_path=self.db)
        except Exception:
            pass
        from services.inventory_service import get_product
        self.assertEqual(get_product(p.id, db_path=self.db).quantity, 5)
        self.assertEqual(get_product(p2.id, db_path=self.db).quantity, 1)
        self.assertEqual(list_sales(db_path=self.db), [])

    def test_product_deletion_preserves_history(self):
        p = _add(self.db)
        sale = record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=10.0)], db_path=self.db)
        from services.inventory_service import delete_product
        delete_product(p.id, db_path=self.db)
        s = get_sale(sale.id, db_path=self.db)
        self.assertIsNotNone(s)
        self.assertEqual(s.items[0].product_id, None)
        self.assertEqual(s.items[0].product_sku_snapshot, p.sku)
        self.assertEqual(s.items[0].product_name_snapshot, p.name)

    def test_historical_price_snapshot(self):
        p = _add(self.db, cost=25.0)
        sale = record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=40.0)], db_path=self.db)
        # change product cost
        from services.inventory_service import update_product
        from datetime import date
        update_product(p.id, sku=p.sku, name=p.name, category=p.category, quantity=p.quantity - 1,
                       unit_cost=99.0, expiry_date=p.expiry_date, db_path=self.db)
        s = get_sale(sale.id, db_path=self.db)
        self.assertEqual(s.items[0].unit_cost_at_sale, to_paise(25.0))
        self.assertEqual(s.items[0].unit_selling_price, to_paise(40.0))

    def test_paise_conversion(self):
        self.assertEqual(to_paise(12.345), 1235)  # rounds
        self.assertEqual(from_paise(1234), 12.34)
        self.assertEqual(to_paise("5"), 500)

    def test_zero_price_product(self):
        p = _add(self.db, cost=0.0)
        sale = record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=0.0)], db_path=self.db)
        self.assertEqual(sale.total_amount, 0)

    def test_fk_enforcement(self):
        conn = get_connection(self.db)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            with self.assertRaises(Exception):
                conn.execute("INSERT INTO sale_items (sale_id, product_id, quantity_sold, unit_selling_price, unit_cost_at_sale, line_total) VALUES (99999, 1, 1, 100, 100, 100)")
                conn.commit()
        finally:
            conn.close()

    def test_sales_metrics(self):
        p = _add(self.db)
        record_sale([SaleItemInput(product_id=p.id, quantity=2, unit_selling_price=30.0)], db_path=self.db)
        m = get_sales_metrics(db_path=self.db)
        self.assertEqual(m["total_sales"], 1)
        self.assertEqual(m["total_items_sold"], 2)
        self.assertEqual(m["total_revenue_inr"], from_paise(to_paise(30.0) * 2))


if __name__ == "__main__":
    unittest.main()
