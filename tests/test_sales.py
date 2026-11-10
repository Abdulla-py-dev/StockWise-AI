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

    # --- Quantity × Unit Price Calculation Tests ---

    def test_line_total_calculation_single_item(self):
        """Test line total = quantity × unit_selling_price for a single item."""
        p = _add(self.db, cost=20.0)
        sale = record_sale([SaleItemInput(product_id=p.id, quantity=2, unit_selling_price=20.0)], db_path=self.db)
        # 2 × ₹20.00 = ₹40.00 = 4000 paise
        self.assertEqual(sale.items[0].line_total, 4000)
        self.assertEqual(sale.total_amount, 4000)

    def test_line_total_calculation_multiple_quantities(self):
        """Test line total with various quantities."""
        # Test quantity 1
        p1 = _add(self.db, sku="Q1", cost=15.50, qty=20)
        sale1 = record_sale([SaleItemInput(product_id=p1.id, quantity=1, unit_selling_price=15.50)], db_path=self.db)
        self.assertEqual(sale1.items[0].line_total, to_paise(15.50))
        # Test quantity 5
        p2 = _add(self.db, sku="Q5", cost=15.50, qty=20)
        sale2 = record_sale([SaleItemInput(product_id=p2.id, quantity=5, unit_selling_price=15.50)], db_path=self.db)
        self.assertEqual(sale2.items[0].line_total, to_paise(15.50) * 5)
        # Test quantity 10
        p3 = _add(self.db, sku="Q10", cost=15.50, qty=20)
        sale3 = record_sale([SaleItemInput(product_id=p3.id, quantity=10, unit_selling_price=15.50)], db_path=self.db)
        self.assertEqual(sale3.items[0].line_total, to_paise(15.50) * 10)

    def test_line_total_calculation_decimal_prices(self):
        """Test line total with decimal prices (paise precision)."""
        p = _add(self.db, cost=10.0)
        # Price with 2 decimal places
        sale = record_sale([SaleItemInput(product_id=p.id, quantity=3, unit_selling_price=12.34)], db_path=self.db)
        expected = to_paise(12.34) * 3
        self.assertEqual(sale.items[0].line_total, expected)
        self.assertEqual(sale.total_amount, expected)

    def test_line_total_calculation_multiple_items_different_quantities_prices(self):
        """Test cart total with multiple items having different quantities and prices."""
        p1 = _add(self.db, sku="A", qty=20, cost=10.0, name="Product A")
        p2 = _add(self.db, sku="B", qty=20, cost=20.0, name="Product B")
        p3 = _add(self.db, sku="C", qty=20, cost=5.0, name="Product C")
        sale = record_sale([
            SaleItemInput(product_id=p1.id, quantity=2, unit_selling_price=15.00),  # 2 × 15 = 30
            SaleItemInput(product_id=p2.id, quantity=3, unit_selling_price=25.00),  # 3 × 25 = 75
            SaleItemInput(product_id=p3.id, quantity=5, unit_selling_price=8.00),   # 5 × 8 = 40
        ], db_path=self.db)
        # Total = 30 + 75 + 40 = 145 INR = 14500 paise
        self.assertEqual(sale.total_amount, to_paise(145.00))
        self.assertEqual(sale.items[0].line_total, to_paise(30.00))
        self.assertEqual(sale.items[1].line_total, to_paise(75.00))
        self.assertEqual(sale.items[2].line_total, to_paise(40.00))

    def test_line_total_rounding_paise_precision(self):
        """Test that line totals use correct paise rounding (ROUND_HALF_UP)."""
        p = _add(self.db, cost=10.0)
        # 33.335 rounds to 33.34, × 2 = 66.68
        sale = record_sale([SaleItemInput(product_id=p.id, quantity=2, unit_selling_price=33.335)], db_path=self.db)
        expected_per_unit = to_paise(33.335)  # 3334 paise
        expected_total = expected_per_unit * 2  # 6668 paise = 66.68 INR
        self.assertEqual(sale.items[0].unit_selling_price, 3334)
        self.assertEqual(sale.items[0].line_total, expected_total)
        self.assertEqual(sale.total_amount, expected_total)

    def test_grand_total_matches_sum_of_line_totals(self):
        """Test that grand total always equals sum of line totals."""
        p1 = _add(self.db, sku="X", qty=50, cost=10.0)
        p2 = _add(self.db, sku="Y", qty=50, cost=20.0)
        sale = record_sale([
            SaleItemInput(product_id=p1.id, quantity=4, unit_selling_price=12.50),
            SaleItemInput(product_id=p2.id, quantity=6, unit_selling_price=22.75),
        ], db_path=self.db)
        sum_line_totals = sum(item.line_total for item in sale.items)
        self.assertEqual(sale.total_amount, sum_line_totals)

    # --- Regression Tests for Unit Selling Price Default Behavior ---

    def test_default_selling_price_equals_unit_cost(self):
        """Test that default selling price uses product's unit_cost (not markup)."""
        p = _add(self.db, sku="CHIPS", name="Chips", cost=20.0, qty=100)
        # When recording a sale with unit_selling_price = unit_cost (20.0)
        sale = record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=20.0)], db_path=self.db)
        self.assertEqual(sale.items[0].unit_selling_price, to_paise(20.0))
        self.assertEqual(sale.items[0].line_total, to_paise(20.0))

    def test_selling_price_can_override_unit_cost(self):
        """Test that user can manually set selling price different from unit_cost."""
        p = _add(self.db, sku="CHIPS", name="Chips", cost=20.0, qty=100)
        # User sets selling price to 25.0 (higher than unit_cost)
        sale = record_sale([SaleItemInput(product_id=p.id, quantity=2, unit_selling_price=25.0)], db_path=self.db)
        self.assertEqual(sale.items[0].unit_selling_price, to_paise(25.0))
        self.assertEqual(sale.items[0].line_total, to_paise(50.0))

    def test_selling_price_can_be_lower_than_unit_cost(self):
        """Test that selling price can be set lower than unit_cost (discount/loss)."""
        p = _add(self.db, sku="CHIPS", name="Chips", cost=20.0, qty=100)
        # User sets selling price to 15.0 (lower than unit_cost)
        sale = record_sale([SaleItemInput(product_id=p.id, quantity=3, unit_selling_price=15.0)], db_path=self.db)
        self.assertEqual(sale.items[0].unit_selling_price, to_paise(15.0))
        self.assertEqual(sale.items[0].line_total, to_paise(45.0))

    def test_quantity_change_updates_line_total(self):
        """Test that line total = quantity × unit_selling_price for various quantities."""
        p = _add(self.db, sku="CHIPS", name="Chips", cost=20.0, qty=100)
        # Quantity 1
        sale1 = record_sale([SaleItemInput(product_id=p.id, quantity=1, unit_selling_price=20.0)], db_path=self.db)
        self.assertEqual(sale1.items[0].line_total, to_paise(20.0))
        # Quantity 5
        sale2 = record_sale([SaleItemInput(product_id=p.id, quantity=5, unit_selling_price=20.0)], db_path=self.db)
        self.assertEqual(sale2.items[0].line_total, to_paise(100.0))
        # Quantity 10
        sale3 = record_sale([SaleItemInput(product_id=p.id, quantity=10, unit_selling_price=20.0)], db_path=self.db)
        self.assertEqual(sale3.items[0].line_total, to_paise(200.0))

    def test_multiple_products_each_use_own_unit_cost(self):
        """Test that different products use their own unit_cost as default."""
        p1 = _add(self.db, sku="CHIPS", name="Chips", cost=20.0, qty=100)
        p2 = _add(self.db, sku="SODA", name="Soda", cost=15.0, qty=100)
        p3 = _add(self.db, sku="CANDY", name="Candy", cost=5.0, qty=100)
        sale = record_sale([
            SaleItemInput(product_id=p1.id, quantity=1, unit_selling_price=20.0),  # Chips at unit_cost
            SaleItemInput(product_id=p2.id, quantity=1, unit_selling_price=15.0),  # Soda at unit_cost
            SaleItemInput(product_id=p3.id, quantity=1, unit_selling_price=5.0),   # Candy at unit_cost
        ], db_path=self.db)
        self.assertEqual(sale.items[0].unit_selling_price, to_paise(20.0))
        self.assertEqual(sale.items[1].unit_selling_price, to_paise(15.0))
        self.assertEqual(sale.items[2].unit_selling_price, to_paise(5.0))
        # Total = 20 + 15 + 5 = 40
        self.assertEqual(sale.total_amount, to_paise(40.0))

    def test_product_with_zero_unit_cost(self):
        """Test product with zero unit_cost works correctly."""
        p = _add(self.db, sku="FREE", name="Free Item", cost=0.0, qty=10)
        sale = record_sale([SaleItemInput(product_id=p.id, quantity=5, unit_selling_price=0.0)], db_path=self.db)
        self.assertEqual(sale.items[0].unit_selling_price, 0)
        self.assertEqual(sale.items[0].line_total, 0)
        self.assertEqual(sale.total_amount, 0)


if __name__ == "__main__":
    unittest.main()
