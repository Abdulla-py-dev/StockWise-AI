"""Automated tests for StockWise AI Phase 2 inventory management.

Tests use a temporary SQLite database so the real inventory database
(database/stockwise.db) is never modified.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta

from database.db import DEFAULT_DB_PATH, get_connection, init_database
from services.inventory_service import (
    HIGH_RISK_DAYS,
    MEDIUM_RISK_DAYS,
    add_product,
    classify_expiry,
    delete_product,
    get_categories,
    get_inventory_metrics,
    get_product,
    import_products_from_rows,
    list_products,
    update_product,
)
from utils.validators import ValidationError, parse_and_validate_csv


def _make_temp_db() -> str:
    """Create and initialize a temporary database; return its path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_database(path)
    return path


class DatabaseInitializationTests(unittest.TestCase):
    def setUp(self):
        self.db_path = _make_temp_db()

    def test_init_creates_inventory_table(self):
        conn = get_connection(self.db_path)
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='inventory'"
            ).fetchall()
            self.assertEqual(len(tables), 1)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(inventory)")}
            expected = {
                "id", "sku", "name", "category", "quantity", "unit_cost",
                "expiry_date", "created_at", "updated_at",
            }
            self.assertTrue(expected.issubset(columns))
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='inventory'"
            ).fetchall()
            self.assertGreaterEqual(len(indexes), 1)
            triggers = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
            self.assertGreaterEqual(len(triggers), 1)
        finally:
            conn.close()

    def test_schema_enforces_unique_sku(self):
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                "INSERT INTO inventory (sku, name, category, quantity, unit_cost, expiry_date) "
                "VALUES ('SKU1', 'A', 'Cat', 1, 1.0, '2026-12-01')"
            )
            conn.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO inventory (sku, name, category, quantity, unit_cost, expiry_date) "
                    "VALUES ('SKU1', 'B', 'Cat', 1, 1.0, '2026-12-01')"
                )
                conn.commit()
        finally:
            conn.close()

    def test_schema_enforces_non_negative_quantity_and_cost(self):
        conn = get_connection(self.db_path)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO inventory (sku, name, category, quantity, unit_cost, expiry_date) "
                    "VALUES ('SKU1', 'A', 'Cat', -1, 1.0, '2026-12-01')"
                )
                conn.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO inventory (sku, name, category, quantity, unit_cost, expiry_date) "
                    "VALUES ('SKU2', 'B', 'Cat', 1, -1.0, '2026-12-01')"
                )
                conn.commit()
        finally:
            conn.close()

    def test_does_not_touch_real_database(self):
        self.assertNotEqual(os.path.abspath(self.db_path), os.path.abspath(DEFAULT_DB_PATH))


class InventoryServiceTests(unittest.TestCase):
    def setUp(self):
        self.db_path = _make_temp_db()
        self.today = date.today()

    def _add(self, **kwargs):
        defaults = dict(
            sku="SKU001",
            name="Milk",
            category="Dairy",
            quantity=10,
            unit_cost=25.0,
            expiry_date=(self.today + timedelta(days=5)).isoformat(),
        )
        defaults.update(kwargs)
        return add_product(db_path=self.db_path, **defaults)

    def test_add_and_list_product(self):
        product = self._add()
        products = list_products(db_path=self.db_path)
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0].sku, "SKU001")
        self.assertEqual(products[0].name, "Milk")
        self.assertEqual(products[0].quantity, 10)

    def test_computed_expiry_fields(self):
        future = self.today + timedelta(days=3)
        product = self._add(expiry_date=future.isoformat())
        self.assertEqual(product.days_until_expiry, 3)
        self.assertEqual(product.expiry_status, "High Risk")

    def test_duplicate_sku_rejected(self):
        self._add(sku="DUP1")
        with self.assertRaises(ValidationError):
            add_product(
                sku="DUP1", name="Other", category="Cat", quantity=1,
                unit_cost=1.0,
                expiry_date=(self.today + timedelta(days=10)).isoformat(),
                db_path=self.db_path,
            )

    def test_invalid_quantity_rejected(self):
        with self.assertRaises(ValidationError):
            add_product(
                sku="Q1", name="A", category="Cat", quantity=-5,
                unit_cost=1.0,
                expiry_date=(self.today + timedelta(days=10)).isoformat(),
                db_path=self.db_path,
            )
        with self.assertRaises(ValidationError):
            add_product(
                sku="Q2", name="B", category="Cat", quantity="not-a-number",
                unit_cost=1.0,
                expiry_date=(self.today + timedelta(days=10)).isoformat(),
                db_path=self.db_path,
            )

    def test_invalid_unit_cost_rejected(self):
        with self.assertRaises(ValidationError):
            add_product(
                sku="C1", name="A", category="Cat", quantity=1,
                unit_cost=-3.0,
                expiry_date=(self.today + timedelta(days=10)).isoformat(),
                db_path=self.db_path,
            )

    def test_invalid_expiry_date_rejected(self):
        with self.assertRaises(ValidationError):
            add_product(
                sku="D1", name="A", category="Cat", quantity=1,
                unit_cost=1.0,
                expiry_date="not-a-date",
                db_path=self.db_path,
            )
        with self.assertRaises(ValidationError):
            add_product(
                sku="D2", name="B", category="Cat", quantity=1,
                unit_cost=1.0,
                expiry_date="",
                db_path=self.db_path,
            )

    def test_edit_product(self):
        product = self._add()
        updated = update_product(
            product.id, sku="SKU001", name="Toned Milk", category="Dairy",
            quantity=20, unit_cost=30.0,
            expiry_date=(self.today + timedelta(days=9)).isoformat(),
            db_path=self.db_path,
        )
        self.assertEqual(updated.name, "Toned Milk")
        self.assertEqual(updated.quantity, 20)
        self.assertEqual(updated.unit_cost, 30.0)

    def test_edit_duplicate_sku_rejected(self):
        self._add(sku="EDIT1")
        self._add(sku="EDIT2", name="Second")
        second = [p for p in list_products(db_path=self.db_path) if p.sku == "EDIT2"][0]
        with self.assertRaises(ValidationError):
            update_product(
                second.id, sku="EDIT1", name="Second", category="Cat",
                quantity=1, unit_cost=1.0,
                expiry_date=(self.today + timedelta(days=10)).isoformat(),
                db_path=self.db_path,
            )

    def test_edit_missing_product_rejected(self):
        with self.assertRaises(ValidationError):
            update_product(
                99999, sku="X", name="Y", category="Cat", quantity=1,
                unit_cost=1.0,
                expiry_date=(self.today + timedelta(days=10)).isoformat(),
                db_path=self.db_path,
            )

    def test_delete_product(self):
        product = self._add()
        self.assertTrue(delete_product(product.id, db_path=self.db_path))
        self.assertIsNone(get_product(product.id, db_path=self.db_path))
        self.assertEqual(len(list_products(db_path=self.db_path)), 0)
        self.assertFalse(delete_product(product.id, db_path=self.db_path))

    def test_classify_expiry_thresholds(self):
        self.assertEqual(classify_expiry(-1), "Expired")
        self.assertEqual(classify_expiry(0), "High Risk")
        self.assertEqual(classify_expiry(HIGH_RISK_DAYS), "High Risk")
        self.assertEqual(classify_expiry(HIGH_RISK_DAYS + 1), "Medium Risk")
        self.assertEqual(classify_expiry(MEDIUM_RISK_DAYS), "Medium Risk")
        self.assertEqual(classify_expiry(MEDIUM_RISK_DAYS + 1), "Low Risk")

    def test_search_and_filter(self):
        self._add(sku="A1", name="Apple", category="Fruit",
                  expiry_date=(self.today + timedelta(days=5)).isoformat())
        self._add(sku="B2", name="Banana", category="Fruit",
                  expiry_date=(self.today + timedelta(days=40)).isoformat())
        self._add(sku="C3", name="Carrot", category="Veg", quantity=20,
                  expiry_date=(self.today - timedelta(days=1)).isoformat())

        results = list_products(search="app", db_path=self.db_path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Apple")

        results = list_products(category="Fruit", db_path=self.db_path)
        self.assertEqual(len(results), 2)

        results = list_products(expiry_status="Expired", db_path=self.db_path)
        self.assertEqual(len(results), 1)

        results = list_products(sort_by="quantity", sort_order="desc", db_path=self.db_path)
        self.assertEqual(results[0].name, "Carrot")  # quantity 20

    def test_get_categories(self):
        self._add(category="Dairy")
        self._add(sku="X2", name="Bread", category="Bakery")
        self._add(sku="X3", name="Cheese", category="Dairy")
        categories = get_categories(db_path=self.db_path)
        self.assertEqual(categories, ["Bakery", "Dairy"])

    def test_metrics_empty_database(self):
        metrics = get_inventory_metrics(db_path=self.db_path)
        self.assertEqual(metrics["total_products"], 0)
        self.assertEqual(metrics["high_risk_products"], 0)
        self.assertEqual(metrics["expired_products"], 0)
        self.assertEqual(metrics["total_units"], 0)
        self.assertEqual(metrics["cost_exposure"], 0.0)

    def test_metrics_with_products(self):
        self._add(sku="M1", name="Milk", category="Dairy", quantity=10,
                  unit_cost=25.0, expiry_date=(self.today + timedelta(days=3)).isoformat())
        self._add(sku="E1", name="Expired Yogurt", category="Dairy", quantity=5,
                  unit_cost=10.0, expiry_date=(self.today - timedelta(days=2)).isoformat())
        metrics = get_inventory_metrics(db_path=self.db_path)
        self.assertEqual(metrics["total_products"], 2)
        self.assertEqual(metrics["total_units"], 15)
        self.assertEqual(metrics["expired_products"], 1)
        # High risk only (expiring within threshold, not expired): Milk at day 3
        self.assertEqual(metrics["high_risk_products"], 1)
        self.assertEqual(metrics["cost_exposure"], 250.0)


class CsvImportTests(unittest.TestCase):
    def setUp(self):
        self.db_path = _make_temp_db()
        self.today = date.today()
        self.header = "sku,name,category,quantity,unit_cost,expiry_date"

    def _csv(self, body: str) -> bytes:
        return (self.header + "\n" + body).encode("utf-8-sig")

    def test_valid_csv_parses(self):
        body = (
            f"A1,Milk,Dairy,10,25.0,{(self.today + timedelta(days=5)).isoformat()}\n"
            f"B2,Bread,Bakery,5,10.0,{(self.today + timedelta(days=12)).isoformat()}\n"
        )
        header, valid, invalid = parse_and_validate_csv(self._csv(body))
        self.assertEqual(header, ["sku", "name", "category", "quantity", "unit_cost", "expiry_date"])
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(invalid), 0)

    def test_missing_columns_rejected(self):
        # Header itself is missing required columns (do not prepend the standard header).
        content = b"sku,name\nA1,Milk\n"
        with self.assertRaises(ValidationError):
            parse_and_validate_csv(content)

    def test_invalid_rows_reported_row_level(self):
        body = (
            ",,Dairy,-1,abc,not-a-date\n"
            f"VALID1,Good,Dairy,1,1.0,{(self.today + timedelta(days=5)).isoformat()}\n"
        )
        header, valid, invalid = parse_and_validate_csv(self._csv(body))
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0].row_number, 2)
        self.assertGreater(len(invalid[0].errors), 0)

    def test_duplicate_sku_within_file(self):
        sku = "DUPFILE"
        future = (self.today + timedelta(days=5)).isoformat()
        body = (
            f"{sku},Milk,Dairy,10,25.0,{future}\n"
            f"{sku},Bread,Bakery,5,10.0,{future}\n"
        )
        header, valid, invalid = parse_and_validate_csv(self._csv(body))
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(invalid), 1)
        self.assertIn("Duplicate SKU", invalid[0].errors[0])

    def test_import_against_database(self):
        future = (self.today + timedelta(days=10)).isoformat()
        add_product(
            sku="EXIST", name="Existing", category="Dairy", quantity=1,
            unit_cost=1.0, expiry_date=future, db_path=self.db_path,
        )
        body = (
            f"EXIST,Duplicate,Dairy,1,1.0,{future}\n"
            f"NEW,New Item,Dairy,2,2.0,{future}\n"
        )
        header, valid, invalid = parse_and_validate_csv(self._csv(body))
        # Both rows are structurally valid; the EXIST duplicate is detected against
        # the database at import time, not during parsing.
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(invalid), 0)
        imported, rejected = import_products_from_rows(valid, db_path=self.db_path)
        self.assertEqual(imported, 1)
        self.assertEqual(len(rejected), 1)
        self.assertIn("Duplicate", rejected[0]["error"])
        self.assertEqual(len(list_products(db_path=self.db_path)), 2)

    def test_empty_database_dashboard_state(self):
        """Dashboard-relevant metrics on an empty database are zero, not sample data."""
        metrics = get_inventory_metrics(db_path=self.db_path)
        self.assertEqual(metrics["total_products"], 0)
        self.assertEqual(metrics["total_units"], 0)


if __name__ == "__main__":
    unittest.main()
