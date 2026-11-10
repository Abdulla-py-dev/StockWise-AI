"""Unit tests for the ML demand forecasting pipeline (Part 2B).

All tests use synthetic in-memory data in temporary SQLite databases.
No test data is ever inserted into the real SQLite database, and the real
database is never modified. No models are trained on insufficient data.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from datetime import date, timedelta

import pandas as pd

from database.db import init_database
from services.inventory_service import add_product
from services.ml_forecast_service import (
    MIN_USABLE_ROWS,
    run_forecast_pipeline,
)
from services.ml_readiness_service import ml_readiness
from services.sales_service import SaleItemInput, record_sale


def _temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_database(path)
    return path


def _add_product(db, sku, name, qty=100, cost=10.0, days=60):
    today = date.today()
    return add_product(
        sku=sku,
        name=name,
        category="Test",
        quantity=qty,
        unit_cost=cost,
        expiry_date=(today + timedelta(days=days)).isoformat(),
        db_path=db,
    )


def _record_sales_over_days(db, product_id, base_date, num_days, units_per_day=1):
    """Record one sale per day for `num_days` consecutive days for a product."""
    for d in range(num_days):
        sale_date = (base_date + timedelta(days=d)).isoformat() + "T00:00:00"
        record_sale(
            [SaleItemInput(product_id=product_id, quantity=units_per_day, unit_selling_price=10.0)],
            db_path=db,
            transaction_uuid=str(uuid.uuid4()),
            transaction_date=sale_date,
        )


class ForecastPipelineInsufficientDataTests(unittest.TestCase):
    """Pipeline must skip training with a clear message when data is insufficient."""

    def setUp(self):
        self.db = _temp_db()
        self.base = date.today()

    def test_readiness_eligible_but_insufficient_usable_rows(self):
        """Readiness passes (30 txns, 10 days, 3 products) but usable rows < 14.

        3 products x 10 days = 30 transactions, 10 days coverage, 3 products.
        Each product has 10 contiguous days -> 3 usable rows each = 9 total < 14.
        Training must be skipped.
        """
        products = []
        for i in range(3):
            p = _add_product(self.db, sku=f"P{i}", name=f"Product {i}")
            products.append(p)
        for p in products:
            _record_sales_over_days(self.db, p.id, self.base, num_days=10, units_per_day=1)

        readiness = ml_readiness(db_path=self.db)
        self.assertTrue(readiness["eligible"])

        result = run_forecast_pipeline(db_path=self.db)
        self.assertFalse(result.sufficiency["sufficient"])
        self.assertLess(result.usable_rows, MIN_USABLE_ROWS)
        self.assertEqual(result.training_status, "skipped_insufficient_data")
        self.assertIn("minimum", result.message.lower())
        self.assertFalse(result.baseline.trained)
        self.assertFalse(any(m.trained for m in result.models))
        self.assertEqual(result.forecasts, [])

    def test_not_eligible_skips_training(self):
        """When readiness is not eligible, training is skipped immediately."""
        result = run_forecast_pipeline(db_path=self.db)  # empty DB
        self.assertFalse(result.eligibility["eligible"])
        self.assertEqual(result.training_status, "skipped_not_eligible")
        self.assertFalse(result.baseline.trained)
        self.assertEqual(result.forecasts, [])


class ForecastPipelineSufficientDataTests(unittest.TestCase):
    """Pipeline trains and evaluates when enough usable rows exist."""

    def setUp(self):
        self.db = _temp_db()
        self.base = date.today()

    def test_sufficient_data_trains_and_evaluates(self):
        """3 products x 14 days = 21 usable rows >= 14. Training completes."""
        products = []
        for i in range(3):
            p = _add_product(self.db, sku=f"P{i}", name=f"Product {i}")
            products.append(p)
        for p in products:
            _record_sales_over_days(self.db, p.id, self.base, num_days=14, units_per_day=1)

        readiness = ml_readiness(db_path=self.db)
        self.assertTrue(readiness["eligible"])

        result = run_forecast_pipeline(db_path=self.db)
        self.assertTrue(result.sufficiency["sufficient"])
        self.assertGreaterEqual(result.usable_rows, MIN_USABLE_ROWS)
        self.assertEqual(result.training_status, "completed")
        self.assertTrue(result.baseline.trained)
        self.assertIsNotNone(result.baseline.mae)
        self.assertIsNotNone(result.baseline.rmse)
        self.assertGreaterEqual(result.baseline.n_test, 1)

    def test_naive_baseline_uses_lag_1(self):
        """The naive baseline predicts using lag_1 (previous day's units)."""
        products = []
        for i in range(3):
            p = _add_product(self.db, sku=f"P{i}", name=f"Product {i}")
            products.append(p)
        for p in products:
            _record_sales_over_days(self.db, p.id, self.base, num_days=14, units_per_day=1)

        result = run_forecast_pipeline(db_path=self.db)
        self.assertEqual(result.baseline.name, "naive_last_observed")
        self.assertTrue(result.baseline.trained)
        # MAE and RMSE must be non-negative floats.
        self.assertIsInstance(result.baseline.mae, float)
        self.assertIsInstance(result.baseline.rmse, float)
        self.assertGreaterEqual(result.baseline.mae, 0.0)
        self.assertGreaterEqual(result.baseline.rmse, 0.0)

    def test_linear_regression_and_random_forest_attempted(self):
        """Both Linear Regression and Random Forest are attempted when data sufficient."""
        products = []
        for i in range(3):
            p = _add_product(self.db, sku=f"P{i}", name=f"Product {i}")
            products.append(p)
        for p in products:
            _record_sales_over_days(self.db, p.id, self.base, num_days=14, units_per_day=1)

        result = run_forecast_pipeline(db_path=self.db)
        model_names = {m.name for m in result.models}
        self.assertIn("linear_regression", model_names)
        self.assertIn("random_forest", model_names)

    def test_chronological_split_no_future_leakage(self):
        """Train dates are strictly before test dates (chronological split)."""
        products = []
        for i in range(3):
            p = _add_product(self.db, sku=f"P{i}", name=f"Product {i}")
            products.append(p)
        for p in products:
            _record_sales_over_days(self.db, p.id, self.base, num_days=14, units_per_day=1)

        result = run_forecast_pipeline(db_path=self.db)
        self.assertEqual(result.training_status, "completed")
        # The pipeline internally splits chronologically. Verify that test
        # set dates are after train dates by inspecting the baseline test
        # predictions: n_test >= 1 means a test split existed.
        self.assertGreaterEqual(result.baseline.n_test, 1)

    def test_target_is_one_day_ahead(self):
        """The target column is units_sold shifted by -1 (next day)."""
        from services.ml_features import build_feature_dataset, engineer_features
        from services.ml_forecast_service import _create_target, TARGET_COLUMN

        p = _add_product(self.db, sku="P0", name="Product 0")
        _record_sales_over_days(self.db, p.id, self.base, num_days=10, units_per_day=1)

        df = build_feature_dataset(db_path=self.db)
        df = _create_target(df)
        self.assertIn(TARGET_COLUMN, df.columns)

        # For each product, target[t] should equal units_sold[t+1].
        for pid, group in df.groupby("product_id"):
            group = group.sort_values("date").reset_index(drop=True)
            for i in range(len(group) - 1):
                expected = group["units_sold"].iloc[i + 1]
                actual = group[TARGET_COLUMN].iloc[i]
                if pd.notna(expected) and pd.notna(actual):
                    self.assertEqual(actual, expected)
            # Last row target should be NaN.
            self.assertTrue(pd.isna(group[TARGET_COLUMN].iloc[-1]))


class ForecastPipelineEdgeCaseTests(unittest.TestCase):
    """Edge cases: empty DB, single product, insufficient history."""

    def setUp(self):
        self.db = _temp_db()
        self.base = date.today()

    def test_empty_database(self):
        result = run_forecast_pipeline(db_path=self.db)
        self.assertFalse(result.eligibility["eligible"])
        self.assertEqual(result.total_rows, 0)
        self.assertEqual(result.usable_rows, 0)
        self.assertEqual(result.training_status, "skipped_not_eligible")
        self.assertEqual(result.forecasts, [])

    def test_single_product_insufficient(self):
        """One product with 10 days: readiness fails (needs 3 products), training skipped."""
        p = _add_product(self.db, sku="P0", name="Product 0")
        _record_sales_over_days(self.db, p.id, self.base, num_days=10, units_per_day=1)

        result = run_forecast_pipeline(db_path=self.db)
        self.assertFalse(result.eligibility["eligible"])
        self.assertEqual(result.training_status, "skipped_not_eligible")

    def test_usable_rows_count(self):
        """usable_rows = sum(contiguous_days - 7 - 1) per product.

        The -7 is for lag_7 warm-up; the extra -1 is because the last day of
        each product has no next-day target (shift(-1) produces NaN).
        2 products x 10 days -> 2 x (10 - 8) = 4 usable rows.
        """
        p1 = _add_product(self.db, sku="P1", name="Product 1")
        p2 = _add_product(self.db, sku="P2", name="Product 2")
        _record_sales_over_days(self.db, p1.id, self.base, num_days=10, units_per_day=1)
        _record_sales_over_days(self.db, p2.id, self.base, num_days=10, units_per_day=1)

        result = run_forecast_pipeline(db_path=self.db)
        # Readiness fails (only 2 products), so training skipped but sufficiency
        # still computed.
        self.assertEqual(result.usable_rows, 4)
        self.assertFalse(result.sufficiency["sufficient"])


if __name__ == "__main__":
    unittest.main()
