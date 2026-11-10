"""Comprehensive unit tests for the ML feature engineering service.

All tests use synthetic in-memory data. No test data is ever inserted into
the real SQLite database, and the real database is never modified.
"""

from __future__ import annotations

import unittest
from datetime import date, timedelta

import pandas as pd

from services.ml_features import (
    DEFAULT_CONFIG,
    FeatureConfig,
    build_feature_dataset,
    engineer_features,
    validate_feature_dataset,
)


def _make_timeseries(product_id, sku, name, start, units):
    """Build a build_timeseries-style list of dicts for consecutive days.

    ``units`` is a list aligned with ``start + i days``.
    """
    rows = []
    for i, u in enumerate(units):
        rows.append({
            "product_id": product_id,
            "sku": sku,
            "name": name,
            "date": (start + timedelta(days=i)).isoformat(),
            "units_sold": int(u),
        })
    return rows


class FeatureCalculationTests(unittest.TestCase):
    """Calendar and target features."""

    def test_calendar_features(self):
        ts = _make_timeseries(1, "SKU1", "Widget", date(2024, 1, 1), [5] * 3)
        # 2024-01-01 is a Monday.
        df = engineer_features(ts)
        self.assertEqual(len(df), 3)
        row = df.iloc[0]
        self.assertEqual(row["day_of_week"], 0)  # Monday
        self.assertEqual(row["day_of_month"], 1)
        self.assertEqual(row["month"], 1)
        self.assertEqual(row["is_weekend"], 0)

    def test_weekend_detection(self):
        # 2024-01-06 is a Saturday, 2024-01-07 is a Sunday.
        ts = _make_timeseries(1, "SKU1", "Widget", date(2024, 1, 6), [1, 1])
        df = engineer_features(ts)
        self.assertEqual(df.iloc[0]["day_of_week"], 5)
        self.assertEqual(df.iloc[0]["is_weekend"], 1)
        self.assertEqual(df.iloc[1]["day_of_week"], 6)
        self.assertEqual(df.iloc[1]["is_weekend"], 1)

    def test_units_sold_preserved(self):
        ts = _make_timeseries(1, "SKU1", "Widget", date(2024, 1, 1), [0, 3, 7])
        df = engineer_features(ts)
        self.assertEqual(list(df["units_sold"]), [0, 3, 7])

    def test_product_identifiers_preserved(self):
        ts = _make_timeseries(42, "SKU-42", "Gadget", date(2024, 1, 1), [1])
        df = engineer_features(ts)
        self.assertEqual(df.iloc[0]["product_id"], 42)
        self.assertEqual(df.iloc[0]["sku"], "SKU-42")
        self.assertEqual(df.iloc[0]["name"], "Gadget")

    def test_sorted_chronologically(self):
        # Input intentionally out of order; output must be chronological.
        ts = [
            {"product_id": 1, "sku": "S", "name": "N", "date": "2024-01-03", "units_sold": 3},
            {"product_id": 1, "sku": "S", "name": "N", "date": "2024-01-01", "units_sold": 1},
            {"product_id": 1, "sku": "S", "name": "N", "date": "2024-01-02", "units_sold": 2},
        ]
        df = engineer_features(ts)
        self.assertEqual(list(df["date"]), [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
        self.assertEqual(list(df["units_sold"]), [1, 2, 3])


class LagCalculationTests(unittest.TestCase):
    """Lag-1 and lag-7 features."""

    def test_lag_1(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [10, 20, 30, 40])
        df = engineer_features(ts)
        # First day has no previous day.
        self.assertTrue(pd.isna(df.iloc[0]["lag_1"]))
        self.assertEqual(df.iloc[1]["lag_1"], 10.0)
        self.assertEqual(df.iloc[2]["lag_1"], 20.0)
        self.assertEqual(df.iloc[3]["lag_1"], 30.0)

    def test_lag_7_requires_sufficient_history(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), list(range(1, 8)))  # 7 days
        df = engineer_features(ts)
        # lag_7 on day 8 would need day 1; with only 7 days all lag_7 are NaN.
        self.assertTrue(df["lag_7"].isna().all())

    def test_lag_7_available_after_eight_days(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), list(range(1, 11)))  # 10 days
        df = engineer_features(ts)
        # Day 8 (index 7) lag_7 = day 1's units = 1.
        self.assertEqual(df.iloc[7]["lag_7"], 1.0)
        # Day 10 (index 9) lag_7 = day 3's units = 3.
        self.assertEqual(df.iloc[9]["lag_7"], 3.0)
        # First 7 days have no lag_7.
        self.assertTrue(df["lag_7"].iloc[:7].isna().all())


class RollingAverageTests(unittest.TestCase):
    """7-day rolling average, past observations only."""

    def test_rolling_avg_uses_past_only(self):
        # units 1..10 over 10 consecutive days.
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), list(range(1, 11)))
        df = engineer_features(ts)
        # Day 8 (index 7): rolling avg over days 1..7 = mean(1..7) = 4.0.
        # Must NOT include day 8's own value (8).
        self.assertAlmostEqual(df.iloc[7]["rolling_avg_7"], 4.0)
        self.assertNotAlmostEqual(df.iloc[7]["rolling_avg_7"], 5.0)  # mean(2..8) would be leakage

    def test_rolling_avg_partial_window_at_start(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [10, 20, 30])
        df = engineer_features(ts)
        self.assertTrue(pd.isna(df.iloc[0]["rolling_avg_7"]))  # no past day
        self.assertAlmostEqual(df.iloc[1]["rolling_avg_7"], 10.0)  # mean([day1])
        self.assertAlmostEqual(df.iloc[2]["rolling_avg_7"], 15.0)  # mean([10, 20])

    def test_rolling_avg_full_window(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), list(range(1, 10)))  # 9 days
        df = engineer_features(ts)
        # Day 9 (index 8): mean(days 2..8) = mean(2..8) = 5.0.
        self.assertAlmostEqual(df.iloc[8]["rolling_avg_7"], 5.0)


class RollingStdTests(unittest.TestCase):
    """7-day rolling standard deviation, past observations only."""

    def test_rolling_std_uses_past_only(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), list(range(1, 11)))
        df = engineer_features(ts)
        # Day 8 (index 7): std over days 1..7.
        expected = pd.Series(range(1, 8), dtype=float).std(ddof=1)
        self.assertAlmostEqual(df.iloc[7]["rolling_std_7"], expected)

    def test_rolling_std_na_with_one_past_observation(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [10, 20])
        df = engineer_features(ts)
        # Day 2 (index 1): only one past observation -> sample std undefined -> NaN.
        self.assertTrue(pd.isna(df.iloc[1]["rolling_std_7"]))

    def test_rolling_std_available_with_two_plus_past(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [10, 20, 30])
        df = engineer_features(ts)
        # Day 3 (index 2): std over [10, 20].
        self.assertAlmostEqual(df.iloc[2]["rolling_std_7"], pd.Series([10.0, 20.0]).std(ddof=1))


class NoDataLeakageTests(unittest.TestCase):
    """Future values must never influence historical features."""

    def test_current_day_excluded_from_own_features(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), list(range(1, 11)))
        df = engineer_features(ts)
        # For every row i >= 1, rolling_avg_7 must equal mean of units[i-7:i]
        # (prior days only), never mean of units[i-6:i+1] (which includes current).
        units = list(range(1, 11))
        for i in range(1, len(units)):
            start = max(0, i - 7)
            expected = sum(units[start:i]) / max(1, i - start)
            actual = df.iloc[i]["rolling_avg_7"]
            if pd.isna(actual):
                continue
            self.assertAlmostEqual(actual, expected)
            # Sanity: it must not be the mean including the current day.
            leaky = sum(units[max(0, i - 6):i + 1]) / (i - max(0, i - 6) + 1)
            self.assertNotAlmostEqual(actual, leaky)

    def test_lag_never_sees_current_or_future(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [100, 1, 1])
        df = engineer_features(ts)
        # Day 2's lag_1 = day 1's units (100), not day 2's own (1).
        self.assertEqual(df.iloc[1]["lag_1"], 100.0)
        # Day 3's lag_1 = day 2's units (1).
        self.assertEqual(df.iloc[2]["lag_1"], 1.0)

    def _assert_feature_equal(self, a, b, msg):
        """Compare two feature values, treating two NaNs as equivalent."""
        if pd.isna(a) and pd.isna(b):
            return
        self.assertEqual(a, b, msg)

    def test_future_change_does_not_alter_past_features(self):
        ts_base = _make_timeseries(1, "S", "N", date(2024, 1, 1), [5, 5, 5, 5])
        ts_future_changed = _make_timeseries(1, "S", "N", date(2024, 1, 1), [5, 5, 5, 999])
        df_base = engineer_features(ts_base)
        df_changed = engineer_features(ts_future_changed)
        # Features for days 1..3 must be identical regardless of day 4's value.
        for i in range(3):
            for col in ["lag_1", "lag_7", "rolling_avg_7", "rolling_std_7"]:
                self._assert_feature_equal(
                    df_base.iloc[i][col], df_changed.iloc[i][col],
                    f"Leakage: {col} on day {i+1} changed when day 4 changed.",
                )


class MultipleProductsTests(unittest.TestCase):
    """Features are product-specific; no cross-product contamination."""

    def test_products_computed_independently(self):
        ts = (
            _make_timeseries(1, "A", "Alpha", date(2024, 1, 1), [10, 10, 10])
            + _make_timeseries(2, "B", "Beta", date(2024, 1, 1), [1, 1, 1])
        )
        df = engineer_features(ts)
        a = df[df["product_id"] == 1].sort_values("date").reset_index(drop=True)
        b = df[df["product_id"] == 2].sort_values("date").reset_index(drop=True)
        self.assertEqual(len(a), 3)
        self.assertEqual(len(b), 3)
        # Product A's lag_1 on day 2 = 10; product B's lag_1 on day 2 = 1.
        self.assertEqual(a.iloc[1]["lag_1"], 10.0)
        self.assertEqual(b.iloc[1]["lag_1"], 1.0)
        # No cross-contamination.
        self.assertFalse((a["lag_1"] == b["lag_1"]).any())

    def test_interleaved_dates_do_not_leak_across_products(self):
        ts = (
            _make_timeseries(1, "A", "Alpha", date(2024, 1, 1), [100, 100])
            + _make_timeseries(2, "B", "Beta", date(2024, 1, 1), [1, 1])
        )
        df = engineer_features(ts)
        rows = df.sort_values(["product_id", "date"]).reset_index(drop=True)
        # Each product's day-2 lag_1 equals its own day-1 value.
        a_rows = rows[rows["product_id"] == 1]
        b_rows = rows[rows["product_id"] == 2]
        self.assertEqual(a_rows.iloc[1]["lag_1"], 100.0)
        self.assertEqual(b_rows.iloc[1]["lag_1"], 1.0)


class MissingDatesTests(unittest.TestCase):
    """Gaps within a product's range are zero-filled (not fabricated)."""

    def test_missing_days_zero_filled_via_build_timeseries_path(self):
        # Simulate build_timeseries output where a middle day is absent and
        # then re-added as zero (as build_timeseries does for contiguous
        # ranges). engineer_features itself preserves whatever rows are given.
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [5, 0, 5])
        df = engineer_features(ts)
        self.assertEqual(list(df["units_sold"]), [5, 0, 5])
        self.assertEqual(len(df), 3)

    def test_zero_sales_produce_valid_features(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [0, 0, 0])
        df = engineer_features(ts)
        self.assertTrue((df["units_sold"] == 0).all())
        self.assertTrue(df["lag_1"].iloc[1:].eq(0.0).all())


class EmptyDatasetTests(unittest.TestCase):
    """Empty inputs are handled gracefully."""

    def test_empty_timeseries(self):
        df = engineer_features([])
        self.assertTrue(df.empty)

    def test_empty_validation_report(self):
        report = validate_feature_dataset(pd.DataFrame(columns=["product_id", "date", "units_sold"]))
        self.assertTrue(report["quality_ok"])
        self.assertEqual(report["total_rows"], 0)
        self.assertEqual(report["products"], 0)

    def test_build_feature_dataset_empty_database(self):
        # Uses a temporary database, never the real one.
        import tempfile
        import os
        from database.db import init_database

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        init_database(path)
        try:
            df = build_feature_dataset(db_path=path)
            self.assertTrue(df.empty)
        finally:
            os.remove(path)


class InsufficientHistoryTests(unittest.TestCase):
    """Products with too little history for lag/rolling features."""

    def test_single_day_history(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [5])
        df = engineer_features(ts)
        self.assertTrue(pd.isna(df.iloc[0]["lag_1"]))
        self.assertTrue(pd.isna(df.iloc[0]["lag_7"]))
        self.assertTrue(pd.isna(df.iloc[0]["rolling_avg_7"]))
        self.assertTrue(pd.isna(df.iloc[0]["rolling_std_7"]))

    def test_lag_7_unavailable_for_short_series(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [1, 2, 3])
        df = engineer_features(ts)
        self.assertTrue(df["lag_7"].isna().all())

    def test_validation_reports_insufficient_history(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [1, 2, 3])
        df = engineer_features(ts)
        report = validate_feature_dataset(df)
        self.assertGreaterEqual(
            len(report["insufficient_history"]), 1,
            "short history should be flagged",
        )
        self.assertTrue(any(
            "lag_7" in " ".join(x["reasons"]) for x in report["insufficient_history"]
        ))


class InvalidDataTests(unittest.TestCase):
    """Negative sales, duplicates, and invalid values are caught."""

    def test_negative_sales_flagged(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [5, -1, 3])
        df = engineer_features(ts)
        report = validate_feature_dataset(df)
        self.assertFalse(report["quality_ok"])
        self.assertGreaterEqual(len(report["negative_sales"]), 1)
        self.assertTrue(any("Negative sales quantity" in x["explanation"] for x in report["negative_sales"]))

    def test_duplicate_product_date_flagged(self):
        ts = [
            {"product_id": 1, "sku": "S", "name": "N", "date": "2024-01-01", "units_sold": 5},
            {"product_id": 1, "sku": "S", "name": "N", "date": "2024-01-01", "units_sold": 3},
        ]
        df = engineer_features(ts)
        report = validate_feature_dataset(df)
        self.assertFalse(report["quality_ok"])
        self.assertGreaterEqual(len(report["duplicate_product_dates"]), 1)
        self.assertTrue(any("duplicate product-date" in x["explanation"].lower() for x in report["duplicate_product_dates"]))

    def test_valid_data_passes_quality(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [1, 2, 3])
        df = engineer_features(ts)
        report = validate_feature_dataset(df)
        self.assertTrue(report["quality_ok"])

    def test_validation_provides_clear_explanations(self):
        ts = _make_timeseries(1, "S", "N", date(2024, 1, 1), [5, -1, 3])
        df = engineer_features(ts)
        report = validate_feature_dataset(df)
        # Every summary message is a plain string; at least one error present.
        self.assertTrue(any("ERROR" in m or "FAILED" in m for m in report["summary"]))
        for issue in report["negative_sales"]:
            self.assertIn("explanation", issue)
            self.assertTrue(len(issue["explanation"]) > 0)


class EndToEndTests(unittest.TestCase):
    """End-to-end path through build_feature_dataset using a temp DB."""

    def test_build_feature_dataset_from_temp_db(self):
        import tempfile
        import os
        from database.db import init_database
        from services.inventory_service import add_product
        from services.sales_service import SaleItemInput, record_sale

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        init_database(path)
        try:
            today = date.today()
            p1 = add_product(
                sku="P1", name="Prod One", category="C", quantity=50,
                unit_cost=10.0,
                expiry_date=(today + timedelta(days=60)).isoformat(),
                db_path=path,
            )
            p2 = add_product(
                sku="P2", name="Prod Two", category="C", quantity=50,
                unit_cost=10.0,
                expiry_date=(today + timedelta(days=60)).isoformat(),
                db_path=path,
            )
            base = date.today()
            for i in range(10):
                d = (base + timedelta(days=i)).isoformat() + "T00:00:00"
                record_sale(
                    [SaleItemInput(product_id=p1.id, quantity=1, unit_selling_price=10.0)],
                    db_path=path, transaction_date=d,
                    transaction_uuid=f"uuid-p1-{i}",
                )
            for i in range(10):
                d = (base + timedelta(days=i)).isoformat() + "T00:00:00"
                record_sale(
                    [SaleItemInput(product_id=p2.id, quantity=2, unit_selling_price=10.0)],
                    db_path=path, transaction_date=d,
                    transaction_uuid=f"uuid-p2-{i}",
                )
            df = build_feature_dataset(db_path=path)
            self.assertFalse(df.empty)
            self.assertEqual(df["product_id"].nunique(), 2)
            self.assertIn("lag_1", df.columns)
            self.assertIn("lag_7", df.columns)
            self.assertIn("rolling_avg_7", df.columns)
            self.assertIn("rolling_std_7", df.columns)
            # All units_sold are genuine integers >= 0.
            self.assertTrue((df["units_sold"] >= 0).all())
            self.assertTrue((df["units_sold"] == df["units_sold"].astype(int)).all())
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
