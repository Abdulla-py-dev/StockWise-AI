"""ML feature engineering service for StockWise AI demand forecasting.

Part 1 of the ML pipeline: data preparation and feature engineering only.
No model training or deployment is performed here.

All features are derived exclusively from genuine historical sales records
(``sale_items`` joined to ``sales`` via :func:`services.ml_readiness_service.build_timeseries`).
No sales data is fabricated, no future observations are used to compute
historical features (no data leakage), and features are computed
product-specifically.

Public API
----------
* :func:`build_feature_dataset` - end-to-end: real sales -> feature DataFrame
* :func:`engineer_features` - compute features from a time-series list of dicts
* :func:`validate_feature_dataset` - validate a feature DataFrame and report issues
* :class:`FeatureConfig` - knobs for lag/rolling windows
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from services.ml_readiness_service import build_timeseries


# ---- Documented readiness criteria (kept consistent with ml_readiness_service) ----

MIN_TRANSACTIONS = 30
MIN_DAYS_COVERAGE = 10
MIN_PRODUCTS_WITH_SALES = 3

# Default feature-engineering knobs.
LAG_1_DAYS = 1
LAG_7_DAYS = 7
ROLLING_WINDOW_DAYS = 7


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature engineering.

    All windows are expressed in days. Rolling statistics are computed on the
    *lagged* series so that the current day's sales are never part of their
    own features (no data leakage).
    """

    lag_1: int = LAG_1_DAYS
    lag_7: int = LAG_7_DAYS
    rolling_window: int = ROLLING_WINDOW_DAYS
    min_rolling_obs_mean: int = 1
    min_rolling_obs_std: int = 2


DEFAULT_CONFIG = FeatureConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dataframe(timeseries: list[dict]) -> pd.DataFrame:
    """Convert a build_timeseries-style list of dicts to a sorted DataFrame.

    Rows with a null ``product_id`` (historical snapshots for deleted products)
    are excluded: they carry no stable product identifier and are not usable
    for demand forecasting of current stock.
    """
    columns = ["product_id", "sku", "name", "date", "units_sold"]
    if not timeseries:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(timeseries)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"timeseries entries missing required keys: {missing}")

    df = df[df["product_id"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.date
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(
    timeseries: list[dict],
    config: FeatureConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Engineer ML features from a genuine sales time series.

    Parameters
    ----------
    timeseries : list of dict
        Output of :func:`services.ml_readiness_service.build_timeseries`.
        Each entry: ``product_id``, ``sku``, ``name``, ``date`` (ISO string),
        ``units_sold`` (int).
    config : FeatureConfig, optional
        Lag/rolling window configuration.

    Returns
    -------
    pd.DataFrame
        One row per product per calendar day (contiguous, missing days filled
        with 0 units by ``build_timeseries``). Columns:

        * ``product_id`` - product identifier (int)
        * ``sku`` - product SKU snapshot (str or None)
        * ``name`` - product name snapshot (str or None)
        * ``date`` - calendar date (datetime.date)
        * ``units_sold`` - genuine units sold that day (int, target)
        * ``day_of_week`` - 0=Monday ... 6=Sunday (int)
        * ``day_of_month`` - 1..31 (int)
        * ``month`` - 1..12 (int)
        * ``is_weekend`` - 1 if Saturday/Sunday else 0 (int)
        * ``lag_1`` - units sold the previous day (float or NaN)
        * ``lag_7`` - units sold 7 days ago (float or NaN; only when
          sufficient history exists)
        * ``rolling_avg_7`` - mean of units sold over the previous 7 days
          (float or NaN; past observations only)
        * ``rolling_std_7`` - sample std dev of units sold over the previous
          7 days (float or NaN; past observations only)

    Data leakage prevention
    -----------------------
    * Observations are sorted chronologically within each product.
    * Lag features use ``shift`` so only *prior* days' sales are used.
    * Rolling statistics are computed on the lagged series, so the current
      day's sales are never included in their own features.
    * All computations are product-specific (grouped by ``product_id``).
    * No future sales are ever used.

    Missing values
    --------------
    Unavailable lag/rolling values (early days of a product's history) are
    left as NaN. They are never imputed or fabricated.
    """
    df = _to_dataframe(timeseries)
    if df.empty:
        return df

    frames: list[pd.DataFrame] = []
    for _pid, group in df.groupby("product_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True)

        # Calendar features: derived solely from the date (no leakage).
        group["day_of_week"] = group["date"].apply(lambda d: d.weekday())  # Mon=0..Sun=6
        group["day_of_month"] = group["date"].apply(lambda d: d.day)
        group["month"] = group["date"].apply(lambda d: d.month)
        group["is_weekend"] = group["day_of_week"].isin([5, 6]).astype(int)

        # Target: genuine sales, never fabricated.
        units = group["units_sold"].astype(float)

        # Lag features: strictly past observations.
        group["lag_1"] = units.shift(config.lag_1)
        group["lag_7"] = units.shift(config.lag_7)

        # Rolling statistics on past observations only (exclude current day).
        lagged = units.shift(1)
        group["rolling_avg_7"] = lagged.rolling(
            window=config.rolling_window,
            min_periods=config.min_rolling_obs_mean,
        ).mean()
        group["rolling_std_7"] = lagged.rolling(
            window=config.rolling_window,
            min_periods=config.min_rolling_obs_std,
        ).std()

        frames.append(group)

    if not frames:
        return df
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["product_id", "date"]).reset_index(drop=True)
    return result


def build_feature_dataset(
    db_path: Optional[str] = None,
    product_id: Optional[int] = None,
    config: FeatureConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Build the ML feature dataset from genuine database sales.

    Wraps :func:`services.ml_readiness_service.build_timeseries` and applies
    :func:`engineer_features`. Read-only: the database is never modified.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite database. Defaults to the project database.
    product_id : int, optional
        When given, only features for that product are produced.
    config : FeatureConfig, optional

    Returns
    -------
    pd.DataFrame
        See :func:`engineer_features` for columns and semantics.
    """
    timeseries = build_timeseries(db_path=db_path, product_id=product_id)
    return engineer_features(timeseries, config=config)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _as_product_id(value) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_duplicate_product_dates(df: pd.DataFrame) -> list[dict]:
    """Report product/date combinations with more than one sales record."""
    issues: list[dict] = []
    if df.empty:
        return issues
    dup = df[df.duplicated(subset=["product_id", "date"], keep=False)]
    if dup.empty:
        return issues
    for (pid, d), sub in dup.groupby(["product_id", "date"]):
        issues.append({
            "product_id": _as_product_id(pid),
            "date": str(d),
            "count": int(len(sub)),
            "explanation": (
                f"Product {pid} has {len(sub)} sale record(s) on {d}. "
                "Duplicate product-date records would distort daily totals."
            ),
        })
    return issues


def validate_missing_dates(df: pd.DataFrame) -> list[dict]:
    """Report calendar gaps within each product's observed date range.

    Gaps are expected (represented as zero sales) and are reported for
    transparency, not treated as errors.
    """
    issues: list[dict] = []
    if df.empty:
        return issues
    for pid, group in df.groupby("product_id"):
        group = group.sort_values("date")
        dates = set(group["date"])
        start = min(dates)
        end = max(dates)
        missing: list[str] = []
        cur = start
        while cur <= end:
            if cur not in dates:
                missing.append(cur.isoformat())
            cur += timedelta(days=1)
        if missing:
            issues.append({
                "product_id": _as_product_id(pid),
                "missing_dates": missing,
                "count": len(missing),
                "explanation": (
                    f"Product {pid} has {len(missing)} calendar day(s) with no recorded "
                    f"sales between {start.isoformat()} and {end.isoformat()}. "
                    "These appear as zero units_sold; no transactions are fabricated."
                ),
            })
    return issues


def validate_negative_sales(df: pd.DataFrame) -> list[dict]:
    """Report rows with negative units_sold (invalid)."""
    issues: list[dict] = []
    if df.empty:
        return issues
    neg = df[df["units_sold"] < 0]
    if neg.empty:
        return issues
    for _, r in neg.iterrows():
        issues.append({
            "product_id": _as_product_id(r["product_id"]),
            "date": str(r["date"]),
            "units_sold": float(r["units_sold"]),
            "explanation": (
                f"Negative sales quantity ({r['units_sold']}) on {r['date']} is invalid."
            ),
        })
    return issues


def validate_invalid_values(df: pd.DataFrame) -> list[dict]:
    """Report rows with missing, negative, or non-integer units_sold."""
    issues: list[dict] = []
    if df.empty:
        return issues
    for _, r in df.iterrows():
        pid = _as_product_id(r["product_id"])
        d = str(r["date"])
        u = r["units_sold"]
        msgs: list[str] = []
        if pd.isna(u):
            msgs.append("units_sold is missing")
        elif u < 0:
            msgs.append(f"negative units_sold ({u})")
        elif not float(u).is_integer():
            msgs.append(f"non-integer units_sold ({u})")
        if msgs:
            issues.append({
                "product_id": pid,
                "date": d,
                "issues": msgs,
                "explanation": "; ".join(msgs) + ".",
            })
    return issues


def validate_insufficient_history(
    df: pd.DataFrame,
    config: FeatureConfig = DEFAULT_CONFIG,
) -> list[dict]:
    """Report products whose history is too short for some lag/rolling features.

    This is informational: features are still produced (as NaN where
    unavailable) so callers can decide whether a product is forecastable.
    """
    issues: list[dict] = []
    if df.empty:
        return issues
    for pid, group in df.groupby("product_id"):
        group = group.sort_values("date")
        n = len(group)
        reasons: list[str] = []
        if n < config.lag_7 + 1:
            reasons.append(
                f"only {n} day(s) of history; lag_7 requires at least "
                f"{config.lag_7 + 1} day(s) (current day plus {config.lag_7} prior days)"
            )
        if n < 2:
            reasons.append(
                "only one day of history; lag_1 and rolling features are unavailable"
            )
        elif n < config.rolling_window + 1:
            reasons.append(
                f"fewer than {config.rolling_window} past observations; rolling_avg_7/rolling_std_7 "
                "use partial windows near the start of the series"
            )
        if reasons:
            issues.append({
                "product_id": _as_product_id(pid),
                "days_of_history": int(n),
                "reasons": reasons,
                "explanation": "; ".join(reasons) + ".",
            })
    return issues


def feature_availability_summary(df: pd.DataFrame) -> list[dict]:
    """Per-product summary of which features are available."""
    summary: list[dict] = []
    if df.empty:
        return summary
    for pid, group in df.groupby("product_id"):
        group = group.sort_values("date")
        n = len(group)
        missing_lag_1 = int(group["lag_1"].isna().sum()) if "lag_1" in group.columns else n
        missing_lag_7 = int(group["lag_7"].isna().sum()) if "lag_7" in group.columns else n
        missing_avg = int(group["rolling_avg_7"].isna().sum()) if "rolling_avg_7" in group.columns else n
        missing_std = int(group["rolling_std_7"].isna().sum()) if "rolling_std_7" in group.columns else n
        summary.append({
            "product_id": _as_product_id(pid),
            "days_of_history": int(n),
            "first_date": str(group["date"].min()),
            "last_date": str(group["date"].max()),
            "total_units_sold": int(group["units_sold"].sum()),
            "missing_lag_1": missing_lag_1,
            "missing_lag_7": missing_lag_7,
            "missing_rolling_avg_7": missing_avg,
            "missing_rolling_std_7": missing_std,
        })
    return summary


def validate_feature_dataset(
    df: pd.DataFrame,
    config: FeatureConfig = DEFAULT_CONFIG,
) -> dict:
    """Validate a feature DataFrame and return a structured report.

    Returns a dict with per-check issue lists, a per-product availability
    summary, and an overall ``quality_ok`` boolean plus human-readable
    ``summary`` messages.
    """
    duplicates = validate_duplicate_product_dates(df)
    missing_dates = validate_missing_dates(df)
    negative_sales = validate_negative_sales(df)
    invalid_values = validate_invalid_values(df)
    insufficient_history = validate_insufficient_history(df, config=config)
    availability = feature_availability_summary(df)

    error_counts = (
        len(duplicates)
        + len(negative_sales)
        + len(invalid_values)
    )
    quality_ok = error_counts == 0

    messages: list[str] = []
    messages.append(f"Total feature rows: {len(df)}")
    messages.append(f"Products: {len(availability)}")
    if duplicates:
        messages.append(f"WARNING: {len(duplicates)} duplicate product-date record(s).")
    if missing_dates:
        total_missing = sum(x["count"] for x in missing_dates)
        messages.append(f"INFO: {total_missing} calendar day(s) with no sales across products (zero-filled).")
    if negative_sales:
        messages.append(f"ERROR: {len(negative_sales)} row(s) with negative sales quantity.")
    if invalid_values:
        messages.append(f"ERROR: {len(invalid_values)} row(s) with invalid units_sold values.")
    if insufficient_history:
        messages.append(
            f"INFO: {len(insufficient_history)} product(s) with limited history for lag/rolling features."
        )
    if quality_ok:
        messages.append("Data quality checks passed (no duplicate/invalid/negative sales).")
    else:
        messages.append("Data quality checks FAILED.")

    return {
        "total_rows": int(len(df)),
        "products": int(len(availability)),
        "duplicate_product_dates": duplicates,
        "missing_dates": missing_dates,
        "negative_sales": negative_sales,
        "invalid_values": invalid_values,
        "insufficient_history": insufficient_history,
        "feature_availability": availability,
        "quality_ok": bool(quality_ok),
        "summary": messages,
    }
