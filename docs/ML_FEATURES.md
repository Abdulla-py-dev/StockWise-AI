# StockWise AI — ML Demand Forecasting: Data Preparation & Feature Engineering (Part 1)

## Status

This document describes **Part 1** of the ML pipeline: data preparation and
feature engineering only. **No model is trained or deployed in this part.**
Part 2 will cover model training and evaluation.

## Scope and safety

The following safety rules were observed throughout Part 1:

1. No fake sales or transactions are generated.
2. Existing historical sales are never modified.
3. The SQLite database is never deleted or reset.
4. The existing database schema is unchanged.
5. Existing inventory, sales, expiry, and reporting functionality is untouched.
6. No model is trained (insufficient genuine data is handled by readiness checks).
7. Existing ML readiness thresholds are unchanged.
8. The Emerald & Charcoal UI theme is preserved.
9. Changes are integrated incrementally (new service + tests + docs); nothing is rebuilt.

## 1. Dataset structure

The ML dataset is built from genuine historical sales records stored in SQLite.

### Source tables (`database/schema.sql`)

| Table | Role |
|-------|------|
| `sales` | One row per transaction: `id`, `transaction_uuid`, `transaction_date`, `total_amount` (integer paise), `created_at`. |
| `sale_items` | One row per line item: `sale_id`, `product_id` (nullable, `ON DELETE SET NULL`), `product_sku_snapshot`, `product_name_snapshot`, `quantity_sold` (>0), `unit_selling_price` (paise), `unit_cost_at_sale` (paise), `line_total` (paise), `created_at`. |
| `inventory` | Current product master: `id`, `sku`, `name`, `category`, `quantity`, `unit_cost`, `expiry_date`, timestamps. |

### How the time series is produced

`services.ml_readiness_service.build_timeseries()` joins `sale_items` to `sales`
and aggregates `quantity_sold` per `product_id` per calendar date. Missing
calendar days within each product's observed first-to-last range are filled
with `0` units so the series is contiguous. **No transactions are fabricated**
and no historical sale is altered.

Rows whose `product_id` is `NULL` (historical snapshots for deleted products)
are preserved in the underlying time series but are **excluded from the ML
feature dataset**, because they carry no stable, current product identifier.

### ML feature dataset columns

One row per product per calendar day, produced by
`services.ml_features.build_feature_dataset()` / `engineer_features()`:

| Column | Type | Meaning |
|--------|------|---------|
| `product_id` | int | Stable product identifier |
| `sku` | str \| None | Product SKU snapshot |
| `name` | str \| None | Product name snapshot |
| `date` | date | Calendar date |
| `units_sold` | int | Genuine units sold that day (target; never fabricated) |
| `day_of_week` | int | 0=Monday … 6=Sunday |
| `day_of_month` | int | 1 … 31 |
| `month` | int | 1 … 12 |
| `is_weekend` | int | 1 if Saturday/Sunday, else 0 |
| `lag_1` | float \| NaN | Units sold the previous day |
| `lag_7` | float \| NaN | Units sold 7 days ago (only with sufficient history) |
| `rolling_avg_7` | float \| NaN | Mean units sold over the previous 7 days (past only) |
| `rolling_std_7` | float \| NaN | Sample std dev over the previous 7 days (past only) |

## 2. Feature definitions

### Calendar features
- `day_of_week`: extracted from the date only (no sales involved).
- `day_of_month`: day of the month, 1–31.
- `month`: month of the year, 1–12.
- `is_weekend`: 1 for Saturday/Sunday, else 0.

### Target
- `units_sold`: genuine quantity sold on that date, aggregated from real
  `sale_items`. Missing calendar days are `0` (zero-filled), never invented.

### Lag features (past observations only)
- `lag_1 = units_sold[t-1]` via `shift(1)`. NaN on the first day of a series.
- `lag_7 = units_sold[t-7]` via `shift(7)`. NaN for the first 7 days and
  whenever fewer than 7 prior days of history exist.

### Rolling features (past observations only, current day excluded)
- `rolling_avg_7 = mean(units_sold[t-7 .. t-1])`, computed on the lagged
  series (`units.shift(1).rolling(7).mean()`). Uses partial windows near the
  start of a series (`min_periods=1`).
- `rolling_std_7 = std(units_sold[t-7 .. t-1])`, sample standard deviation
  (`ddof=1`), computed on the lagged series. NaN when fewer than 2 past
  observations exist (`min_periods=2` effectively, since `std` of one value is
  undefined).

### Configuration

`FeatureConfig` (in `services/ml_features.py`) exposes `lag_1`, `lag_7`,
`rolling_window`, `min_rolling_obs_mean`, and `min_rolling_obs_std`. Defaults:
`lag_1=1`, `lag_7=7`, `rolling_window=7`, `min_rolling_obs_mean=1`,
`min_rolling_obs_std=2`.

## 3. Data leakage prevention

Leakage is prevented by construction:

1. **Chronological ordering**: observations are sorted by date within each
   product before any feature is computed.
2. **Product-specific computation**: all lag/rolling features are computed
   inside a `groupby("product_id")`, so one product's sales never influence
   another's features.
3. **Shifted lags**: `lag_1`/`lag_7` use `shift`, so only strictly prior days'
   sales are used. The current day's sales never appear in its own lag.
4. **Rolling on lagged series**: rolling statistics are computed on
   `units.shift(1)`, so the window `[t-7 .. t-1]` excludes the current day `t`.
   This guarantees `rolling_avg_7[t]` never incorporates `units_sold[t]`.
5. **No future data**: every feature at date `t` depends only on dates `<= t-1`
   (calendar features depend only on `t` itself).

## 4. Missing value handling

Unavailable lag/rolling values (early days of a product's history) are left
as `NaN`. They are **never imputed or fabricated**. Callers can inspect
`validate_feature_dataset()` results or `feature_availability_summary()` to
see exactly how many rows have missing values per feature and per product.

## 5. Dataset validation

`services.ml_features.validate_feature_dataset(df)` runs these checks:

| Check | What it detects | Severity |
|-------|----------------|----------|
| Duplicate product-date records | More than one sales row for the same product on the same date | Error |
| Missing dates | Calendar days within a product's first–last range with no sales | Info (zero-filled) |
| Negative sales quantities | `units_sold < 0` | Error |
| Invalid values | Missing, negative, or non-integer `units_sold` | Error |
| Insufficient history for lag features | Products with too few days for `lag_7` / rolling features | Info |
| Product-level data availability | Per-product days of history, first/last date, total units, missing-feature counts | Summary |

The report returns per-check issue lists (each with a human-readable
`explanation`), a per-product feature-availability summary, an overall
`quality_ok` boolean, and human-readable `summary` messages.

## 6. Readiness requirements

The ML readiness criteria (unchanged from
`services.ml_readiness_service.py`) must all be met before forecasting is
considered eligible:

- At least **30** genuine transactions.
- At least **10** calendar days of historical coverage.
- At least **3** distinct products with sales.
- Data quality checks must pass (`quality_ok == True`).

These thresholds are **not** altered by Part 1.

## 7. Limitations of the available sales history

- **History depth**: forecasting eligibility requires 30+ transactions over
  10+ days across 3+ products. If the live database does not meet these, the
  feature dataset will be small and `quality_ok` may be False.
- **Deleted products**: sales whose `product_id` is `NULL` (product deleted
  after the sale) are excluded from ML features; only snapshots remain.
- **No external features**: features are derived solely from internal sales
  dates and quantities. No pricing, promotions, seasonality calendars, or
  external demand signals are used in Part 1.
- **Zero-filled gaps**: missing calendar days are represented as `0` units
  sold. This is honest (no sale occurred) but should not be interpreted as
  recorded zero-demand transactions.
- **Sample standard deviation**: `rolling_std_7` uses `ddof=1` and is NaN when
  fewer than 2 past observations exist.
- **Part 1 is preparation only**: no model is trained, validated, or deployed.
  Suitability for training is determined by the readiness checks above.

## 8. Files

| File | Purpose |
|------|---------|
| `services/ml_features.py` | Feature engineering service + dataset validation (new) |
| `tests/test_ml_features.py` | Comprehensive unit tests, synthetic data only (new) |
| `docs/ML_FEATURES.md` | This document (new) |

No existing files were modified.

## 9. Recommended next step (Part 2)

Once the live database satisfies the readiness criteria (30+ transactions,
10+ days, 3+ products, quality OK), proceed to Part 2: model training and
evaluation using the feature dataset produced here (e.g. a scikit-learn
regressor with time-series cross-validation, never using future data). Until
then, forecasting remains disabled and the readiness message guides the user
to record more genuine sales.
