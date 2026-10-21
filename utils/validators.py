"""Input and CSV validation utilities for StockWise AI.

All validation is pure Python: no formulas, macros, or arbitrary code are
ever evaluated from user-supplied data (including uploaded CSV files).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

SKU_MAX_LENGTH = 100
NAME_MAX_LENGTH = 200
CATEGORY_MAX_LENGTH = 100

REQUIRED_CSV_COLUMNS = ["sku", "name", "category", "quantity", "unit_cost", "expiry_date"]


class ValidationError(ValueError):
    """Raised when supplied input data fails validation."""


def validate_sku(value) -> str:
    if value is None:
        raise ValidationError("SKU is required.")
    text = str(value).strip()
    if not text:
        raise ValidationError("SKU is required.")
    if len(text) > SKU_MAX_LENGTH:
        raise ValidationError(f"SKU must be {SKU_MAX_LENGTH} characters or fewer.")
    return text


def validate_name(value) -> str:
    if value is None:
        raise ValidationError("Product name is required.")
    text = str(value).strip()
    if not text:
        raise ValidationError("Product name is required.")
    if len(text) > NAME_MAX_LENGTH:
        raise ValidationError(f"Product name must be {NAME_MAX_LENGTH} characters or fewer.")
    return text


def validate_category(value) -> str:
    if value is None:
        raise ValidationError("Category is required.")
    text = str(value).strip()
    if not text:
        raise ValidationError("Category is required.")
    if len(text) > CATEGORY_MAX_LENGTH:
        raise ValidationError(f"Category must be {CATEGORY_MAX_LENGTH} characters or fewer.")
    return text


def validate_quantity(value) -> int:
    if value is None or str(value).strip() == "":
        raise ValidationError("Quantity is required.")
    try:
        quantity = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError("Quantity must be a whole number.")
    if quantity < 0:
        raise ValidationError("Quantity cannot be negative.")
    return quantity


def validate_unit_cost(value) -> float:
    if value is None or str(value).strip() == "":
        raise ValidationError("Unit cost is required.")
    try:
        cost = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        raise ValidationError("Unit cost must be a number.")
    if cost < 0:
        raise ValidationError("Unit cost cannot be negative.")
    return round(cost, 2)


def validate_expiry_date(value) -> str:
    if value is None:
        raise ValidationError("Expiry date is required.")
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        raise ValidationError("Expiry date is required.")
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        raise ValidationError("Expiry date must be in YYYY-MM-DD format.")
    return parsed.isoformat()


@dataclass
class CsvRow:
    """A single parsed CSV row with row-level validation results."""

    row_number: int
    data: dict
    errors: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_csv_columns(columns) -> list:
    """Return normalized column names or raise if required columns are missing."""
    normalized = [str(c).strip() for c in columns]
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in normalized]
    if missing:
        raise ValidationError("Missing required column(s): " + ", ".join(missing))
    return normalized


def parse_and_validate_csv(file_content_bytes: bytes):
    """Parse CSV bytes and validate every row.

    Returns a tuple of ``(header, valid_rows, invalid_rows)`` where
    ``valid_rows`` is a list of cleaned dicts and ``invalid_rows`` is a list
    of :class:`CsvRow` instances carrying row-level error messages.

    Invalid rows are reported explicitly and never silently skipped. The CSV
    is parsed with Python's csv module only; no formulas, macros, or arbitrary
    code from the file are executed.
    """
    text = file_content_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValidationError("The CSV file is empty or has no header row.")
    validate_csv_columns(reader.fieldnames)

    valid_rows = []
    invalid_rows = []
    seen_skus = set()

    for row_number, row in enumerate(reader, start=2):  # header row is line 1
        csv_row = CsvRow(row_number=row_number, data=dict(row))
        errors = []

        # Presence checks for required columns.
        for col in REQUIRED_CSV_COLUMNS:
            value = (row.get(col) or "").strip()
            if value == "":
                errors.append(f"{col} is required.")

        if not errors:
            try:
                sku = validate_sku(row.get("sku"))
                if sku in seen_skus:
                    errors.append(f"Duplicate SKU '{sku}' within the import file.")
                else:
                    seen_skus.add(sku)
                name = validate_name(row.get("name"))
                category = validate_category(row.get("category"))
                quantity = validate_quantity(row.get("quantity"))
                unit_cost = validate_unit_cost(row.get("unit_cost"))
                expiry_date = validate_expiry_date(row.get("expiry_date"))
            except ValidationError as exc:
                errors.append(str(exc))
            else:
                csv_row.data = {
                    "sku": sku,
                    "name": name,
                    "category": category,
                    "quantity": quantity,
                    "unit_cost": unit_cost,
                    "expiry_date": expiry_date,
                }

        csv_row.errors = errors
        if csv_row.is_valid:
            valid_rows.append(csv_row.data)
        else:
            invalid_rows.append(csv_row)

    return reader.fieldnames, valid_rows, invalid_rows
