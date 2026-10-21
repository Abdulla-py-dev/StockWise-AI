-- StockWise AI — Phase 2 Inventory Schema
-- Local SQLite database for inventory management.
-- Created automatically by database/db.py on application startup.
-- An existing database is never overwritten.

CREATE TABLE IF NOT EXISTS inventory (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sku           TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    category      TEXT NOT NULL,
    quantity      INTEGER NOT NULL CHECK (quantity >= 0),
    unit_cost     REAL NOT NULL CHECK (unit_cost >= 0),
    expiry_date   TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for lookups, category filtering, and expiry-date ordering.
CREATE INDEX IF NOT EXISTS idx_inventory_sku ON inventory(sku);
CREATE INDEX IF NOT EXISTS idx_inventory_category ON inventory(category);
CREATE INDEX IF NOT EXISTS idx_inventory_expiry_date ON inventory(expiry_date);

-- Keep updated_at current on every row update.
CREATE TRIGGER IF NOT EXISTS trg_inventory_updated_at
AFTER UPDATE ON inventory
BEGIN
    UPDATE inventory SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
