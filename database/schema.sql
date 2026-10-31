-- StockWise AI — Phase 2 Inventory Schema + Phase 5 Step 3 Sales Schema
-- Local SQLite database for inventory management and sales tracking.
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

CREATE INDEX IF NOT EXISTS idx_inventory_sku ON inventory(sku);
CREATE INDEX IF NOT EXISTS idx_inventory_category ON inventory(category);
CREATE INDEX IF NOT EXISTS idx_inventory_expiry_date ON inventory(expiry_date);

CREATE TRIGGER IF NOT EXISTS trg_inventory_updated_at
AFTER UPDATE ON inventory
BEGIN
    UPDATE inventory SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Sales header (one row per transaction). Amounts in integer paise.
CREATE TABLE IF NOT EXISTS sales (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_uuid TEXT NOT NULL UNIQUE,
    transaction_date TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_amount     INTEGER NOT NULL CHECK (total_amount >= 0),
    created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sales_transaction_date ON sales(transaction_date);

-- Sale line items. Prices/costs stored as historical snapshots in paise.
CREATE TABLE IF NOT EXISTS sale_items (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    sale_id               INTEGER NOT NULL,
    product_id            INTEGER,
    product_sku_snapshot  TEXT,
    product_name_snapshot TEXT,
    quantity_sold         INTEGER NOT NULL CHECK (quantity_sold > 0),
    unit_selling_price    INTEGER NOT NULL CHECK (unit_selling_price >= 0),
    unit_cost_at_sale     INTEGER NOT NULL CHECK (unit_cost_at_sale >= 0),
    line_total            INTEGER NOT NULL CHECK (line_total >= 0),
    created_at            TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sale_id) REFERENCES sales(id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES inventory(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_sale_items_sale_id ON sale_items(sale_id);
CREATE INDEX IF NOT EXISTS idx_sale_items_product_id ON sale_items(product_id);
