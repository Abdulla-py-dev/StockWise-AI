"""Database connection management and schema initialization for StockWise AI.

All paths resolve relative to the project root (the directory that contains
this ``database/`` package) rather than the current working directory, so the
application behaves the same regardless of where it is launched from.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "database"
DEFAULT_DB_PATH = DATABASE_DIR / "stockwise.db"
SCHEMA_PATH = DATABASE_DIR / "schema.sql"


def get_connection(db_path=DEFAULT_DB_PATH):
    """Open a connection to the SQLite database.

    The database file (and its parent directory) are created automatically if
    missing. An existing database is never overwritten.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database(db_path=DEFAULT_DB_PATH):
    """Create the database and schema if they do not already exist.

    Safe to call on every application start: it only creates missing objects
    and never deletes or overwrites existing data.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    try:
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()
