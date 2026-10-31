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


import hashlib
import shutil

def _backup_if_exists(db_path: Path) -> None:
    """Create a verified backup of an existing database before migration.

    Idempotent: only creates a backup if one does not already exist, so repeated
    startups do not overwrite the original backup.
    """
    if not db_path.exists():
        return
    backup_path = db_path.with_suffix(db_path.suffix + ".bak")
    if backup_path.exists():
        return
    shutil.copy2(db_path, backup_path)
    # Verify by comparing size and hash.
    if db_path.stat().st_size != backup_path.stat().st_size:
        raise RuntimeError("Database backup size mismatch.")
    def _hash(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    if _hash(db_path) != _hash(backup_path):
        raise RuntimeError("Database backup hash mismatch.")


def init_database(db_path=DEFAULT_DB_PATH):
    """Create the database and schema if they do not already exist.

    Safe to call on every application start: it only creates missing objects
    and never deletes or overwrites existing data. An existing database is
    backed up (once) before schema migration is applied.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _backup_if_exists(db_path)
    conn = get_connection(db_path)
    try:
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()
