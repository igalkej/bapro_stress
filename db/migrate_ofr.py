"""
Migration OFR: Add ofr_fsi column to fsi_components.

Changes:
  - fsi_components: add `ofr_fsi` column (z-scored OFR FSI value)

Idempotent: safe to run multiple times.

Usage:
    python db/migrate_ofr.py
    docker compose run --rm app python db/migrate_ofr.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from db.connection import get_engine


OFR_COLUMNS = [
    ("ofr_fsi",         "FLOAT"),  # composite (stored as reference, not used in PCA)
    ("ofr_credit",      "FLOAT"),
    ("ofr_equity",      "FLOAT"),
    ("ofr_safe_assets", "FLOAT"),
    ("ofr_funding",     "FLOAT"),
    ("ofr_volatility",  "FLOAT"),
    ("ofr_us",          "FLOAT"),
    ("ofr_other_adv",   "FLOAT"),
    ("ofr_em",          "FLOAT"),
]


def migrate_postgres(conn):
    print("PostgreSQL migration OFR...")
    for col, dtype in OFR_COLUMNS:
        conn.execute(text(
            f"ALTER TABLE fsi_components ADD COLUMN IF NOT EXISTS {col} {dtype}"
        ))
        print(f"  fsi_components.{col}: OK")


def migrate_sqlite(conn):
    print("SQLite migration OFR...")
    existing = {
        row[1] for row in conn.execute(
            text("PRAGMA table_info(fsi_components)")
        ).fetchall()
    }
    for col, dtype in OFR_COLUMNS:
        if col not in existing:
            conn.execute(text(
                f"ALTER TABLE fsi_components ADD COLUMN {col} REAL"
            ))
            print(f"  fsi_components.{col}: OK")
        else:
            print(f"  fsi_components.{col} already exists: skip")


def main():
    engine = get_engine()
    is_pg = not engine.url.drivername.startswith("sqlite")

    with engine.begin() as conn:
        if is_pg:
            migrate_postgres(conn)
        else:
            migrate_sqlite(conn)

    print("Migration OFR complete.")
    print("Next: rebuild FSI with  python src/data/build_fsi_target.py --start ... --end ...")


if __name__ == "__main__":
    main()
