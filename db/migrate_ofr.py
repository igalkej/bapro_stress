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


def migrate_postgres(conn):
    print("PostgreSQL migration OFR...")
    conn.execute(text(
        "ALTER TABLE fsi_components ADD COLUMN IF NOT EXISTS ofr_fsi FLOAT"
    ))
    print("  fsi_components.ofr_fsi: OK")


def migrate_sqlite(conn):
    print("SQLite migration OFR...")
    cols = [
        row[1] for row in conn.execute(
            text("PRAGMA table_info(fsi_components)")
        ).fetchall()
    ]
    if "ofr_fsi" not in cols:
        conn.execute(text(
            "ALTER TABLE fsi_components ADD COLUMN ofr_fsi REAL"
        ))
        print("  fsi_components.ofr_fsi: OK")
    else:
        print("  fsi_components.ofr_fsi already exists: skip")


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
