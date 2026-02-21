"""
Load data/fsi_target.csv into the fsi_target table.

Idempotent: rows with the same date are silently skipped.

Usage:
    python db/seed_fsi.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from config import FSI_CSV
from db.connection import get_engine


def seed_fsi(conn, is_pg: bool) -> int:
    """Insert rows from fsi_target.csv. Returns count of newly inserted rows."""
    if not FSI_CSV.exists():
        print(f"fsi_target.csv not found at {FSI_CSV}. Run build_fsi_target.py first.")
        return 0

    inserted = 0
    with FSI_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_pg:
                sql = text(
                    """
                    INSERT INTO fsi_target (date, fsi_value)
                    VALUES (:date, :fsi_value)
                    ON CONFLICT (date) DO NOTHING
                    """
                )
            else:
                sql = text(
                    """
                    INSERT OR IGNORE INTO fsi_target (date, fsi_value)
                    VALUES (:date, :fsi_value)
                    """
                )
            result = conn.execute(sql, {"date": row["date"], "fsi_value": float(row["fsi_value"])})
            if result.rowcount:
                inserted += 1

    return inserted


def main():
    engine = get_engine()
    is_pg = not engine.url.drivername.startswith("sqlite")

    with engine.begin() as conn:
        count = seed_fsi(conn, is_pg)

    print(f"Seeded {count} FSI rows.")

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM fsi_target")).scalar()
    print(f"Total rows in fsi_target: {total}")


if __name__ == "__main__":
    main()
