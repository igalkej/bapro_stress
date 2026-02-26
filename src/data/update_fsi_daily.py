"""
Daily FSI refresh script.

Downloads the latest market data, rebuilds the FSI via PCA, and upserts
the results into fsi_target and fsi_components tables.

Integrated into daily_pipeline.py so it runs automatically every business
day before the TiDE prediction step.

Usage:
    python src/data/update_fsi_daily.py
    python src/data/update_fsi_daily.py --date 2026-02-25
    docker compose run --rm app python src/data/update_fsi_daily.py
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sqlalchemy import text

from db.connection import get_engine
from src.data.build_fsi_target import build_fsi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def update_fsi_daily(target_date: str | None = None) -> dict:
    """
    Rebuild FSI from the earliest stored date up to target_date (default: today).
    Upserts both fsi_components and fsi_target in the DB.

    Returns dict: {start, end, rows}
    """
    end = target_date or pd.Timestamp.now().strftime("%Y-%m-%d")

    engine = get_engine()
    is_pg = not engine.url.drivername.startswith("sqlite")

    # Use earliest FSI date already in DB so we preserve full history
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MIN(date) FROM fsi_target")).fetchone()
    start = str(row[0])[:10] if row and row[0] else "2023-01-01"

    log.info("Rebuilding FSI from %s to %s", start, end)

    # build_fsi: downloads raw data, runs PCA, saves CSV, upserts fsi_components
    df = build_fsi(start, end)

    # build_fsi does NOT write to fsi_target — do it here
    if is_pg:
        upsert_sql = text(
            "INSERT INTO fsi_target (date, fsi_value) VALUES (:d, :v) "
            "ON CONFLICT (date) DO UPDATE SET fsi_value = EXCLUDED.fsi_value"
        )
    else:
        upsert_sql = text(
            "INSERT OR REPLACE INTO fsi_target (date, fsi_value) VALUES (:d, :v)"
        )

    with engine.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(upsert_sql, {"d": r["date"], "v": float(r["fsi_value"])})

    log.info("fsi_target updated: %d rows (%s to %s)", len(df), start, end)
    return {"start": start, "end": end, "rows": len(df)}


def main():
    parser = argparse.ArgumentParser(description="Daily FSI refresh")
    parser.add_argument(
        "--date", default=None,
        help="Target end date YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()

    result = update_fsi_daily(args.date)
    print(
        f"FSI updated: {result['rows']} rows from {result['start']} to {result['end']}"
    )


if __name__ == "__main__":
    main()
