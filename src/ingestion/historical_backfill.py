"""
Historical backfill: ingest GDELT and RSS articles for every business day
in a date range.

Idempotency is fully delegated to the individual ingestors — re-running the
same range a second time will not insert duplicate rows.

Usage:
    python src/ingestion/historical_backfill.py --date-from 2023-01-01 --date-to 2024-12-31
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.ingestion.gdelt_ingest import fetch_and_store_gdelt
from src.ingestion.rss_scraper import fetch_and_store_rss
from src.utils.log import get_logger

log = get_logger(__name__)


def run_backfill(date_from: str, date_to: str) -> dict:
    """
    Call fetch_and_store_gdelt + fetch_and_store_rss for every business day
    in [date_from, date_to].

    Returns dict: {total_articles, total_days, errors}.
    """
    business_days = pd.bdate_range(date_from, date_to)
    total_days = len(business_days)
    total_articles = 0
    errors = []

    log.info("start... backfill", ts=datetime.now().strftime("%Y/%m/%d %H:%M"),
             total_days=total_days, date_from=date_from, date_to=date_to)

    for bday in business_days:
        day_str = bday.strftime("%Y-%m-%d")
        log.info("Backfilling day: %s", day_str)
        day_count = 0

        try:
            gdelt_new = fetch_and_store_gdelt(day_str, day_str)
            day_count += gdelt_new
        except Exception as exc:
            log.warning("GDELT failed for %s: %s", day_str, exc)
            errors.append({"date": day_str, "source": "gdelt", "error": str(exc)})

        try:
            rss_new = fetch_and_store_rss(day_str, day_str)
            day_count += rss_new
        except Exception as exc:
            log.warning("RSS failed for %s: %s", day_str, exc)
            errors.append({"date": day_str, "source": "rss", "error": str(exc)})

        log.info("  Day %s: %d new articles", day_str, day_count)
        total_articles += day_count

    summary = {
        "total_articles": total_articles,
        "total_days": total_days,
        "errors": len(errors),
    }
    log.info("finish... backfill", ts=datetime.now().strftime("%Y/%m/%d %H:%M"),
             total_articles=total_articles, total_days=total_days, errors=len(errors))
    for err in errors:
        log.warning("backfill_day_error", **err)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Historical backfill for BAPRO article corpus")
    parser.add_argument("--date-from", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--date-to", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    summary = run_backfill(args.date_from, args.date_to)
    log.info("backfill_main_done", **summary)


if __name__ == "__main__":
    main()
