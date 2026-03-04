"""
Wayback Machine scraper for Argentine financial news archives.

Queries the Wayback CDX API for page snapshots on configured Argentine news
domains, extracts headlines from archived HTML (streaming only the first few
KB per page to stay lightweight), and stores articles + embeddings
idempotently.  Useful for backfilling dates beyond GDELT DOC 2.0's ~90-day
rolling window.

Idempotency: articles are keyed by URL (UNIQUE constraint).  Any URL already
in the database is silently skipped, so re-running the same date range a
second time inserts zero rows.

Usage:
    python src/ingestion/wayback_scraper.py --date-from 2025-12-01 --date-to 2025-12-31
"""
import argparse
import html as _html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from config import EMBEDDING_MODEL
from db.connection import get_engine
from src.utils.log import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Source domains and article-URL patterns
# ---------------------------------------------------------------------------
# Each entry: (cdx_prefix_to_search, regex_that_must_match_article_urls)
# The regex filters out category/tag/author pages and keeps only individual
# article URLs (which tend to have longer paths or an article-ID segment).
WAYBACK_SOURCES = [
    (
        "ambito.com/economia/",
        r"ambito\.com/economia/[^?#/]{3,}/[^?#]{10,}",
    ),
    (
        "cronista.com/finanzas-mercados/",
        r"cronista\.com/finanzas-mercados/[^?#]{15,}",
    ),
    (
        "infobae.com/economia/",
        r"infobae\.com/economia/\d{4}/\d{2}/\d{2}/[^?#]{5,}",
    ),
    (
        "iprofesional.com/economia/",
        r"iprofesional\.com/economia/\d{5,}",
    ),
]

CDX_API_URL   = "https://web.archive.org/cdx/search/cdx"
CDX_DELAY     = 1.5    # seconds between CDX queries
FETCH_DELAY   = 1.5    # seconds between archived-page fetches
FETCH_TIMEOUT = 15     # seconds timeout per page fetch
MAX_CDX_ROWS  = 100    # max CDX rows per domain per day
STREAM_MAX_KB = 8      # stop streaming after this many KB (title is in first ~4 KB)


# ---------------------------------------------------------------------------
# CDX query
# ---------------------------------------------------------------------------

def _cdx_for_day(session: requests.Session,
                 domain_prefix: str,
                 day_str: str) -> list[tuple[str, str]]:
    """
    Query Wayback CDX API for archived URLs under domain_prefix on day_str.

    Returns list of (timestamp, original_url).  Each canonical URL appears
    once (collapse=urlkey).
    """
    day_nodash = day_str.replace("-", "")
    params = {
        "url":      f"{domain_prefix}*",
        "output":   "json",
        "from":     day_nodash,
        "to":       day_nodash,
        "limit":    MAX_CDX_ROWS,
        "fl":       "timestamp,original",
        "filter":   ["statuscode:200", "mimetype:text/html"],
        "collapse": "urlkey",
    }
    try:
        resp = session.get(CDX_API_URL, params=params, timeout=20)
        resp.raise_for_status()
        rows = resp.json()
    except Exception as exc:
        log.warning("cdx_request_failed", domain=domain_prefix, day=day_str,
                    error=str(exc))
        return []

    # First row is the column-name header; skip it
    if not rows or len(rows) <= 1:
        return []

    header = rows[0]
    try:
        ts_col  = header.index("timestamp")
        url_col = header.index("original")
    except ValueError:
        return []

    return [
        (r[ts_col], r[url_col])
        for r in rows[1:]
        if len(r) > max(ts_col, url_col)
    ]


def _is_article_url(url: str, pattern: str) -> bool:
    """Return True if url matches the article-URL pattern for this source."""
    return bool(re.search(pattern, url, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Archived-page title extraction
# ---------------------------------------------------------------------------

def _fetch_title(session: requests.Session, timestamp: str,
                 original_url: str) -> str | None:
    """
    Stream the Wayback snapshot just long enough to find </title>, then parse
    the title text.  Downloads at most STREAM_MAX_KB per page.

    Returns the cleaned headline string, or None on any failure.
    """
    wb_url = f"https://web.archive.org/web/{timestamp}/{original_url}"
    try:
        partial = ""
        with session.get(wb_url, stream=True, timeout=FETCH_TIMEOUT) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=1024):
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8", errors="replace")
                partial += chunk
                if len(partial) >= STREAM_MAX_KB * 1024:
                    break
                if "</title>" in partial.lower():
                    break
    except Exception as exc:
        log.debug("wayback_fetch_failed", url=wb_url, error=str(exc))
        return None

    m = re.search(r"<title[^>]*>(.*?)</title>", partial, re.IGNORECASE | re.DOTALL)
    if not m:
        return None

    title = _html.unescape(m.group(1).strip())

    # Remove common " | SiteName" or " – SiteName" suffixes
    for sep in (" | ", " – ", " — ", " · "):
        if sep in title:
            title = title[: title.rfind(sep)].strip()
            break

    title = " ".join(title.split())  # collapse whitespace
    return title if len(title) >= 8 else None


# ---------------------------------------------------------------------------
# DB helpers — idempotent, mirrors rss_scraper.py pattern
# ---------------------------------------------------------------------------

def _url_in_db(conn, url: str) -> bool:
    """Return True if this URL already exists in the articles table."""
    n = conn.execute(
        text("SELECT COUNT(*) FROM articles WHERE url = :url"), {"url": url}
    ).scalar()
    return (n or 0) > 0


def _insert_article(conn, date_str: str, url: str, headline: str,
                    is_pg: bool) -> int | None:
    """
    Insert one article row.  Returns the new row id, or None if the URL
    already exists (no-op due to ON CONFLICT / INSERT OR IGNORE).
    """
    if is_pg:
        result = conn.execute(
            text(
                """
                INSERT INTO articles (date, url, headline, source)
                VALUES (:date, :url, :headline, :source)
                ON CONFLICT (url) DO NOTHING
                RETURNING id
                """
            ),
            {"date": date_str, "url": url,
             "headline": headline, "source": "wayback"},
        )
        row = result.fetchone()
        return row[0] if row else None
    else:
        result = conn.execute(
            text(
                """
                INSERT OR IGNORE INTO articles (date, url, headline, source)
                VALUES (:date, :url, :headline, :source)
                """
            ),
            {"date": date_str, "url": url,
             "headline": headline, "source": "wayback"},
        )
        if result.rowcount:
            return conn.execute(
                text("SELECT id FROM articles WHERE url = :url"), {"url": url}
            ).scalar()
        return None


def _store_embedding(conn, art_id: int, vec, is_pg: bool) -> None:
    """Store embedding for art_id; silent no-op if already present."""
    if is_pg:
        conn.execute(
            text(
                """
                INSERT INTO article_embeddings (id, embedding)
                VALUES (:id, :emb)
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"id": art_id, "emb": vec.tolist()},
        )
    else:
        conn.execute(
            text(
                """
                INSERT OR IGNORE INTO article_embeddings (id, embedding)
                VALUES (:id, :emb)
                """
            ),
            {"id": art_id, "emb": json.dumps(vec.tolist())},
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_and_store_wayback(date_from: str, date_to: str) -> int:
    """
    For every business day in [date_from, date_to]:
      1. Query Wayback CDX for article URLs on each configured domain.
      2. Skip URLs already in the database.
      3. Stream each new archived page and extract its <title>.
      4. Insert article row + embedding (both idempotent).

    Returns the total number of new article rows inserted.
    """
    log.info("start... wayback_ingest", ts=datetime.now().strftime("%Y/%m/%d %H:%M"),
             date_from=date_from, date_to=date_to)

    engine    = get_engine()
    is_pg     = not engine.url.drivername.startswith("sqlite")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)
    total_new = 0

    business_days = pd.bdate_range(date_from, date_to)

    with requests.Session() as session:
        session.headers["User-Agent"] = (
            "Mozilla/5.0 (compatible; research-bot/1.0)"
        )

        for bday in business_days:
            day_str  = bday.strftime("%Y-%m-%d")
            day_new  = 0
            new_ids:   list[int] = []
            new_texts: list[str] = []

            for domain_prefix, article_pattern in WAYBACK_SOURCES:
                log.info("cdx_query", domain=domain_prefix, day=day_str)
                cdx_rows = _cdx_for_day(session, domain_prefix, day_str)
                log.info("cdx_results", domain=domain_prefix, day=day_str,
                         found=len(cdx_rows))
                time.sleep(CDX_DELAY)

                for timestamp, original_url in cdx_rows:
                    if not _is_article_url(original_url, article_pattern):
                        continue

                    # Check existence BEFORE fetching the page (saves bandwidth)
                    with engine.connect() as conn:
                        if _url_in_db(conn, original_url):
                            log.debug("url_already_in_db", url=original_url)
                            continue

                    headline = _fetch_title(session, timestamp, original_url)
                    time.sleep(FETCH_DELAY)

                    if not headline:
                        log.debug("no_title", url=original_url)
                        continue

                    with engine.begin() as conn:
                        art_id = _insert_article(conn, day_str, original_url,
                                                 headline, is_pg)

                    if art_id is not None:
                        new_ids.append(art_id)
                        new_texts.append(headline)
                        day_new += 1
                        log.debug("article_inserted", date=day_str,
                                  url=original_url[:60])

            # Batch-embed all new articles for this day in one model call
            if new_ids:
                vectors = emb_model.encode(new_texts, convert_to_numpy=True)
                with engine.begin() as conn:
                    for art_id, vec in zip(new_ids, vectors):
                        _store_embedding(conn, art_id, vec, is_pg)

            log.info("wayback_day_done", day=day_str, new_articles=day_new)
            total_new += day_new

    log.info("finish... wayback_ingest", ts=datetime.now().strftime("%Y/%m/%d %H:%M"),
             total_new=total_new)
    return total_new


def main():
    parser = argparse.ArgumentParser(
        description="Wayback Machine scraper for Argentine financial news archives"
    )
    parser.add_argument("--date-from", required=True, metavar="YYYY-MM-DD",
                        help="Start date (inclusive)")
    parser.add_argument("--date-to",   required=True, metavar="YYYY-MM-DD",
                        help="End date (inclusive)")
    args = parser.parse_args()

    count = fetch_and_store_wayback(args.date_from, args.date_to)
    log.info("wayback_main_done", count=count)


if __name__ == "__main__":
    main()
