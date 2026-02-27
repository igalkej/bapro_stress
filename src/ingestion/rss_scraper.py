"""
RSS ingestor for Argentine financial news feeds.

Parses Ambito, Cronista, and Infobae RSS feeds, filters articles by date,
embeds headlines, and stores them idempotently in articles + article_embeddings.

Usage:
    python src/ingestion/rss_scraper.py --date-from 2024-01-01 --date-to 2024-01-31
"""
import argparse
import json
import sys
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import feedparser
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from config import EMBEDDING_MODEL
from db.connection import get_engine
from src.utils.log import get_logger

RSS_FEEDS = {
    "ambito_rss":   "https://www.ambito.com/rss/economia.xml",
    "cronista_rss": "https://www.cronista.com/rss/mercados.rss",
    "infobae_rss":  "https://www.infobae.com/feeds/rss/economia/",
}

log = get_logger(__name__)


def _parse_pub_date(entry) -> str | None:
    """Parse feed entry pub date to YYYY-MM-DD string. Returns None on failure."""
    raw = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    # Fallback: try feedparser's parsed time tuple
    tt = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if tt:
        try:
            import time
            return pd.Timestamp(time.mktime(tt), unit="s").strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


def _insert_article(conn, date_str: str, url: str, headline: str,
                    source: str, is_pg: bool) -> int | None:
    """Insert one article row. Returns new id or None if duplicate."""
    if is_pg:
        sql = text(
            """
            INSERT INTO articles (date, url, headline, source)
            VALUES (:date, :url, :headline, :source)
            ON CONFLICT (url) DO NOTHING
            RETURNING id
            """
        )
        result = conn.execute(sql, {"date": date_str, "url": url,
                                    "headline": headline, "source": source})
        row = result.fetchone()
        return row[0] if row else None
    else:
        sql = text(
            """
            INSERT OR IGNORE INTO articles (date, url, headline, source)
            VALUES (:date, :url, :headline, :source)
            """
        )
        result = conn.execute(sql, {"date": date_str, "url": url,
                                    "headline": headline, "source": source})
        if result.rowcount:
            new_id = conn.execute(
                text("SELECT id FROM articles WHERE url = :url"), {"url": url}
            ).scalar()
            return new_id
        return None


def _store_embedding(conn, art_id: int, vec, is_pg: bool) -> None:
    """Store one embedding row, skip if already exists."""
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


def fetch_and_store_rss(date_from: str, date_to: str) -> int:
    """
    Parse all RSS feeds, filter entries by [date_from, date_to],
    embed headlines, and store idempotently.

    Returns the number of new article rows inserted.
    """
    log.info("start... rss_ingest", ts=datetime.now().strftime("%Y/%m/%d %H:%M"),
             date_from=date_from, date_to=date_to)
    engine = get_engine()
    is_pg = not engine.url.drivername.startswith("sqlite")
    model = SentenceTransformer(EMBEDDING_MODEL)

    date_from_ts = pd.Timestamp(date_from)
    date_to_ts = pd.Timestamp(date_to)

    total_new = 0

    for feed_key, feed_url in RSS_FEEDS.items():
        log.info("Parsing feed: %s (%s)", feed_key, feed_url)
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as exc:
            log.warning("Failed to parse %s: %s", feed_key, exc)
            continue

        entries_in_range = []
        for entry in parsed.entries:
            url = getattr(entry, "link", None)
            headline = getattr(entry, "title", None)
            if not url or not headline:
                continue
            url = url.strip()
            headline = headline.strip()
            pub_date_str = _parse_pub_date(entry)
            if not pub_date_str:
                continue
            pub_ts = pd.Timestamp(pub_date_str)
            if pub_ts < date_from_ts or pub_ts > date_to_ts:
                continue
            entries_in_range.append((pub_date_str, url, headline))

        log.info("  %d entries in range [%s, %s]", len(entries_in_range), date_from, date_to)

        new_ids = []
        new_headlines = []

        with engine.begin() as conn:
            for date_str, url, headline in entries_in_range:
                new_id = _insert_article(conn, date_str, url, headline, feed_key, is_pg)
                if new_id is not None:
                    new_ids.append(new_id)
                    new_headlines.append(headline)

        if new_ids:
            vectors = model.encode(new_headlines, convert_to_numpy=True)
            with engine.begin() as conn:
                for art_id, vec in zip(new_ids, vectors):
                    _store_embedding(conn, art_id, vec, is_pg)

        log.info("  Inserted %d new articles from %s", len(new_ids), feed_key)
        total_new += len(new_ids)

    log.info("finish... rss_ingest", ts=datetime.now().strftime("%Y/%m/%d %H:%M"),
             total_new=total_new)
    return total_new


def main():
    parser = argparse.ArgumentParser(description="RSS ingestor for Argentine financial news")
    parser.add_argument("--date-from", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--date-to", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    count = fetch_and_store_rss(args.date_from, args.date_to)
    log.info("rss_main_done", count=count)


if __name__ == "__main__":
    main()
