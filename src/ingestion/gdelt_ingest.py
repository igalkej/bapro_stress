"""
GDELT DOC 2.0 ingestor for Argentina financial news.

Fetches articles for a date range, embeds headline+themes, and stores
them idempotently in the articles + article_embeddings tables.

Usage:
    python src/ingestion/gdelt_ingest.py --date-from 2024-01-05 --date-to 2024-01-05
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from config import EMBEDDING_MODEL
from db.connection import get_engine

GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = (
    "sourcecountry:AR (economia OR finanzas OR deuda OR bonos OR mercado OR crisis OR BCRA OR Merval OR peso OR dolar)"
)
GDELT_FIELDS = "title,url,tone,themes,seendate,sourcecountry"
GDELT_MAXRECORDS = 250
GDELT_REQUEST_DELAY = 6  # seconds between API calls (GDELT rate limit: ~10 req/min)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _fetch_gdelt_day(date_str: str, session: requests.Session) -> list[dict]:
    """Call GDELT DOC 2.0 for a single day. Returns list of article dicts."""
    # GDELT date format: YYYYMMDDHHMMSS
    day = date_str.replace("-", "")
    params = {
        "query": GDELT_QUERY,
        "mode": "artlist",
        "maxrecords": GDELT_MAXRECORDS,
        "startdatetime": f"{day}000000",
        "enddatetime": f"{day}235959",
        "format": "json",
        "fields": GDELT_FIELDS,
    }
    try:
        resp = session.get(GDELT_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("articles", []) or []
    except Exception as exc:
        log.warning("GDELT API error for %s: %s", date_str, exc)
        return []


def _store_articles(conn, articles: list[dict], is_pg: bool) -> list[int]:
    """Insert articles rows, return list of new IDs (skips duplicates)."""
    new_ids = []
    for art in articles:
        url = (art.get("url") or "").strip()
        if not url:
            continue
        headline = (art.get("title") or "").strip()
        tone_raw = art.get("tone")
        try:
            gdelt_tone = float(tone_raw) if tone_raw is not None else None
        except (ValueError, TypeError):
            gdelt_tone = None
        themes_raw = art.get("themes") or ""
        gdelt_themes = themes_raw if isinstance(themes_raw, str) else ",".join(themes_raw)
        seen = art.get("seendate") or ""
        # seendate format: 20240105T123456Z  or  YYYYMMDDTHHMMSSZ
        if "T" in seen:
            date_part = seen[:8]
        else:
            date_part = seen[:8]
        if len(date_part) == 8 and date_part.isdigit():
            date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
        else:
            log.debug("Skipping article with unparseable date: %s", seen)
            continue

        if is_pg:
            sql = text(
                """
                INSERT INTO articles (date, url, headline, gdelt_tone, gdelt_themes, source)
                VALUES (:date, :url, :headline, :gdelt_tone, :gdelt_themes, :source)
                ON CONFLICT (url) DO NOTHING
                RETURNING id
                """
            )
        else:
            sql = text(
                """
                INSERT OR IGNORE INTO articles (date, url, headline, gdelt_tone, gdelt_themes, source)
                VALUES (:date, :url, :headline, :gdelt_tone, :gdelt_themes, :source)
                """
            )

        params = {
            "date": date_str,
            "url": url,
            "headline": headline,
            "gdelt_tone": gdelt_tone,
            "gdelt_themes": gdelt_themes,
            "source": "gdelt",
        }
        result = conn.execute(sql, params)

        if is_pg:
            row = result.fetchone()
            if row:
                new_ids.append(row[0])
        else:
            if result.rowcount:
                new_id = conn.execute(
                    text("SELECT id FROM articles WHERE url = :url"), {"url": url}
                ).scalar()
                if new_id:
                    new_ids.append(new_id)

    return new_ids


def _embed_and_store(conn, article_ids: list[int], articles: list[dict],
                     model: SentenceTransformer, is_pg: bool) -> None:
    """Embed articles and store in article_embeddings."""
    if not article_ids:
        return

    url_to_id = {}
    for aid, art in zip(article_ids, articles):
        url = (art.get("url") or "").strip()
        if url:
            url_to_id[url] = aid

    id_to_text = {}
    for art in articles:
        url = (art.get("url") or "").strip()
        if url not in url_to_id:
            continue
        headline = (art.get("title") or "").strip()
        themes_raw = art.get("themes") or ""
        if isinstance(themes_raw, str):
            themes_list = [t for t in themes_raw.split(",") if t]
        else:
            themes_list = list(themes_raw)
        embed_text = headline + " " + " ".join(themes_list)
        id_to_text[url_to_id[url]] = embed_text.strip()

    if not id_to_text:
        return

    ids_list = list(id_to_text.keys())
    texts = [id_to_text[i] for i in ids_list]
    vectors = model.encode(texts, convert_to_numpy=True)

    if is_pg:
        emb_sql = text(
            """
            INSERT INTO article_embeddings (id, embedding)
            VALUES (:id, :emb)
            ON CONFLICT (id) DO NOTHING
            """
        )
    else:
        emb_sql = text(
            """
            INSERT OR IGNORE INTO article_embeddings (id, embedding)
            VALUES (:id, :emb)
            """
        )

    for art_id, vec in zip(ids_list, vectors):
        vec_stored = vec.tolist() if is_pg else json.dumps(vec.tolist())
        conn.execute(emb_sql, {"id": art_id, "emb": vec_stored})


def fetch_and_store_gdelt(date_from: str, date_to: str) -> int:
    """
    Fetch GDELT articles for [date_from, date_to] business days,
    embed headlines+themes, store idempotently.

    Returns the number of new article rows inserted.
    """
    engine = get_engine()
    is_pg = not engine.url.drivername.startswith("sqlite")
    model = SentenceTransformer(EMBEDDING_MODEL)

    business_days = pd.bdate_range(date_from, date_to)
    total_new = 0

    with requests.Session() as session:
        for bday in business_days:
            day_str = bday.strftime("%Y-%m-%d")
            log.info("Fetching GDELT for %s", day_str)
            articles = _fetch_gdelt_day(day_str, session)
            log.info("  Got %d articles from GDELT", len(articles))

            if not articles:
                time.sleep(GDELT_REQUEST_DELAY)
                continue

            with engine.begin() as conn:
                new_ids = _store_articles(conn, articles, is_pg)
                if new_ids:
                    # We need to re-fetch the articles for embedding by matching new_ids
                    # Build a filtered list matching new_ids order
                    id_set = set(new_ids)
                    # Re-query to get headline+themes for new articles
                    rows = conn.execute(
                        text(
                            "SELECT id, headline, gdelt_themes FROM articles WHERE id IN :ids"
                        ),
                        {"ids": tuple(new_ids)},
                    ).fetchall()
                    art_dicts = [
                        {"url": "", "title": r.headline, "themes": r.gdelt_themes}
                        for r in rows
                    ]
                    # Use a simpler approach: embed directly from rows
                    texts = []
                    ids_order = []
                    for r in rows:
                        hl = r.headline or ""
                        th = r.gdelt_themes or ""
                        themes_list = [t for t in th.split(",") if t]
                        embed_text = (hl + " " + " ".join(themes_list)).strip()
                        texts.append(embed_text)
                        ids_order.append(r.id)

                    if texts:
                        vectors = model.encode(texts, convert_to_numpy=True)
                        if is_pg:
                            emb_sql = text(
                                """
                                INSERT INTO article_embeddings (id, embedding)
                                VALUES (:id, :emb)
                                ON CONFLICT (id) DO NOTHING
                                """
                            )
                        else:
                            emb_sql = text(
                                """
                                INSERT OR IGNORE INTO article_embeddings (id, embedding)
                                VALUES (:id, :emb)
                                """
                            )
                        for art_id, vec in zip(ids_order, vectors):
                            vec_stored = vec.tolist() if is_pg else json.dumps(vec.tolist())
                            conn.execute(emb_sql, {"id": art_id, "emb": vec_stored})

                log.info("  Inserted %d new articles for %s", len(new_ids), day_str)
                total_new += len(new_ids)

            time.sleep(GDELT_REQUEST_DELAY)

    log.info("GDELT ingest done. Total new articles: %d", total_new)
    return total_new


def main():
    parser = argparse.ArgumentParser(description="GDELT ingestor for Argentina financial news")
    parser.add_argument("--date-from", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--date-to", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    count = fetch_and_store_gdelt(args.date_from, args.date_to)
    print(f"Inserted {count} new GDELT articles.")


if __name__ == "__main__":
    main()
