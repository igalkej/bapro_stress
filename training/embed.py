"""
Embedding catch-up step.

Encodes any articles that are missing an embedding and stores the 384-dim
vector in article_embeddings. Embed text: headline + gdelt_themes.
Idempotent: already-embedded articles are skipped.

Usage:
    python training/embed.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

from db.connection import get_engine
from config import EMBEDDING_MODEL
from src.utils.log import get_logger

log = get_logger(__name__)


def fetch_unembedded_articles(conn):
    return conn.execute(
        text(
            """
            SELECT a.id, a.headline, a.gdelt_themes
            FROM articles a
            LEFT JOIN article_embeddings ae ON ae.id = a.id
            WHERE ae.id IS NULL
            ORDER BY a.date
            """
        )
    ).fetchall()


def build_embed_text(headline, gdelt_themes) -> str:
    hl = (headline or "").strip()
    themes_raw = (gdelt_themes or "").strip()
    themes_list = [t for t in themes_raw.split(",") if t] if themes_raw else []
    return (hl + " " + " ".join(themes_list)).strip()


def store_embedding(conn, art_id: int, vector: np.ndarray, is_pg: bool):
    if is_pg:
        conn.execute(
            text(
                """
                INSERT INTO article_embeddings (id, embedding)
                VALUES (:id, :emb)
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"id": art_id, "emb": vector.tolist()},
        )
    else:
        conn.execute(
            text(
                """
                INSERT OR IGNORE INTO article_embeddings (id, embedding)
                VALUES (:id, :emb)
                """
            ),
            {"id": art_id, "emb": json.dumps(vector.tolist())},
        )


def main():
    log.info("embed_model_loading", model=EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    engine = get_engine()
    is_pg = not engine.url.drivername.startswith("sqlite")

    with engine.connect() as conn:
        articles = fetch_unembedded_articles(conn)

    if not articles:
        log.info("embed_skip", reason="all articles already have embeddings")
        return

    log.info("start... embed_articles", ts=datetime.now().strftime("%Y/%m/%d %H:%M"),
             n=len(articles))
    texts = [build_embed_text(row.headline, row.gdelt_themes) for row in articles]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    with engine.begin() as conn:
        for row, vec in zip(articles, embeddings):
            store_embedding(conn, row.id, vec, is_pg)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM article_embeddings")).scalar()

    log.info("finish... embed_articles", ts=datetime.now().strftime("%Y/%m/%d %H:%M"),
             stored=len(articles), total_in_db=count)


if __name__ == "__main__":
    main()
