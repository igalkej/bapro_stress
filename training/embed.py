"""
Embedding extraction step.

Encodes each document with sentence-transformers (all-MiniLM-L6-v2) and
stores the 384-dim vector as a JSON string in the embeddings table.
Idempotent: already-embedded documents are skipped.

Usage:
    python training/embed.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

from db.connection import get_engine
from config import EMBEDDING_MODEL


def fetch_unembedded_docs(conn):
    return conn.execute(
        text(
            """
            SELECT d.id, d.content
            FROM documents d
            LEFT JOIN embeddings e ON e.doc_id = d.id
            WHERE e.id IS NULL
            ORDER BY d.doc_date
            """
        )
    ).fetchall()


def store_embedding(conn, doc_id: int, vector: np.ndarray, model_name: str):
    conn.execute(
        text(
            """
            INSERT OR IGNORE INTO embeddings (doc_id, embedding_vector, embedding_model)
            VALUES (:doc_id, :vec, :model)
            """
        ),
        {"doc_id": doc_id, "vec": json.dumps(vector.tolist()), "model": model_name},
    )


def main():
    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    engine = get_engine()

    with engine.connect() as conn:
        docs = fetch_unembedded_docs(conn)

    if not docs:
        print("All documents already have embeddings. Nothing to do.")
        return

    print(f"Encoding {len(docs)} documents...")
    texts = [row.content for row in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    with engine.begin() as conn:
        for row, vec in zip(docs, embeddings):
            store_embedding(conn, row.id, vec, EMBEDDING_MODEL)

    print(f"Stored {len(docs)} embeddings in the database.")

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM embeddings")).scalar()
    print(f"Total embeddings in DB: {count}")


if __name__ == "__main__":
    main()
