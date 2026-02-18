"""
Embedding extraction step.

Encodes each document in the `documents` table using the local
sentence-transformers model (all-MiniLM-L6-v2) and stores the resulting
384-dim float32 vector in the `embeddings` table.

Idempotent: documents already present in `embeddings` are skipped.

Usage (inside the container):
    python training/embed.py
"""
import sys
sys.path.insert(0, "/workspace")

import numpy as np
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

from db.connection import get_engine
from config import EMBEDDING_MODEL


def fetch_unembedded_docs(conn):
    rows = conn.execute(
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
    return rows


def store_embedding(conn, doc_id: int, vector: np.ndarray, model_name: str):
    vector_list = vector.tolist()
    conn.execute(
        text(
            """
            INSERT INTO embeddings (doc_id, embedding_vector, embedding_model)
            VALUES (:doc_id, :vec, :model)
            ON CONFLICT (doc_id) DO NOTHING
            """
        ),
        {"doc_id": doc_id, "vec": vector_list, "model": model_name},
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

    print(f"Encoding {len(docs)} documents …")
    texts = [row.content for row in docs]
    # batch encode — model handles batching internally
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    with engine.begin() as conn:
        for row, vec in zip(docs, embeddings):
            store_embedding(conn, row.id, vec, EMBEDDING_MODEL)

    print(f"Stored {len(docs)} embeddings in the database.")

    # Verification
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM embeddings")).scalar()
    print(f"Total embeddings in DB: {count}")


if __name__ == "__main__":
    main()
