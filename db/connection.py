from sqlalchemy import create_engine, text
from config import DATABASE_URL

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_date    TEXT NOT NULL UNIQUE,
    filename    TEXT NOT NULL,
    doc_type    TEXT NOT NULL DEFAULT 'bloomberg_wrap',
    content     TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stress_index (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    index_date   TEXT NOT NULL UNIQUE,
    stress_value REAL NOT NULL,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS embeddings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id           INTEGER REFERENCES documents(id) UNIQUE,
    embedding_vector TEXT NOT NULL,
    embedding_model  TEXT DEFAULT 'all-MiniLM-L6-v2',
    created_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id            INTEGER REFERENCES documents(id),
    stress_score_pred REAL NOT NULL,
    model_version     TEXT,
    predicted_at      TEXT DEFAULT (datetime('now'))
);
"""


def get_engine():
    engine = create_engine(DATABASE_URL)
    if DATABASE_URL.startswith("sqlite"):
        with engine.begin() as conn:
            for stmt in _SQLITE_SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
    return engine
