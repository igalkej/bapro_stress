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

CREATE TABLE IF NOT EXISTS articles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    date          TEXT NOT NULL,
    url           TEXT NOT NULL UNIQUE,
    headline      TEXT,
    gdelt_tone    REAL,
    gdelt_themes  TEXT,
    source        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS article_embeddings (
    id        INTEGER PRIMARY KEY REFERENCES articles(id),
    embedding TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fsi_target (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    date      TEXT NOT NULL UNIQUE,
    fsi_value REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS fsi_components (
    date        TEXT NOT NULL UNIQUE,
    merv_vol    REAL,
    argt_spread REAL,
    usd_ars     REAL,
    emb_spread  REAL
);

CREATE TABLE IF NOT EXISTS training_predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    date          TEXT NOT NULL,
    fsi_actual    REAL,
    fsi_pred      REAL NOT NULL,
    split         TEXT NOT NULL,
    model_version TEXT NOT NULL,
    UNIQUE (date, split)
);

CREATE TABLE IF NOT EXISTS daily_predictions (
    date          TEXT NOT NULL UNIQUE,
    fsi_pred      REAL NOT NULL,
    model_version TEXT,
    predicted_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS optuna_trials (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    study_name    TEXT    NOT NULL,
    trial_number  INTEGER NOT NULL,
    rank_val      INTEGER NOT NULL,
    mape_val      REAL    NOT NULL,
    mape_test     REAL,
    mae_test      REAL,
    rmse_test     REAL,
    is_production INTEGER DEFAULT 0,
    hyperparams   TEXT    NOT NULL,
    model_version TEXT,
    created_at    TEXT    DEFAULT (datetime('now'))
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
