CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    doc_date    DATE         NOT NULL UNIQUE,
    filename    VARCHAR(255) NOT NULL,
    doc_type    VARCHAR(50)  NOT NULL DEFAULT 'bloomberg_wrap',
    content     TEXT         NOT NULL,
    created_at  TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS stress_index (
    id           SERIAL PRIMARY KEY,
    index_date   DATE  NOT NULL UNIQUE,
    stress_value FLOAT NOT NULL,
    created_at   TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS embeddings (
    id               SERIAL PRIMARY KEY,
    doc_id           INTEGER REFERENCES documents(id) UNIQUE,
    embedding_vector FLOAT[] NOT NULL,
    embedding_model  VARCHAR(255) DEFAULT 'all-MiniLM-L6-v2',
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id                SERIAL PRIMARY KEY,
    doc_id            INTEGER REFERENCES documents(id),
    stress_score_pred FLOAT  NOT NULL,
    model_version     VARCHAR(255),
    predicted_at      TIMESTAMP DEFAULT NOW()
);
