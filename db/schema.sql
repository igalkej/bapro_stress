CREATE TABLE IF NOT EXISTS articles (
    id            SERIAL PRIMARY KEY,
    date          DATE         NOT NULL,
    url           VARCHAR(2048) NOT NULL UNIQUE,
    headline      TEXT,
    gdelt_tone    FLOAT,
    gdelt_themes  TEXT,
    source        VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS article_embeddings (
    id        INTEGER PRIMARY KEY REFERENCES articles(id),
    embedding FLOAT[] NOT NULL
);

CREATE TABLE IF NOT EXISTS fsi_target (
    id        SERIAL PRIMARY KEY,
    date      DATE  NOT NULL UNIQUE,
    fsi_value FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS fsi_components (
    date         DATE  NOT NULL UNIQUE,
    merv_vol     FLOAT,
    argt_spread  FLOAT,
    usd_ars      FLOAT,
    emb_spread   FLOAT
);

CREATE TABLE IF NOT EXISTS training_predictions (
    id            SERIAL PRIMARY KEY,
    date          DATE  NOT NULL,
    fsi_actual    FLOAT,
    fsi_pred      FLOAT NOT NULL,
    split         VARCHAR(10) NOT NULL,
    horizon       INT   NOT NULL DEFAULT 1,
    trial_number  INT   NOT NULL DEFAULT 0,
    model_version VARCHAR(255) NOT NULL,
    UNIQUE (date, split, horizon, trial_number)
);

CREATE TABLE IF NOT EXISTS daily_predictions (
    date          DATE         NOT NULL UNIQUE,
    fsi_pred      FLOAT        NOT NULL,
    model_version VARCHAR(255),
    predicted_at  TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS optuna_trials (
    id            SERIAL PRIMARY KEY,
    study_name    VARCHAR(255) NOT NULL,
    trial_number  INT          NOT NULL,
    rank_val      INT          NOT NULL,
    mape_val      FLOAT        NOT NULL,
    mape_test     FLOAT,
    mae_test      FLOAT,
    rmse_test     FLOAT,
    is_production BOOLEAN      DEFAULT FALSE,
    hyperparams   TEXT         NOT NULL,
    model_version VARCHAR(255),
    created_at    TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS models (
    id            SERIAL PRIMARY KEY,
    model_version VARCHAR(255) NOT NULL,
    trial_number  INT          NOT NULL,
    rank_val      INT          NOT NULL,
    is_production BOOLEAN      NOT NULL DEFAULT FALSE,
    hyperparams   TEXT         NOT NULL,
    architecture  TEXT         NOT NULL,
    train_samples INT,
    val_samples   INT,
    test_samples  INT,
    mape_val      FLOAT,
    mape_test     FLOAT,
    mae_test      FLOAT,
    rmse_test     FLOAT,
    artifact_path VARCHAR(512),
    created_at    TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training_loss (
    id            SERIAL PRIMARY KEY,
    model_version VARCHAR(255) NOT NULL,
    trial_number  INT          NOT NULL,
    rank_val      INT          NOT NULL,
    epoch         INT          NOT NULL,
    train_loss    FLOAT,
    val_loss      FLOAT
);
