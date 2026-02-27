# BAPRO Financial Stress System

Argentina sovereign credit stress nowcasting pipeline. Ingests financial news from GDELT and Argentine media, builds a quantitative Financial Stress Index (FSI) from market data, and trains a Darts TiDE model to nowcast stress levels for days where articles exist but FSI has not yet been published.

## Features

- **Real-time ingestion** — GDELT DOC 2.0 (Argentina filter) + RSS from Ambito, Cronista, Infobae
- **FSI ground truth** — PCA of ^MERV volatility, ARGT sovereign spread, ARS/USD FX, EMB
- **TiDE nowcasting** — 384-dim sentence-transformer embeddings as past + future covariates
- **Dynamic gap nowcasting** — predicts exactly the business days between last published FSI and today, using real article embeddings (not zero vectors)
- **Optuna tuning** — automated hyperparameter search with trial history stored in DB
- **Dash dashboard** — historical FSI chart, training fit, daily prediction gauge
- **Structured logging** — `structlog` with start/finish bracketing across all pipeline modules
- **Daily automation** — GitHub Actions cron Mon-Fri 06:00 UTC

## Quick start (Docker)

```bash
git clone https://github.com/igalkej/bapro_stress.git
cd bapro_stress

# 1. Build images and start database
docker compose build && docker compose up -d postgres

# 2. Build FSI target from market data
docker compose run --rm app python src/data/build_fsi_target.py \
    --start 2025-11-01 --end 2026-02-27

# 3. Seed FSI and ingest historical articles
docker compose run --rm app python db/seed_fsi.py
docker compose run --rm app python src/ingestion/historical_backfill.py \
    --date-from 2025-11-01 --date-to 2026-02-27

# 4. Embed articles and train model
docker compose run --rm app python training/embed.py
docker compose run --rm app python training/train.py

# 5. Run today's nowcast and start dashboard
docker compose run --rm app python src/ingestion/daily_pipeline.py
docker compose up -d dashboard
```

Dashboard → http://localhost:8050
pgAdmin → http://localhost:5050 (admin@bapro.com / admin)

## Dashboard tabs

| Tab | Contents |
|-----|---------|
| **Entrenamiento** | FSI series bounded to training period (train+val+test), out-of-sample fit, model metrics, EDA panels |
| **Predicciones** | Nowcast stress gauge with 95% CI, prediction history chart, date selector + article viewer |
| **Comparar Modelos** | Multi-model overlay chart comparing all Optuna finalist predictions |

## Repository structure

```
bapro_stress/
├── config.py                        # Centralised constants and paths
├── db/
│   ├── connection.py                # get_engine() — SQLite auto-schema
│   ├── schema.sql                   # PostgreSQL schema
│   └── seed_fsi.py                  # Load fsi_target.csv into DB
├── src/
│   ├── data/
│   │   ├── build_fsi_target.py      # Compute FSI via yfinance + PCA
│   │   └── update_fsi_daily.py      # Append latest market data to fsi_target
│   ├── ingestion/
│   │   ├── gdelt_ingest.py          # GDELT DOC 2.0 fetcher
│   │   ├── rss_scraper.py           # RSS scraper (Ambito/Cronista/Infobae)
│   │   ├── daily_pipeline.py        # Daily ingest + nowcast orchestrator
│   │   └── historical_backfill.py   # One-time historical article download
│   └── utils/
│       └── log.py                   # Centralised structlog setup (get_logger)
├── training/
│   ├── embed.py                     # Encode articles → article_embeddings
│   └── train.py                     # TiDE training + Optuna search
├── prediction/
│   └── predict.py                   # CLI: score FSI for a given date
├── dashboard/
│   └── app.py                       # Dash application
├── notebooks/
│   └── 01_eda_corpus_real.ipynb     # Exploratory data analysis
├── docs/
│   └── plan.md                      # Architecture and pipeline docs
├── .github/workflows/
│   └── daily_pipeline.yml           # GitHub Actions cron
└── docker-compose.yml
```

## Database schema

| Table | Description |
|-------|-------------|
| `articles` | News articles (headline, source, URL, GDELT tone) |
| `article_embeddings` | 384-dim sentence-transformer embeddings |
| `fsi_target` | Daily FSI value (PCA z-score) |
| `fsi_components` | Normalised component values |
| `models` | Optuna finalist model metadata + metrics |
| `training_predictions` | Out-of-sample val/test predictions per finalist trial |
| `training_loss` | Epoch-level train/val loss per finalist trial |
| `optuna_trials` | Hyperparameter search trial history |
| `daily_predictions` | Nowcast scores from daily pipeline |

## ML details

- **Model**: TiDE (Temporal Identity Encoder) — Darts 0.41.0
- **Alignment**: nowcasting — news at date t predicts FSI at date t (not t+1)
- **Input**: `input_chunk_length` business days look-back (tuned by Optuna); 1 day ahead output
- **Gap nowcasting**: `n = business days from last published FSI to target date`; uses real article embeddings for gap days as `future_covariates`
- **Lag offset**: val/test evaluation starts at `split_boundary + input_chunk_length` to avoid look-back window leaking training observations into held-out sets
- **Covariates**: mean-pooled 384-dim article embeddings (past + future); zero vector on days without articles
- **Uncertainty**: 95% CI from test-set RMSE
- **Artifact**: `artifacts/tide_model.pt`

## Logging

All pipeline modules use `structlog` via `src/utils/log.py`:

```python
from src.utils.log import get_logger
log = get_logger(__name__)

log.info("start... backfill", ts="2026/02/27 10:00")
# ... work ...
log.info("finish... backfill", ts="2026/02/27 10:45", rows=1234)
```

Log level and format are controlled via env vars `LOG_LEVEL` (default `INFO`) and `LOG_FORMAT` (`console` or `json`).

## Requirements

- Docker + Docker Compose
- Python 3.11 (local dev)

See `requirements.txt` for Python dependencies.
