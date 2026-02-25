# BAPRO Financial Stress System

Argentina sovereign credit stress prediction pipeline. Ingests financial news from GDELT and Argentine media, builds a quantitative Financial Stress Index (FSI) from market data, and trains a Darts TiDE model to predict next-day stress levels.

## Features

- **Real-time ingestion** — GDELT DOC 2.0 (Argentina filter) + RSS from Ambito, Cronista, Infobae
- **FSI ground truth** — PCA of ^MERV volatility, ARGT sovereign spread, ARS/USD FX, EMB
- **TiDE forecasting** — 384-dim sentence-transformer embeddings as covariates
- **Optuna tuning** — automated hyperparameter search with trial history stored in DB
- **Dash dashboard** — historical FSI chart, training fit, daily prediction gauge
- **Daily automation** — GitHub Actions cron Mon-Fri 06:00 UTC

## Quick start (Docker)

```bash
git clone https://github.com/igalkej/bapro_stress.git
cd bapro_stress

# 1. Build images and start database
docker compose build && docker compose up -d postgres

# 2. Build FSI target from market data
docker compose run --rm app python src/data/build_fsi_target.py \
    --start 2025-11-01 --end 2026-02-24

# 3. Seed FSI and ingest historical articles
docker compose run --rm app python db/seed_fsi.py
docker compose run --rm app python src/ingestion/historical_backfill.py \
    --date-from 2025-11-01 --date-to 2026-02-24

# 4. Embed articles and train model
docker compose run --rm app python training/embed.py
docker compose run --rm app python training/train.py

# 5. Run today's prediction and start dashboard
docker compose run --rm app python src/ingestion/daily_pipeline.py
docker compose up -d dashboard
```

Dashboard → http://localhost:8050
pgAdmin → http://localhost:5050 (admin@bapro.com / admin)

## Dashboard tabs

| Tab | Contents |
|-----|---------|
| **Entrenamiento** | FSI historical series + out-of-sample val/test fit + model metrics + EDA panels |
| **Predicciones** | Daily stress score gauge with 95% CI, prediction history chart, date selector + click-to-select date → news article viewer |

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
│   │   └── build_fsi_target.py      # Compute FSI via yfinance + PCA
│   └── ingestion/
│       ├── gdelt_ingest.py          # GDELT DOC 2.0 fetcher
│       ├── rss_scraper.py           # RSS scraper (Ambito/Cronista/Infobae)
│       ├── daily_pipeline.py        # Daily ingest + predict orchestrator
│       └── historical_backfill.py   # One-time historical article download
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

Active tables:

| Table | Description |
|-------|-------------|
| `articles` | News articles (headline, source, URL, GDELT tone) |
| `article_embeddings` | 384-dim sentence-transformer embeddings |
| `fsi_target` | Daily FSI value (PCA z-score) |
| `fsi_components` | Normalised component values |
| `training_predictions` | Out-of-sample val/test predictions from training |
| `daily_predictions` | Forward predictions from daily pipeline |
| `optuna_trials` | Hyperparameter search trial history |

## ML details

- **Model**: TiDE (Temporal Identity Encoder) — Darts 0.41.0
- **Input**: `input_chunk_length` business days look-back (tuned by Optuna); 1 day ahead
- **Lag offset**: val/test evaluation starts at `split_boundary + input_chunk_length` to avoid look-back window leaking training observations into held-out sets
- **Covariates**: mean-pooled 384-dim article embeddings (past + future); zero vector on days without articles
- **Uncertainty**: 95% CI from test-set RMSE
- **Artifact**: `artifacts/tide_model.pt`

## Requirements

- Docker + Docker Compose
- Python 3.11 (local dev)

See `requirements.txt` for Python dependencies.
