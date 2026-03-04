# BAPRO Financial Stress System

Argentina sovereign credit stress nowcasting pipeline. Ingests financial news from GDELT and Argentine media, builds a quantitative Financial Stress Index (FSI) from market data and global stress indicators, and trains a Darts TiDE model to nowcast stress levels for days where articles exist but FSI has not yet been published.

## Features

- **Real-time ingestion** — GDELT DOC 2.0 (Argentina filter) + RSS from Ambito, Cronista, Infobae
- **FSI ground truth** — PCA of 4 Argentine market series + 8 OFR FSI sub-indicators (global stress context)
- **TiDE nowcasting** — 384-dim sentence-transformer embeddings as past + future covariates
- **Dynamic gap nowcasting** — predicts exactly the business days between last published FSI and today, using real article embeddings (not zero vectors)
- **Optuna tuning** — automated hyperparameter search with trial history stored in DB
- **Dash dashboard** — historical FSI chart, training fit, daily prediction gauge, multi-model comparison
- **Structured logging** — `structlog` with start/finish bracketing across all pipeline modules
- **Daily automation** — GitHub Actions cron Mon-Fri 06:00 UTC

## Quick start (Docker)

```bash
git clone https://github.com/igalkej/bapro_stress.git
cd bapro_stress

# 1. Build images and start database
docker compose build && docker compose up -d postgres

# 2. (First run only) Run DB migrations
docker compose run --rm app python db/migrate_ml03.py
docker compose run --rm app python db/migrate_ml04.py
docker compose run --rm app python db/migrate_ofr.py

# 3. Build FSI target from market data
docker compose run --rm app python src/data/build_fsi_target.py \
    --start 2000-01-03 --end 2026-03-03

# 4. Ingest historical articles
docker compose run --rm app python src/ingestion/historical_backfill.py \
    --date-from 2000-01-03 --date-to 2026-03-03

# 5. Embed articles and train model
docker compose run --rm app python training/embed.py
docker compose run --rm app python training/train.py

# 6. Run today's nowcast and start dashboard
docker compose run --rm app python src/ingestion/daily_pipeline.py
docker compose up -d dashboard
```

Dashboard → http://localhost:8050
pgAdmin → http://localhost:5050 (admin@bapro.com / admin)

## Dashboard tabs

| Tab | Contents |
|-----|---------|
| **Entrenamiento** | FSI series bounded to training period, out-of-sample val/test fit, model metrics, EDA panels (distribution, articles/day, GDELT tone, FSI components, Optuna trials, loss curves) |
| **Predicciones** | Nowcast stress gauge with 95% CI, prediction history fan chart, date selector + article viewer |
| **Comparar Modelos** | Multi-model overlay chart comparing all Optuna finalist predictions |

## Repository structure

```
bapro_stress/
├── config.py                        # Centralised constants and paths
├── db/
│   ├── connection.py                # get_engine() — SQLite auto-schema
│   ├── schema.sql                   # PostgreSQL schema
│   ├── migrate_ml03.py              # Migration: horizon, models, training_loss tables
│   ├── migrate_ml04.py              # Migration: trial_number column
│   └── migrate_ofr.py               # Migration: OFR FSI columns in fsi_components
├── src/
│   ├── data/
│   │   ├── build_fsi_target.py      # Compute FSI via yfinance + OFR PCA → writes to DB
│   │   └── update_fsi_daily.py      # Rebuild FSI up to today (called by daily pipeline)
│   ├── ingestion/
│   │   ├── gdelt_ingest.py          # GDELT DOC 2.0 fetcher
│   │   ├── rss_scraper.py           # RSS scraper (Ambito/Cronista/Infobae)
│   │   ├── daily_pipeline.py        # Daily ingest + FSI refresh + nowcast orchestrator
│   │   └── historical_backfill.py   # One-time historical article download
│   └── utils/
│       └── log.py                   # Centralised structlog setup (get_logger)
├── training/
│   ├── embed.py                     # Encode articles → article_embeddings
│   └── train.py                     # TiDE training + Optuna search
├── prediction/
│   └── predict.py                   # CLI: score FSI for a given date
├── dashboard/
│   └── app.py                       # Dash application (3 tabs)
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
| `fsi_components` | 13 normalised component values (4 AR + 9 OFR) |
| `models` | Optuna finalist model metadata, metrics, artifact path |
| `training_predictions` | Out-of-sample val/test predictions per finalist trial |
| `training_loss` | Epoch-level train/val loss per finalist trial |
| `optuna_trials` | Hyperparameter search trial history |
| `daily_predictions` | Nowcast scores from daily pipeline |

## FSI components

The FSI is the first principal component of 12 inputs:

| Component | Direction | Proxy |
|-----------|-----------|-------|
| ^MERV 30d rolling volatility | + | Local equity stress |
| ARGT sovereign bond ETF | − | Sovereign spread |
| ARS=X (USD/ARS FX rate) | + | Currency pressure |
| EMB EM bond ETF | − | External contagion |
| OFR Credit | + | Global credit stress |
| OFR Equity valuation | + | Global equity stress |
| OFR Safe assets | + | Flight-to-safety pressure |
| OFR Funding | + | Global funding stress |
| OFR Volatility | + | Global volatility |
| OFR United States | + | US financial stress |
| OFR Other advanced economies | + | DM stress spillover |
| OFR Emerging markets | + | EM stress context |

OFR FSI composite is stored in `fsi_components` as reference but excluded from the PCA to avoid multicollinearity. Sign validated against PASO 2019 (Aug 12, 2019 stress peak).

## ML details

- **Model**: TiDE (Temporal Identity Encoder) — Darts 0.41.0
- **Alignment**: nowcasting — news at date t predicts FSI at date t (not t+1)
- **Input**: `input_chunk_length` business days look-back (tuned by Optuna); 1 day ahead output
- **Gap nowcasting**: `n = business days from last published FSI to target date`; uses real article embeddings for gap days as `future_covariates`
- **Lag offset**: val/test evaluation starts at `split_boundary + input_chunk_length` to avoid look-back window leaking training observations into held-out sets
- **Covariates**: mean-pooled 384-dim article embeddings (past + future); training uses only days with real embeddings — no zero-vector fabrication
- **Data integrity**: `training/train.py` fails fast with an error log listing specific dates if any business day has no articles, so ingestion gaps are caught before training
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

## Daily automation

`.github/workflows/daily_pipeline.yml` runs Mon-Fri at 06:00 UTC:

1. Refreshes FSI with latest market data (`update_fsi_daily`)
2. Ingests GDELT + RSS for the previous business day
3. Runs TiDE nowcast → writes to `daily_predictions`

> **Note**: the model artifact (`artifacts/tide_model.pt`) is not available in the ephemeral GitHub Actions runner. The workflow requires the artifact to be present in the Docker volume — run the pipeline locally with Docker for full end-to-end execution.

## Requirements

- Docker + Docker Compose
- Python 3.11 (local dev)

See `requirements.txt` for Python dependencies.
