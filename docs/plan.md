# BAPRO Financial Stress System

Argentina sovereign credit stress prediction pipeline.

## Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Database | PostgreSQL 16 (Docker) / SQLite (local dev) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384 dims) |
| Forecasting | Darts TiDE (Temporal Identity Encoder) |
| Hyperparameter search | Optuna |
| Dashboard | Dash + Plotly on port 8050 |
| Containerisation | Docker Compose |
| CI/CD | GitHub Actions (daily cron Mon-Fri 06:00 UTC) |

## Architecture

```
News sources (GDELT DOC 2.0 + RSS)
        │
        ▼
src/ingestion/  ──► articles + article_embeddings tables
        │
        ▼
src/data/build_fsi_target.py  ──► fsi_target table (PCA of market data)
        │
        ▼
training/train.py (TiDE + Optuna)  ──► artifacts/tide_model.pt
                                    ──► training_predictions table
        │
        ▼
src/ingestion/daily_pipeline.py  ──► daily_predictions table
        │
        ▼
dashboard/app.py  ──► http://localhost:8050
```

## FSI Ground Truth

Built from Argentine market data via yfinance + PCA:

| Component | Direction | Proxy |
|-----------|-----------|-------|
| ^MERV volatility | + | Local equity stress |
| ARGT (sovereign bond ETF) | − | Sovereign spread |
| ARS=X (FX rate) | + | Currency stress |
| EMB (EM bond ETF) | − | External contagion |

Sign validated vs PASO 2019 (Aug 12, 2019 stress peak).

## ML Model

- **TiDE** (Temporal Identity Encoder) from Darts 0.41.0
- Input chunk: 2 business days look-back
- Output chunk: 1 business day ahead
- Covariates: 384-dim mean-pooled article embeddings per day (past + future)
- Days without articles: zero vector
- Hyperparameters: tuned with Optuna (stored in `optuna_trials` table)
- Train/val/test split: 70/15/15% dynamic
- Model artifact: `artifacts/tide_model.pt`

## Database Schema (active tables)

| Table | Contents |
|-------|---------|
| `articles` | Headline, source, URL, GDELT tone per article |
| `article_embeddings` | 384-dim embedding per article |
| `fsi_target` | Daily FSI value (PCA score) |
| `fsi_components` | Raw normalised component values |
| `training_predictions` | Val/test out-of-sample TiDE predictions |
| `daily_predictions` | Forward predictions from daily pipeline |
| `optuna_trials` | Hyperparameter search results |

## Running with Docker (standard)

```bash
# 1. Build and start database
docker compose build && docker compose up -d postgres

# 2. Build FSI target from market data
docker compose run --rm app python src/data/build_fsi_target.py \
    --start 2025-11-01 --end 2026-02-24

# 3. Seed FSI into DB
docker compose run --rm app python db/seed_fsi.py

# 4. Ingest historical articles
docker compose run --rm app python src/ingestion/historical_backfill.py \
    --date-from 2025-11-01 --date-to 2026-02-24

# 5. Embed articles
docker compose run --rm app python training/embed.py

# 6. Train model (with Optuna search)
docker compose run --rm app python training/train.py

# 7. Run daily prediction
docker compose run --rm app python src/ingestion/daily_pipeline.py

# 8. Start dashboard
docker compose up -d dashboard
# → http://localhost:8050
```

## Running locally (no Docker)

```bash
.braprostress_venv/Scripts/python.exe src/data/build_fsi_target.py --start ... --end ...
.braprostress_venv/Scripts/python.exe db/seed_fsi.py
.braprostress_venv/Scripts/python.exe src/ingestion/historical_backfill.py --date-from ... --date-to ...
.braprostress_venv/Scripts/python.exe training/embed.py
.braprostress_venv/Scripts/python.exe training/train.py
.braprostress_venv/Scripts/python.exe src/ingestion/daily_pipeline.py
.braprostress_venv/Scripts/python.exe dashboard/app.py
```

## Daily automation

`.github/workflows/daily_pipeline.yml` runs Mon-Fri at 06:00 UTC:

1. Ingest GDELT + RSS for the previous business day
2. Embed new articles
3. Run TiDE prediction → write to `daily_predictions`

## Key conventions

- `config.py` is the single source of truth for constants and paths.
- SQLite vs PostgreSQL differences handled in code (vectors, upserts, deletes).
- Print statements must use ASCII only (Windows terminal is cp1252).
- Never commit: `*.db`, `artifacts/*.pt`, `.braprostress_venv/`
- Run app commands via `docker compose exec` or `docker compose run --rm app`.
