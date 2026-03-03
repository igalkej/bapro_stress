# BAPRO Financial Stress System

Argentina sovereign credit stress nowcasting pipeline.

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
src/data/build_fsi_target.py  ──► fsi_target + fsi_components tables
        │                         (yfinance market data + OFR FSI → PCA)
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

Built from 12 inputs via PCA (first principal component):

### Argentine market series (via yfinance)

| Component | Direction | Proxy |
|-----------|-----------|-------|
| ^MERV 30d rolling volatility | + | Local equity stress |
| ARGT sovereign bond ETF | − | Sovereign spread |
| ARS=X (USD/ARS FX rate) | + | Currency pressure |
| EMB EM bond ETF | − | External contagion |

### OFR Financial Stress Index sub-indicators (via financialresearch.gov)

| Component | Direction | Proxy |
|-----------|-----------|-------|
| OFR Credit | + | Global credit stress |
| OFR Equity valuation | + | Global equity stress |
| OFR Safe assets | + | Flight-to-safety pressure |
| OFR Funding | + | Global funding stress |
| OFR Volatility | + | Global volatility |
| OFR United States | + | US financial stress |
| OFR Other advanced economies | + | DM stress spillover |
| OFR Emerging markets | + | EM stress context |

The OFR FSI composite is stored in `fsi_components` as reference but excluded from the PCA to avoid multicollinearity. All 13 component values (4 AR + 9 OFR) are z-scored and stored in `fsi_components`.

Sign validated against PASO 2019 (Aug 12, 2019 stress peak): FSI value on that date must be in the top 10th percentile; sign is flipped otherwise.

## ML Model

- **TiDE** (Temporal Identity Encoder) from Darts 0.41.0
- **Alignment**: nowcasting — news@t predicts FSI@t (not t+1)
- Input chunk: tuned by Optuna (candidates: 2, 5, up to min(10, TRAIN_SIZE-2)); output chunk: 1 business day ahead
- Lag offset: `historical_forecasts` start at `split_boundary + input_chunk_length` for both val and test, so the look-back window never includes observations from the other split (no boundary leakage)
- Optuna guard: trials where `input_chunk_length >= VAL_SIZE` or `>= TEST_SIZE` are pruned to avoid out-of-bounds errors
- Covariates: mean-pooled 384-dim article embeddings per day (past + future); zero vector for days without articles
- Train/val/test split: 70/15/15% dynamic
- Model artifact: `artifacts/tide_model.pt`

## Database Schema

| Table | Contents |
|-------|---------|
| `articles` | Headline, source, URL, GDELT tone per article |
| `article_embeddings` | 384-dim embedding per article |
| `fsi_target` | Daily FSI value (PCA score) |
| `fsi_components` | 13 normalised component values (4 AR + 9 OFR) |
| `models` | Optuna finalist model metadata, metrics, artifact path |
| `training_predictions` | Val/test out-of-sample TiDE predictions per finalist |
| `training_loss` | Epoch-level train/val loss per finalist trial |
| `optuna_trials` | Hyperparameter search results |
| `daily_predictions` | Nowcast scores from daily pipeline |

## Running with Docker (standard)

```bash
# 1. Build and start database
docker compose build && docker compose up -d postgres

# 2. (First run only) Run DB migrations
docker compose run --rm app python db/migrate_ml03.py
docker compose run --rm app python db/migrate_ml04.py
docker compose run --rm app python db/migrate_ofr.py

# 3. Build FSI target — downloads yfinance market data + OFR FSI, runs PCA, writes to DB
docker compose run --rm app python src/data/build_fsi_target.py \
    --start 2000-01-03 --end 2026-03-03

# 4. Ingest historical articles
docker compose run --rm app python src/ingestion/historical_backfill.py \
    --date-from 2000-01-03 --date-to 2026-03-03

# 5. Embed articles
docker compose run --rm app python training/embed.py

# 6. Train model (with Optuna search)
docker compose run --rm app python training/train.py

# 7. Run daily prediction
docker compose run --rm app python src/ingestion/daily_pipeline.py

# 8. Start dashboard
docker compose up -d dashboard
# → http://localhost:8050

# pgAdmin (DB inspector) → http://localhost:5050 (admin@bapro.com / admin)
```

## Running locally (no Docker)

Uses SQLite at `<repo>/bapro_stress.db` (auto-created on first run).

```bash
.braprostress_venv/Scripts/python.exe db/migrate_ofr.py
.braprostress_venv/Scripts/python.exe src/data/build_fsi_target.py --start ... --end ...
.braprostress_venv/Scripts/python.exe src/ingestion/historical_backfill.py --date-from ... --date-to ...
.braprostress_venv/Scripts/python.exe training/embed.py
.braprostress_venv/Scripts/python.exe training/train.py
.braprostress_venv/Scripts/python.exe src/ingestion/daily_pipeline.py
.braprostress_venv/Scripts/python.exe dashboard/app.py
```

## Daily automation

`.github/workflows/daily_pipeline.yml` runs Mon-Fri at 06:00 UTC:

1. Refreshes FSI with latest market data (`update_fsi_daily`)
2. Ingests GDELT + RSS for the previous business day
3. Runs TiDE nowcast → writes to `daily_predictions`

The workflow can also be triggered manually with an optional `target_date` input.

> **Note**: the model artifact (`artifacts/tide_model.pt`) lives in the local Docker volume and is not available in the ephemeral GitHub Actions runner. Run the pipeline locally with Docker for full end-to-end execution including training and prediction.

## Dashboard

| Tab | Contents |
|-----|---------|
| **Entrenamiento** | FSI time series bounded to training period (train+val+test) + out-of-sample val/test fit curves (with lag-offset start) + model metrics table + EDA sub-tabs (FSI distribution, articles/day, GDELT tone correlation, FSI components, Optuna trials, loss curves) |
| **Predicciones** | FSI history fan chart with range selector + date dropdown + click-on-chart → news article viewer panel + daily stress score gauge with 95% CI |
| **Comparar Modelos** | Multi-model overlay chart comparing all Optuna finalist predictions (val + test) |

## Key conventions

- `config.py` is the single source of truth for constants and paths.
- SQLite vs PostgreSQL differences handled in code (vectors, upserts, deletes).
- Print statements must use ASCII only (Windows terminal is cp1252).
- Logging via `structlog` (`src/utils/log.py`); long tasks use `start...`/`finish...` bracketing with `ts=` timestamp.
- Never commit: `*.db`, `artifacts/*.pt`, `.braprostress_venv/`
- Run app commands via `docker compose exec` or `docker compose run --rm app`.
- GDELT query uses OR syntax: `(economia OR finanzas OR ...)` — parenthesised AND is rejected by the API.
- yfinance: GD30.BA delisted → falls back to ARGT automatically.
