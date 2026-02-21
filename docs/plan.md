
# BAPRO Financial Stress System

Argentina sovereign credit stress prediction pipeline — January 2024.

## Stack
- Python 3.11 venv at `.braprostress_venv/`
- SQLite for local dev, PostgreSQL for Docker
- sentence-transformers (`all-MiniLM-L6-v2`) → Darts TiDE (input_chunk=5, output_chunk=1)
- 384-dim document embeddings used as TiDE past covariates
- Dash dashboard on port 8050

## Running locally (no Docker)

Always use the project venv:
```
.braprostress_venv/Scripts/python.exe <script>
```

Pipeline order (run each step once, they are idempotent):
```
.braprostress_venv/Scripts/python.exe db/seed.py
.braprostress_venv/Scripts/python.exe training/embed.py
.braprostress_venv/Scripts/python.exe training/train.py
.braprostress_venv/Scripts/python.exe dashboard/app.py
```

Dashboard: http://localhost:8050

## Running with Docker

```
docker compose build && docker compose up -d postgres
docker compose run --rm app python db/seed.py
make embed && make train
make dashboard
```

## Key conventions

- `config.py` is the single source of truth for constants and paths.
  - No script should hardcode `/workspace` or absolute paths.
  - Use `Path(__file__).resolve().parent.parent` for repo-relative imports.
- SQLite vs PostgreSQL differences are handled in the code:
  - Vectors: JSON string (SQLite) vs FLOAT[] (PostgreSQL)
  - Upserts: `INSERT OR IGNORE` (SQLite) vs `ON CONFLICT DO NOTHING` (PostgreSQL)
  - Deletes: portable `IN (...)` clause, not `= ANY(:ids)`
- Print statements must use ASCII only (Windows terminal is cp1252).
  No `→`, `…`, `—` or other non-ASCII characters in print() calls.
- Never commit: `*.db`, `artifacts/*.pkl`, `artifacts/*.json`, `.braprostress_venv/`

## Data
- 22 business days in January 2024 (Jan 1 excluded — New Year)
- 3 stress spikes: Jan 5 (~2.91), Jan 11 (~2.64), Jan 17 (~2.83)
- Post-spike calm/recovery: Jan 22–31 (stress values -1.8 to -0.4)
- `data/stress_index.csv` has 22 rows matching the 22 `.txt` documents

## File map
| File | Role |
|------|------|
| `config.py` | Centralised constants and paths |
| `db/connection.py` | `get_engine()` — auto-creates SQLite schema |
| `db/seed.py` | Load docs + stress index into DB |
| `training/embed.py` | Encode docs → store embeddings |
| `training/train.py` | Darts TiDE → save artifacts/tide_model/ + predictions |
| `prediction/predict.py` | CLI: `--text "..."` or `--file path.txt` |
| `dashboard/app.py` | Dash UI: historical chart + live prediction tab |
