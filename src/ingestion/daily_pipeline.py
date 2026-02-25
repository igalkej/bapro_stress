"""
Daily ingestion and prediction pipeline.

Ingests GDELT + RSS articles for a target date, then runs the TiDE model
to produce a stress score for that date. Idempotent: if a prediction already
exists for the target date, the run is skipped.

Default target date: yesterday (most recent business day).

Usage:
    python src/ingestion/daily_pipeline.py
    python src/ingestion/daily_pipeline.py --date 2024-01-15
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import EMBEDDING_MODEL, TIDE_MODEL_PATH, FORECAST_HORIZON
from db.connection import get_engine
from src.ingestion.gdelt_ingest import fetch_and_store_gdelt
from src.ingestion.rss_scraper import fetch_and_store_rss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _last_business_day() -> str:
    """Return yesterday's date, rolling back to Friday if today is Monday."""
    today = pd.Timestamp.now().normalize()
    offset = pd.offsets.BusinessDay(1)
    last_bday = (today - offset)
    return last_bday.strftime("%Y-%m-%d")


def _first_forecast_date(context_date: str) -> str:
    """Return the first business day after context_date (= t+1 prediction date)."""
    return (pd.Timestamp(context_date) + pd.offsets.BusinessDay(1)).strftime("%Y-%m-%d")


def _prediction_exists(conn, date_str: str) -> bool:
    """Return True if a daily prediction row already exists for the first forecast date."""
    first_pred_date = _first_forecast_date(date_str)
    result = conn.execute(
        text("SELECT COUNT(*) FROM daily_predictions WHERE date = :date"),
        {"date": first_pred_date},
    ).scalar()
    return (result or 0) > 0


def _get_context_for_prediction(conn, target_date: str, input_chunk: int,
                                 is_pg: bool) -> tuple[pd.DataFrame | None, list[int]]:
    """
    Fetch the last `input_chunk` business days of fsi_target + mean-pooled
    article embeddings up to and including target_date.

    Returns (context_df, target_article_ids).
    context_df columns: date, fsi_value, embedding (list of floats)
    """
    # Get fsi_target rows up to and including target_date
    fsi_rows = conn.execute(
        text(
            """
            SELECT date, fsi_value FROM fsi_target
            WHERE date <= :target_date
            ORDER BY date DESC
            LIMIT :n
            """
        ),
        {"target_date": target_date, "n": input_chunk},
    ).fetchall()

    if not fsi_rows:
        return None, []

    fsi_df = pd.DataFrame(fsi_rows, columns=["date", "fsi_value"])
    fsi_df = fsi_df.sort_values("date").reset_index(drop=True)

    # Get mean-pooled embeddings per day for all days in fsi_df
    dates_in_context = fsi_df["date"].tolist()
    date_embeddings = {}
    target_article_ids = []

    for d in dates_in_context:
        rows = conn.execute(
            text(
                """
                SELECT ae.embedding, a.id
                FROM article_embeddings ae
                JOIN articles a ON a.id = ae.id
                WHERE a.date = :date
                """
            ),
            {"date": d},
        ).fetchall()

        if not rows:
            continue

        vecs = []
        for row in rows:
            emb_raw = row[0]
            if isinstance(emb_raw, str):
                vec = json.loads(emb_raw)
            else:
                vec = list(emb_raw)
            vecs.append(vec)
            if d == target_date:
                target_article_ids.append(row[1])

        date_embeddings[d] = np.mean(vecs, axis=0).tolist()

    if not date_embeddings:
        return None, []

    fsi_df["embedding"] = fsi_df["date"].map(date_embeddings)
    fsi_df = fsi_df.dropna(subset=["embedding"])

    return fsi_df, target_article_ids


def _load_tide_model(model_path: str):
    """Load TiDE model with weights_only=False patch for PyTorch 2.6+."""
    import torch
    from darts.models import TiDEModel
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    try:
        model = TiDEModel.load(model_path)
    finally:
        torch.load = _orig
    return model


def _run_tide_prediction(
    context_df: pd.DataFrame, model_path: str
) -> list[tuple[str, float]]:
    """Load TiDE model and predict FORECAST_HORIZON steps ahead.

    Returns list of (date_str, score) for each forecasted business day.
    """
    from darts import TimeSeries

    model = _load_tide_model(model_path)
    input_chunk = model.input_chunk_length

    # Use last input_chunk rows
    ctx = context_df.tail(input_chunk).reset_index(drop=True)
    ctx["date"] = pd.to_datetime(ctx["date"].astype(str).str[:10])

    fsi_series = TimeSeries.from_dataframe(
        ctx[["date", "fsi_value"]].rename(columns={"fsi_value": "value"}),
        time_col="date",
        value_cols=["value"],
        freq="B",
        fill_missing_dates=True,
        fillna_value=0.0,
    )

    emb_dim = len(ctx["embedding"].iloc[0])
    emb_array = np.array(ctx["embedding"].tolist())
    emb_df = pd.DataFrame(
        emb_array,
        columns=[f"emb_{i}" for i in range(emb_dim)],
    )
    emb_df["date"] = ctx["date"].values

    # Extend covariates by FORECAST_HORIZON extra business days (zero vectors)
    # so that future_covariates cover the full prediction window.
    last_date = pd.Timestamp(ctx["date"].iloc[-1])
    extra_rows = []
    for i in range(1, FORECAST_HORIZON + 1):
        extra_rows.append({
            **{f"emb_{j}": 0.0 for j in range(emb_dim)},
            "date": last_date + pd.offsets.BusinessDay(i),
        })
    emb_df = pd.concat([emb_df, pd.DataFrame(extra_rows)], ignore_index=True)

    cov_series = TimeSeries.from_dataframe(
        emb_df,
        time_col="date",
        value_cols=[f"emb_{i}" for i in range(emb_dim)],
        freq="B",
        fill_missing_dates=True,
        fillna_value=0.0,
    )

    pred = model.predict(
        n=FORECAST_HORIZON,
        series=fsi_series,
        past_covariates=cov_series,
        future_covariates=cov_series,
    )
    pred_dates = pd.bdate_range(
        start=last_date + pd.offsets.BusinessDay(1),
        periods=FORECAST_HORIZON,
    )
    return [
        (d.strftime("%Y-%m-%d"), float(pred.values()[i, 0]))
        for i, d in enumerate(pred_dates)
    ]


def _store_prediction(conn, date_str: str, score: float,
                      model_version: str, is_pg: bool) -> None:
    """Insert a daily prediction row for the target date."""
    if is_pg:
        sql = text(
            """
            INSERT INTO daily_predictions (date, fsi_pred, model_version)
            VALUES (:date, :score, :ver)
            ON CONFLICT (date) DO UPDATE SET
                fsi_pred      = EXCLUDED.fsi_pred,
                model_version = EXCLUDED.model_version,
                predicted_at  = NOW()
            """
        )
    else:
        sql = text(
            """
            INSERT OR REPLACE INTO daily_predictions (date, fsi_pred, model_version)
            VALUES (:date, :score, :ver)
            """
        )
    conn.execute(sql, {"date": date_str, "score": score, "ver": model_version})


def run_daily(target_date: str | None = None) -> dict:
    """
    Full daily pipeline: ingest GDELT + RSS for target_date, then predict.

    Returns dict with keys: date, gdelt_new, rss_new, stress_score, status.
    """
    if target_date is None:
        target_date = _last_business_day()

    log.info("Daily pipeline starting for date: %s", target_date)

    engine = get_engine()
    is_pg = not engine.url.drivername.startswith("sqlite")

    # Idempotency check
    with engine.connect() as conn:
        if _prediction_exists(conn, target_date):
            log.info("Prediction already exists for %s. Skipping.", target_date)
            print(f"Prediction already exists. Skipping.")
            return {"date": target_date, "status": "skipped"}

    # Ingest
    gdelt_new = fetch_and_store_gdelt(target_date, target_date)
    rss_new = fetch_and_store_rss(target_date, target_date)

    if gdelt_new + rss_new == 0:
        # No new articles — check if articles were already ingested on a prior run
        with engine.connect() as conn:
            existing = conn.execute(
                text("SELECT COUNT(*) FROM articles WHERE date = :date"),
                {"date": target_date},
            ).scalar() or 0
        if existing == 0:
            msg = f"No articles found for {target_date}"
            log.error(msg)
            raise RuntimeError(msg)
        log.info("No new articles (all already ingested). Proceeding with %d existing.", existing)

    # Build prediction context
    model_path = TIDE_MODEL_PATH
    if not Path(model_path).exists():
        log.warning("TiDE model not found at %s. Skipping prediction.", model_path)
        return {
            "date": target_date,
            "gdelt_new": gdelt_new,
            "rss_new": rss_new,
            "stress_score": None,
            "status": "no_model",
        }

    with engine.connect() as conn:
        model = _load_tide_model(model_path)
        input_chunk = model.input_chunk_length
        model_version = Path(model_path).stem

        context_df, target_article_ids = _get_context_for_prediction(
            conn, target_date, input_chunk, is_pg
        )

    if context_df is None or len(context_df) < input_chunk:
        msg = f"Not enough context data for prediction (need {input_chunk} days)"
        log.error(msg)
        raise RuntimeError(msg)

    forecasts = _run_tide_prediction(context_df, model_path)
    for pred_date, score in forecasts:
        log.info("Forecast: %s -> %.4f", pred_date, score)

    with engine.begin() as conn:
        for pred_date, score in forecasts:
            _store_prediction(conn, pred_date, score, model_version, is_pg)

    result = {
        "date": target_date,
        "gdelt_new": gdelt_new,
        "rss_new": rss_new,
        "stress_score": forecasts[0][1] if forecasts else None,
        "forecasts": {d: s for d, s in forecasts},
        "status": "ok",
    }
    log.info("Daily pipeline complete: %s", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Daily BAPRO pipeline")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    result = run_daily(args.date)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
