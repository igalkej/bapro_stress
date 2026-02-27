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
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import EMBEDDING_MODEL, TIDE_MODEL_PATH, FORECAST_HORIZON, EMBEDDING_DIM
from db.connection import get_engine
from src.ingestion.gdelt_ingest import fetch_and_store_gdelt
from src.ingestion.rss_scraper import fetch_and_store_rss
from src.data.update_fsi_daily import update_fsi_daily
from src.utils.log import get_logger

log = get_logger(__name__)


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
    """Return True if a nowcast already exists for target_date."""
    result = conn.execute(
        text("SELECT COUNT(*) FROM daily_predictions WHERE date = :date"),
        {"date": date_str},
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


def _get_last_fsi_date(conn) -> str | None:
    """Return the most recent date in fsi_target, or None if empty."""
    row = conn.execute(text("SELECT MAX(date) FROM fsi_target")).fetchone()
    return str(row[0])[:10] if row and row[0] else None


def _get_gap_embeddings(conn, gap_dates: list[str],
                        is_pg: bool) -> dict[str, list[float]]:
    """
    Return mean-pooled embeddings for each gap date (days with articles but
    no published FSI yet). Falls back to a zero vector if no articles exist.
    """
    result: dict[str, list[float]] = {}
    for d in gap_dates:
        rows = conn.execute(
            text(
                """
                SELECT ae.embedding FROM article_embeddings ae
                JOIN articles a ON a.id = ae.id
                WHERE a.date = :date
                """
            ),
            {"date": d},
        ).fetchall()
        if rows:
            vecs = [
                json.loads(r[0]) if isinstance(r[0], str) else list(r[0])
                for r in rows
            ]
            result[d] = np.mean(vecs, axis=0).tolist()
        else:
            result[d] = [0.0] * EMBEDDING_DIM
    return result


def _run_tide_prediction(
    context_df: pd.DataFrame,
    model_path: str,
    n: int,
    gap_embeds: dict[str, list[float]],
) -> list[tuple[str, float]]:
    """
    Nowcast FSI for `n` gap days using the TiDE model.

    context_df: FSI + embedding rows for the input_chunk days ending at
                last_fsi_date (the last date with known FSI).
    gap_embeds: {date_str: embedding_vector} for each gap day to nowcast.
                Real article embeddings, NOT zero vectors.

    Returns list of (date_str, score) for each nowcasted date.
    """
    from darts import TimeSeries

    assert n >= 1, f"n must be >= 1 for nowcasting, got {n}"

    model = _load_tide_model(model_path)
    input_chunk = model.input_chunk_length

    # Use last input_chunk rows (FSI context ending at last_fsi_date)
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

    # Append gap-day embeddings (real article vectors, not zeros)
    gap_rows = []
    for d in sorted(gap_embeds):
        row = {f"emb_{j}": v for j, v in enumerate(gap_embeds[d])}
        row["date"] = pd.Timestamp(d)
        gap_rows.append(row)
    emb_df = pd.concat([emb_df, pd.DataFrame(gap_rows)], ignore_index=True)

    cov_series = TimeSeries.from_dataframe(
        emb_df,
        time_col="date",
        value_cols=[f"emb_{i}" for i in range(emb_dim)],
        freq="B",
        fill_missing_dates=True,
        fillna_value=0.0,
    )

    pred = model.predict(
        n=n,
        series=fsi_series,
        past_covariates=cov_series,
        future_covariates=cov_series,
    )
    last_ctx_date = pd.Timestamp(ctx["date"].iloc[-1])
    pred_dates = pd.bdate_range(
        start=last_ctx_date + pd.offsets.BusinessDay(1),
        periods=n,
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

    log.info("start... daily_pipeline", ts=datetime.now().strftime("%Y/%m/%d %H:%M"),
             target_date=target_date)

    engine = get_engine()
    is_pg = not engine.url.drivername.startswith("sqlite")

    # Idempotency check
    with engine.connect() as conn:
        if _prediction_exists(conn, target_date):
            log.info("daily_pipeline_skipped", reason="prediction_exists",
                     target_date=target_date)
            return {"date": target_date, "status": "skipped"}

    # Refresh FSI target with latest market data (non-fatal if yfinance is down)
    try:
        fsi_result = update_fsi_daily(target_date)
        log.info("fsi_refreshed", rows=fsi_result["rows"],
                 start=fsi_result["start"], end=fsi_result["end"])
    except Exception as exc:
        log.warning("fsi_refresh_failed", reason=str(exc), note="non-fatal")

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
            log.error("no_articles", date=target_date)
            raise RuntimeError(msg)
        log.info("articles_already_ingested", date=target_date, count=existing)

    # Build prediction context
    model_path = TIDE_MODEL_PATH
    if not Path(model_path).exists():
        log.warning("model_not_found", path=model_path, note="skipping prediction")
        return {
            "date": target_date,
            "gdelt_new": gdelt_new,
            "rss_new": rss_new,
            "stress_score": None,
            "status": "no_model",
        }

    with engine.connect() as conn:
        # Compute the nowcast gap: business days with articles but without FSI
        last_fsi_date = _get_last_fsi_date(conn)
        assert last_fsi_date is not None, "fsi_target is empty — seed FSI data first"

        gap_dates = pd.bdate_range(
            pd.Timestamp(last_fsi_date) + pd.offsets.BusinessDay(1),
            pd.Timestamp(target_date),
        ).strftime("%Y-%m-%d").tolist()

        if not gap_dates:
            log.info("nowcast_no_gap", last_fsi_date=last_fsi_date,
                     target_date=target_date,
                     note="FSI already covers target_date, nothing to nowcast")
            return {
                "date": target_date,
                "gdelt_new": gdelt_new,
                "rss_new": rss_new,
                "stress_score": None,
                "status": "no_gap",
            }

        n = len(gap_dates)
        log.info("nowcast_gap", n=n, gap_from=gap_dates[0], gap_to=gap_dates[-1])

        model = _load_tide_model(model_path)
        input_chunk = model.input_chunk_length
        model_version = Path(model_path).stem

        # Context: FSI + embeddings for input_chunk days ending at last_fsi_date
        context_df, _ = _get_context_for_prediction(
            conn, last_fsi_date, input_chunk, is_pg
        )

        # Gap embeddings: real article vectors for each day being nowcasted
        gap_embeds = _get_gap_embeddings(conn, gap_dates, is_pg)

    if context_df is None or len(context_df) < input_chunk:
        msg = f"Not enough context data for nowcasting (need {input_chunk} days)"
        log.error("insufficient_context", need=input_chunk,
                  found=len(context_df) if context_df is not None else 0)
        raise RuntimeError(msg)

    forecasts = _run_tide_prediction(context_df, model_path, n, gap_embeds)
    for pred_date, score in forecasts:
        log.info("nowcast_score", date=pred_date, score=round(score, 4))

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
    log.info("finish... daily_pipeline", ts=datetime.now().strftime("%Y/%m/%d %H:%M"),
             target_date=result["date"], status=result.get("status"),
             gdelt_new=result.get("gdelt_new"), rss_new=result.get("rss_new"),
             stress_score=result.get("stress_score"))
    return result


def main():
    parser = argparse.ArgumentParser(description="Daily BAPRO pipeline")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    result = run_daily(args.date)
    log.info("daily_pipeline_main_done", **{k: v for k, v in result.items()
                                            if k != "forecasts"})


if __name__ == "__main__":
    main()
