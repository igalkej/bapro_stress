"""
CLI predictor — Darts TiDE model.

Loads the last input_chunk_length days of FSI + article embeddings from the
DB as temporal context and calls model.predict(n=1).

Usage:
    python prediction/predict.py
    python prediction/predict.py --date 2026-02-24
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import TIDE_MODEL_PATH
from db.connection import get_engine


def load_context(engine, target_date: str, n_steps: int):
    """
    Return the last n_steps business days of FSI values + mean-pooled
    article embeddings up to and including target_date.
    """
    with engine.connect() as conn:
        fsi_rows = conn.execute(
            text(
                """
                SELECT date, fsi_value FROM fsi_target
                WHERE date <= :target_date
                ORDER BY date DESC
                LIMIT :n
                """
            ),
            {"target_date": target_date, "n": n_steps},
        ).fetchall()

        if not fsi_rows:
            raise RuntimeError("No FSI data found in DB. Run build_fsi_target.py first.")

        fsi_df = pd.DataFrame(fsi_rows, columns=["date", "fsi_value"])
        fsi_df = fsi_df.sort_values("date").reset_index(drop=True)

        date_embeddings = {}
        for d in fsi_df["date"].tolist():
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
            if not rows:
                continue
            vecs = []
            for row in rows:
                emb_raw = row[0]
                vecs.append(json.loads(emb_raw) if isinstance(emb_raw, str) else list(emb_raw))
            date_embeddings[str(d)[:10]] = np.mean(vecs, axis=0).tolist()

    fsi_df["date"] = fsi_df["date"].astype(str).str[:10]
    fsi_df["embedding"] = fsi_df["date"].map(date_embeddings)
    fsi_df = fsi_df.dropna(subset=["embedding"])
    return fsi_df


def _load_model():
    import torch
    from darts.models import TiDEModel
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    try:
        model = TiDEModel.load(TIDE_MODEL_PATH)
    finally:
        torch.load = _orig
    return model


def run_predict(target_date: str) -> float:
    from darts import TimeSeries

    if not Path(TIDE_MODEL_PATH).exists():
        print(f"Model not found at {TIDE_MODEL_PATH}. Run training/train.py first.")
        sys.exit(1)

    model = _load_model()
    input_chunk = model.input_chunk_length

    engine = get_engine()
    ctx = load_context(engine, target_date, input_chunk)

    if len(ctx) < input_chunk:
        print(f"Need at least {input_chunk} days of context, found {len(ctx)}.")
        sys.exit(1)

    ctx = ctx.tail(input_chunk).reset_index(drop=True)

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
    emb_df = pd.DataFrame(emb_array, columns=[f"emb_{i}" for i in range(emb_dim)])
    emb_df["date"] = ctx["date"].values

    # Extend by 1 extra business day (zero vector) for future_covariates
    last_date = pd.Timestamp(ctx["date"].iloc[-1])
    next_bday = last_date + pd.offsets.BusinessDay(1)
    extra = pd.DataFrame(
        [np.zeros(emb_dim, dtype=np.float32)],
        columns=[f"emb_{i}" for i in range(emb_dim)],
    )
    extra["date"] = next_bday
    emb_df = pd.concat([emb_df, extra], ignore_index=True)

    cov_series = TimeSeries.from_dataframe(
        emb_df,
        time_col="date",
        value_cols=[f"emb_{i}" for i in range(emb_dim)],
        freq="B",
        fill_missing_dates=True,
        fillna_value=0.0,
    )

    pred = model.predict(
        n=1,
        series=fsi_series,
        past_covariates=cov_series,
        future_covariates=cov_series,
    )
    return float(pred.values()[0, 0])


def main():
    parser = argparse.ArgumentParser(description="Score FSI for a target date using TiDE.")
    parser.add_argument(
        "--date",
        default=pd.Timestamp.now().strftime("%Y-%m-%d"),
        help="Target date YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()

    score = run_predict(args.date)
    print(f"FSI stress score for {args.date}: {score:.4f}")

    if score >= 2.5:
        label = "ALTO ESTRES"
    elif score >= 1.0:
        label = "ESTRES ELEVADO"
    elif score >= -1.0:
        label = "NEUTRAL"
    else:
        label = "CALMA"
    print(f"Nivel: {label}")


if __name__ == "__main__":
    main()
