"""
CLI predictor — Darts TiDE model.

Loads the last input_chunk_length data points from the DB as temporal context,
encodes the new document, and calls model.predict(n=1).

Usage:
    python prediction/predict.py --text "Global markets sold off..."
    python prediction/predict.py --file /path/to/document.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import EMBEDDING_MODEL, TIDE_MODEL_PATH
from db.connection import get_engine


def load_context(engine, n_steps):
    """Return the last n_steps (stress_value, embedding) rows from DB."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT d.doc_date, e.embedding_vector, s.stress_value
                FROM documents d
                JOIN embeddings   e ON e.doc_id     = d.id
                JOIN stress_index s ON s.index_date = d.doc_date
                ORDER BY d.doc_date DESC
                LIMIT :n
                """
            ),
            {"n": n_steps},
        ).fetchall()

    if not rows:
        raise RuntimeError("No data found in DB. Run the full pipeline first.")

    rows = list(reversed(rows))   # oldest → newest
    dates = [pd.Timestamp(str(r.doc_date)) for r in rows]
    stress_vals = [float(r.stress_value) for r in rows]
    embeddings = []
    for r in rows:
        vec_raw = r.embedding_vector
        vec = json.loads(vec_raw) if isinstance(vec_raw, str) else list(vec_raw)
        embeddings.append(vec)
    return dates, stress_vals, embeddings


def build_predict_series(dates, stress_vals, embeddings, new_embedding):
    """
    Build target + past_covariates TimeSeries for model.predict().

    For TiDE with output_chunk_length=1, past_covariates must cover
    [context_start, prediction_time] inclusive, so we append the new
    document embedding as the covariate for the next time step.
    """
    from darts import TimeSeries

    # Infer a 1-day frequency offset from the last two dates
    if len(dates) >= 2:
        delta = dates[-1] - dates[-2]
    else:
        delta = pd.Timedelta(days=1)
    next_date = dates[-1] + delta

    target_df = pd.DataFrame({"stress_value": stress_vals}, index=pd.DatetimeIndex(dates))
    target_series = TimeSeries.from_dataframe(target_df, freq=None)

    dim = len(embeddings[0])
    all_embeddings = embeddings + [new_embedding]
    all_dates = dates + [next_date]
    cov_df = pd.DataFrame(
        np.array(all_embeddings, dtype=np.float32),
        index=pd.DatetimeIndex(all_dates),
        columns=[f"emb_{i}" for i in range(dim)],
    )
    cov_series = TimeSeries.from_dataframe(cov_df, freq=None)

    return target_series, cov_series


def main():
    parser = argparse.ArgumentParser(description="Score a financial document with TiDE.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Document text as a quoted string.")
    group.add_argument("--file", help="Path to a .txt document file.")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text_input = f.read()
    else:
        text_input = args.text

    # Load TiDE model
    from darts.models import TiDEModel
    if not Path(TIDE_MODEL_PATH).exists():
        print(f"Model not found at {TIDE_MODEL_PATH}. Run training/train.py first.")
        sys.exit(1)

    print("Loading TiDE model...")
    model = TiDEModel.load(TIDE_MODEL_PATH)
    n_context = model.input_chunk_length

    # Load DB context
    engine = get_engine()
    dates, stress_vals, embeddings = load_context(engine, n_context)

    if len(dates) < n_context:
        print(f"Need at least {n_context} data points in DB, found {len(dates)}.")
        sys.exit(1)

    # Encode new document
    print("Encoding document...")
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(EMBEDDING_MODEL)
    new_emb = st_model.encode([text_input], convert_to_numpy=True)[0].tolist()

    # Build TimeSeries and predict
    target_series, cov_series = build_predict_series(dates, stress_vals, embeddings, new_emb)
    pred = model.predict(n=1, series=target_series, past_covariates=cov_series)
    score = float(pred.values()[0][0])

    print(f"Stress score: {score:.4f}")

    if score >= 2.5:
        label = "HIGH STRESS"
    elif score >= 1.0:
        label = "ELEVATED STRESS"
    elif score >= -1.0:
        label = "NEUTRAL"
    else:
        label = "CALM"
    print(f"Level: {label}")


if __name__ == "__main__":
    main()
