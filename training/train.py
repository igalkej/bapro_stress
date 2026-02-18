"""
Training step — Darts TiDE model.

Joins documents + embeddings + stress_index on date.
Splits 14 train / 4 val / 4 test (temporal order).
Trains TiDE with 384-dim document embeddings as past covariates.
Saves artifacts/tide_model/ + artifacts/metadata.json.
Writes all-sample predictions back to the predictions table.

Usage:
    python training/train.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import get_engine
from config import ARTIFACTS_DIR, TIDE_MODEL_PATH

# Lazy imports — darts is large
from darts import TimeSeries
from darts.models import TiDEModel
from darts.metrics import mae, rmse

# ---------------------------------------------------------------------------
# Split sizes
# ---------------------------------------------------------------------------
TRAIN_SIZE = 14
VAL_SIZE = 4
TEST_SIZE = 4
INPUT_CHUNK = 5
OUTPUT_CHUNK = 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    d.id        AS doc_id,
                    d.doc_date,
                    e.embedding_vector AS vec,
                    s.stress_value
                FROM documents d
                JOIN embeddings   e ON e.doc_id     = d.id
                JOIN stress_index s ON s.index_date = d.doc_date
                ORDER BY d.doc_date
                """
            )
        ).fetchall()

    if not rows:
        raise RuntimeError(
            "No joined rows found. Run db/seed.py and training/embed.py first."
        )

    records = []
    for row in rows:
        vec_raw = row.vec
        vec = np.array(json.loads(vec_raw) if isinstance(vec_raw, str) else vec_raw,
                       dtype=np.float32)
        records.append({
            "doc_id": row.doc_id,
            "doc_date": str(row.doc_date),
            "vec": vec,
            "stress_value": float(row.stress_value),
        })

    df = pd.DataFrame(records)
    df["doc_date"] = pd.to_datetime(df["doc_date"])
    df = df.sort_values("doc_date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# TimeSeries builders
# ---------------------------------------------------------------------------

def build_target_series(df):
    return TimeSeries.from_dataframe(
        df, time_col="doc_date", value_cols="stress_value", freq=None
    )


def build_covariate_series(df):
    n = len(df)
    dim = len(df.iloc[0]["vec"])
    mat = np.stack(df["vec"].values).astype(np.float32)   # (n, 384)
    cov_df = pd.DataFrame(
        mat,
        index=df["doc_date"].values,
        columns=[f"emb_{i}" for i in range(dim)],
    )
    cov_df.index = pd.DatetimeIndex(cov_df.index)
    return TimeSeries.from_dataframe(cov_df, freq=None)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_split(model, target, covariates, start, name):
    """Run historical forecasts over a slice and return MAE and RMSE."""
    preds = model.historical_forecasts(
        series=target,
        past_covariates=covariates,
        start=start,
        forecast_horizon=OUTPUT_CHUNK,
        stride=1,
        retrain=False,
        verbose=False,
    )
    actual = target.slice_intersect(preds)
    split_mae = float(mae(actual, preds))
    split_rmse = float(rmse(actual, preds))
    print(f"  {name}: MAE={split_mae:.4f}  RMSE={split_rmse:.4f}")
    return split_mae, split_rmse, preds


# ---------------------------------------------------------------------------
# Predictions table writer
# ---------------------------------------------------------------------------

def write_predictions(engine, df, preds_series, model_version):
    doc_ids = df["doc_id"].tolist()
    placeholders = ",".join(f":id_{i}" for i in range(len(doc_ids)))
    params = {f"id_{i}": int(doc_ids[i]) for i in range(len(doc_ids))}

    preds_df = preds_series.to_dataframe().reset_index()
    preds_df.columns = ["time", "stress_score_pred"]
    preds_df["time"] = pd.to_datetime(preds_df["time"])

    date_to_docid = {
        pd.Timestamp(row.doc_date): int(row.doc_id)
        for _, row in df.iterrows()
    }

    with engine.begin() as conn:
        conn.execute(
            text(f"DELETE FROM predictions WHERE doc_id IN ({placeholders})"),
            params,
        )
        for _, row in preds_df.iterrows():
            doc_id = date_to_docid.get(row["time"])
            if doc_id is None:
                continue
            conn.execute(
                text(
                    """
                    INSERT INTO predictions (doc_id, stress_score_pred, model_version)
                    VALUES (:doc_id, :pred, :version)
                    """
                ),
                {"doc_id": doc_id, "pred": float(row["stress_score_pred"]),
                 "version": model_version},
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    engine = get_engine()
    print("Loading dataset...")
    df = load_dataset(engine)
    n = len(df)
    print(f"Loaded {n} samples")

    if n < TRAIN_SIZE + VAL_SIZE + TEST_SIZE:
        raise RuntimeError(
            f"Need at least {TRAIN_SIZE + VAL_SIZE + TEST_SIZE} samples, got {n}."
        )

    target = build_target_series(df)
    covariates = build_covariate_series(df)

    # Temporal split by position
    target_train = target[:TRAIN_SIZE]
    target_val = target[:TRAIN_SIZE + VAL_SIZE]      # TiDE needs full context
    target_test = target                              # full series for test eval
    cov_train = covariates[:TRAIN_SIZE]
    cov_full = covariates

    print(f"Split: {TRAIN_SIZE} train / {VAL_SIZE} val / {TEST_SIZE} test")

    # Build model
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    model = TiDEModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        num_encoder_layers=1,
        num_decoder_layers=1,
        decoder_output_dim=8,
        hidden_size=32,
        temporal_width_past=4,
        temporal_width_future=4,
        use_layer_norm=True,
        dropout=0.1,
        n_epochs=500,
        batch_size=4,
        optimizer_kwargs={"lr": 5e-4},
        model_name="tide_bapro",
        work_dir=ARTIFACTS_DIR,
        save_checkpoints=True,
        force_reset=True,
        pl_trainer_kwargs={"enable_progress_bar": True},
    )

    print("Training TiDE model...")
    model.fit(
        series=target_train,
        past_covariates=cov_train,
        val_series=target_val,
        val_past_covariates=cov_full,
        verbose=True,
    )

    # Save model
    model.save(TIDE_MODEL_PATH)
    print(f"Model saved: {TIDE_MODEL_PATH}")

    # Evaluate all splits using historical forecasts
    print("Evaluating splits...")

    # Training window (start after enough context)
    train_start = INPUT_CHUNK  # first index we can forecast from
    train_mae_val, train_rmse_val, _ = evaluate_split(
        model, target_train, cov_train[:TRAIN_SIZE], train_start, "Train"
    )

    val_mae_val, val_rmse_val, _ = evaluate_split(
        model, target_val, cov_full[:TRAIN_SIZE + VAL_SIZE],
        TRAIN_SIZE, "Val"
    )

    test_mae_val, test_rmse_val, all_preds = evaluate_split(
        model, target_test, cov_full,
        TRAIN_SIZE + VAL_SIZE, "Test"
    )

    # Full-series historical forecasts for DB predictions
    print("Generating full-series predictions for DB...")
    full_preds = model.historical_forecasts(
        series=target,
        past_covariates=cov_full,
        start=INPUT_CHUNK,
        forecast_horizon=OUTPUT_CHUNK,
        stride=1,
        retrain=False,
        verbose=False,
    )

    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Build metadata
    tide_params = {
        "input_chunk_length": INPUT_CHUNK,
        "output_chunk_length": OUTPUT_CHUNK,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "decoder_output_dim": 8,
        "hidden_size": 32,
        "temporal_width_past": 4,
        "temporal_width_future": 4,
        "use_layer_norm": True,
        "dropout": 0.1,
        "n_epochs": 500,
        "batch_size": 4,
        "lr": 5e-4,
    }
    metadata = {
        "model_version": model_version,
        "train_samples": TRAIN_SIZE,
        "val_samples": VAL_SIZE,
        "test_samples": TEST_SIZE,
        "train_mae": train_mae_val,
        "train_rmse": train_rmse_val,
        "val_mae": val_mae_val,
        "val_rmse": val_rmse_val,
        "test_mae": test_mae_val,
        "test_rmse": test_rmse_val,
        "tide_params": tide_params,
    }

    meta_path = Path(ARTIFACTS_DIR) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    # Write predictions to DB
    write_predictions(engine, df, full_preds, model_version)
    print(f"Wrote {len(full_preds)} predictions to DB. Version: {model_version}")


if __name__ == "__main__":
    main()
