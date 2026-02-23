"""
Training step — Darts TiDE model on real FSI data.

Joins articles + article_embeddings + fsi_target on date.
Mean-pools article embeddings per day to produce one 384-dim vector.
Splits 70/15/15 by row count (temporal order preserved).
Trains TiDE with 384-dim embeddings as past covariates.
Saves artifacts/tide_model.pt + artifacts/metadata.json.
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

from darts import TimeSeries
from darts.models import TiDEModel
from darts.metrics import mae, rmse

# ---------------------------------------------------------------------------
# Split fractions
# ---------------------------------------------------------------------------
TRAIN_PCT = 0.70
VAL_PCT   = 0.15
# TEST_PCT  = 0.15  (remainder)

INPUT_CHUNK  = 2
OUTPUT_CHUNK = 1


# ---------------------------------------------------------------------------
# Data loading — real corpus (articles + fsi_target)
# ---------------------------------------------------------------------------

def load_dataset(engine):
    """
    Join articles + article_embeddings + fsi_target on date.
    Mean-pool all article embeddings per business day.

    Returns DataFrame with columns: date (datetime), vec (np.array), fsi_value (float).
    """
    is_pg = not engine.url.drivername.startswith("sqlite")

    with engine.connect() as conn:
        # Get all fsi_target dates
        fsi_rows = conn.execute(
            text("SELECT date, fsi_value FROM fsi_target ORDER BY date")
        ).fetchall()

        # Get all article embeddings with their dates
        emb_rows = conn.execute(
            text(
                """
                SELECT a.date, ae.embedding
                FROM article_embeddings ae
                JOIN articles a ON a.id = ae.id
                ORDER BY a.date
                """
            )
        ).fetchall()

    if not fsi_rows:
        raise RuntimeError("fsi_target table is empty. Run make seed_fsi first.")
    if not emb_rows:
        raise RuntimeError("article_embeddings table is empty. Run make embed first.")

    # Build per-day embedding dict (mean-pool)
    date_vecs: dict[str, list] = {}
    for row in emb_rows:
        date_str = str(row[0])[:10]
        emb_raw = row[1]
        vec = np.array(
            json.loads(emb_raw) if isinstance(emb_raw, str) else list(emb_raw),
            dtype=np.float32,
        )
        if date_str not in date_vecs:
            date_vecs[date_str] = []
        date_vecs[date_str].append(vec)

    mean_vecs: dict[str, np.ndarray] = {
        d: np.mean(vecs, axis=0) for d, vecs in date_vecs.items()
    }

    # Determine embedding dimension from first available vec
    emb_dim = next(iter(mean_vecs.values())).shape[0] if mean_vecs else 384
    zero_vec = np.zeros(emb_dim, dtype=np.float32)

    records = []
    for row in fsi_rows:
        date_str = str(row[0])[:10]
        fsi_val = float(row[1])
        # Use zero vector for days without articles so the target has no gaps
        vec = mean_vecs.get(date_str, zero_vec)
        records.append({"date": date_str, "vec": vec, "fsi_value": fsi_val})

    if not records:
        raise RuntimeError(
            "No overlap between fsi_target and article_embeddings dates. "
            "Run backfill first."
        )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# TimeSeries builders
# ---------------------------------------------------------------------------

def build_target_series(df):
    return TimeSeries.from_dataframe(
        df, time_col="date", value_cols="fsi_value",
        fill_missing_dates=True, freq="B"
    )


def build_covariate_series(df):
    dim = len(df.iloc[0]["vec"])
    mat = np.stack(df["vec"].values).astype(np.float32)
    cov_df = pd.DataFrame(
        mat,
        index=df["date"].values,
        columns=[f"emb_{i}" for i in range(dim)],
    )
    cov_df.index = pd.DatetimeIndex(cov_df.index)
    ts = TimeSeries.from_dataframe(cov_df, fill_missing_dates=True, freq="B")
    # Fill NaN (days without articles) with zero vectors
    filled_df = ts.to_dataframe().fillna(0.0)
    return TimeSeries.from_dataframe(filled_df, freq="B")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_split(model, target, covariates, start, name):
    preds = model.historical_forecasts(
        series=target,
        past_covariates=covariates,
        future_covariates=covariates,
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
# Predictions table writer — updated for articles table
# ---------------------------------------------------------------------------

def write_predictions(engine, df, preds_series, model_version):
    """
    Write predictions to DB keyed by date.
    """
    is_pg = not engine.url.drivername.startswith("sqlite")

    preds_df = preds_series.to_dataframe().reset_index()
    preds_df.columns = ["time", "stress_score_pred"]
    preds_df["time"] = pd.to_datetime(preds_df["time"])
    preds_df = preds_df.dropna(subset=["stress_score_pred"])

    with engine.begin() as conn:
        for _, row in preds_df.iterrows():
            date_str = row["time"].strftime("%Y-%m-%d")
            pred_val = float(row["stress_score_pred"])
            if is_pg:
                sql = text(
                    """
                    INSERT INTO predictions (date, stress_score_pred, model_version)
                    VALUES (:date, :pred, :version)
                    ON CONFLICT (date, model_version) DO UPDATE SET stress_score_pred = EXCLUDED.stress_score_pred
                    """
                )
            else:
                sql = text(
                    """
                    INSERT OR REPLACE INTO predictions (date, stress_score_pred, model_version)
                    VALUES (:date, :pred, :version)
                    """
                )
            conn.execute(sql, {"date": date_str, "pred": pred_val, "version": model_version})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    engine = get_engine()
    print("Loading dataset...")
    df = load_dataset(engine)
    n = len(df)
    print(f"Loaded {n} samples spanning {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    TRAIN_SIZE = int(n * TRAIN_PCT)
    VAL_SIZE   = int(n * VAL_PCT)
    TEST_SIZE  = n - TRAIN_SIZE - VAL_SIZE

    min_required = INPUT_CHUNK + 1
    if n < min_required:
        raise RuntimeError(
            f"Need at least {min_required} samples with both FSI and embeddings, got {n}."
        )

    target     = build_target_series(df)
    covariates = build_covariate_series(df)

    # Temporal split by position
    target_train = target[:TRAIN_SIZE]
    target_val   = target[:TRAIN_SIZE + VAL_SIZE]
    target_test  = target
    cov_full     = covariates

    print(f"Split: {TRAIN_SIZE} train / {VAL_SIZE} val / {TEST_SIZE} test  (70/15/15%)")

    # Build and train model
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
        past_covariates=cov_full[:TRAIN_SIZE],
        future_covariates=cov_full[:TRAIN_SIZE],
        val_series=target_val,
        val_past_covariates=cov_full,
        val_future_covariates=cov_full,
        verbose=True,
    )

    model.save(TIDE_MODEL_PATH)
    print(f"Model saved: {TIDE_MODEL_PATH}")

    # Evaluate splits
    print("Evaluating splits...")
    train_mae_val, train_rmse_val, _ = evaluate_split(
        model, target_train, cov_full[:TRAIN_SIZE], INPUT_CHUNK, "Train"
    )
    val_mae_val, val_rmse_val, _ = evaluate_split(
        model, target_val, cov_full[:TRAIN_SIZE + VAL_SIZE], TRAIN_SIZE, "Val"
    )
    test_mae_val, test_rmse_val, _ = evaluate_split(
        model, target_test, cov_full, TRAIN_SIZE + VAL_SIZE, "Test"
    )

    # Full predictions for DB
    print("Generating full-series predictions for DB...")
    full_preds = model.historical_forecasts(
        series=target,
        past_covariates=cov_full,
        future_covariates=cov_full,
        start=INPUT_CHUNK,
        forecast_horizon=OUTPUT_CHUNK,
        stride=1,
        retrain=False,
        verbose=False,
    )

    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

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

    write_predictions(engine, df, full_preds, model_version)
    print(f"Wrote {len(full_preds)} predictions to DB. Version: {model_version}")


if __name__ == "__main__":
    main()
