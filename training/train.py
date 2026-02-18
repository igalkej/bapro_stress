"""
Training step.

Joins documents + embeddings + stress_index on date.
Applies PCA (n_components=20) to reduce the 384-dim embedding.
Trains XGBoost on the PCA-reduced features.
Saves artifacts/pca.pkl, artifacts/xgb_model.pkl, artifacts/metadata.json.
Writes predictions back to the `predictions` table.

Usage (inside the container):
    python training/train.py
"""
import sys
sys.path.insert(0, "/workspace")

import json
import pickle
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sqlalchemy import text

from db.connection import get_engine
from config import ARTIFACTS_DIR, PCA_COMPONENTS

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

PCA_PATH = os.path.join(ARTIFACTS_DIR, "pca.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "xgb_model.pkl")
META_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")


def load_dataset(engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    d.id         AS doc_id,
                    d.doc_date,
                    e.embedding_vector AS vec,
                    s.stress_value
                FROM documents d
                JOIN embeddings  e ON e.doc_id   = d.id
                JOIN stress_index s ON s.index_date = d.doc_date
                ORDER BY d.doc_date
                """
            )
        ).fetchall()

    if not rows:
        raise RuntimeError(
            "No joined rows found. Did you run db/seed.py and training/embed.py first?"
        )

    records = []
    for row in rows:
        records.append(
            {
                "doc_id": row.doc_id,
                "doc_date": row.doc_date,
                "vec": np.array(row.vec, dtype=np.float32),
                "stress_value": float(row.stress_value),
            }
        )

    df = pd.DataFrame(records)
    X = np.stack(df["vec"].values)
    y = df["stress_value"].values
    return df, X, y


def write_predictions(engine, df, preds, model_version: str):
    # Clear existing predictions for these doc_ids
    doc_ids = df["doc_id"].tolist()
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM predictions WHERE doc_id = ANY(:ids)"),
            {"ids": doc_ids},
        )
        for doc_id, pred in zip(doc_ids, preds):
            conn.execute(
                text(
                    """
                    INSERT INTO predictions (doc_id, stress_score_pred, model_version)
                    VALUES (:doc_id, :score, :version)
                    """
                ),
                {"doc_id": int(doc_id), "score": float(pred), "version": model_version},
            )


def main():
    engine = get_engine()
    df, X, y = load_dataset(engine)
    n_samples = len(df)
    print(f"Loaded {n_samples} samples (dim={X.shape[1]})")

    # Temporal holdout: last 3 rows are validation
    val_size = 3
    train_idx = slice(0, n_samples - val_size)
    val_idx = slice(n_samples - val_size, n_samples)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # PCA — fit on training set only
    n_components = min(PCA_COMPONENTS, X_train.shape[0], X_train.shape[1])
    print(f"Fitting PCA({n_components}) on {len(y_train)} training samples …")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    print(f"Explained variance ratio (cumulative): {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

    # XGBoost
    print("Training XGBoost …")
    xgb = XGBRegressor(
        max_depth=2,
        n_estimators=200,
        learning_rate=0.05,
        reg_lambda=2.0,
        subsample=0.8,
        random_state=42,
        eval_metric="mae",
    )
    xgb.fit(
        X_train_pca,
        y_train,
        eval_set=[(X_val_pca, y_val)],
        verbose=False,
    )

    # Evaluate
    y_val_pred = xgb.predict(X_val_pca)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    y_train_pred = xgb.predict(X_train_pca)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    print(f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

    # Save artifacts
    with open(PCA_PATH, "wb") as f:
        pickle.dump(pca, f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(xgb, f)

    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    feature_importances = {
        f"pca_{i}": float(v)
        for i, v in enumerate(xgb.feature_importances_)
    }
    metadata = {
        "model_version": model_version,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "pca_components": int(n_components),
        "pca_explained_variance": float(pca.explained_variance_ratio_.cumsum()[-1]),
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "feature_importances": feature_importances,
        "xgb_params": xgb.get_params(),
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Artifacts saved → {ARTIFACTS_DIR}")

    # Predict on all samples and write to DB
    X_all_pca = pca.transform(X)
    all_preds = xgb.predict(X_all_pca)
    write_predictions(engine, df, all_preds, model_version)
    print(f"Wrote {len(all_preds)} predictions to the predictions table.")
    print(f"Model version: {model_version}")


if __name__ == "__main__":
    main()
