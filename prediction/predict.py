"""
Prediction CLI.

Loads the trained PCA + XGBoost model from artifacts/ and scores a new document.
The document is encoded with the local sentence-transformers model.

Usage:
    python prediction/predict.py --text "Global markets sold off sharply as..."
    python prediction/predict.py --file /workspace/my_report.txt
"""
import sys
sys.path.insert(0, "/workspace")

import argparse
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

from config import ARTIFACTS_DIR, EMBEDDING_MODEL

PCA_PATH = os.path.join(ARTIFACTS_DIR, "pca.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "xgb_model.pkl")


def load_artifacts():
    if not os.path.exists(PCA_PATH):
        raise FileNotFoundError(
            f"PCA artifact not found at {PCA_PATH}. Run training/train.py first."
        )
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run training/train.py first."
        )
    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        xgb = pickle.load(f)
    return pca, xgb


def predict(text: str) -> float:
    pca, xgb = load_artifacts()

    model = SentenceTransformer(EMBEDDING_MODEL)
    vec = model.encode([text], convert_to_numpy=True)  # shape: (1, 384)

    vec_pca = pca.transform(vec)  # shape: (1, n_components)
    score = xgb.predict(vec_pca)[0]
    return float(score)


def main():
    parser = argparse.ArgumentParser(description="Score a financial document for stress.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Raw document text (quoted string)")
    group.add_argument("--file", type=str, help="Path to a .txt file")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    score = predict(text)
    print(f"Stress score: {score:.4f}")


if __name__ == "__main__":
    main()
