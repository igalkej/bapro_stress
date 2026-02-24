import os
from pathlib import Path

# Resolve repo root (two levels up from this file when running locally)
_REPO_ROOT = Path(__file__).resolve().parent

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{_REPO_ROOT / 'bapro_stress.db'}",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Use local artifacts/ dir when ARTIFACTS_DIR env var is not set
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", str(_REPO_ROOT / "artifacts"))

# TiDE model save path (Darts saves to a single .pt file via model.save())
TIDE_MODEL_PATH = os.getenv("TIDE_MODEL_PATH", str(_REPO_ROOT / "artifacts" / "tide_model.pt"))

LOGS_DIR = Path(os.getenv("LOGS_DIR", str(_REPO_ROOT / "logs")))
FSI_CSV = _REPO_ROOT / "data" / "fsi_target.csv"

# Optuna hyperparameter search
OPTUNA_N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "3"))
OPTUNA_TOP_K    = int(os.getenv("OPTUNA_TOP_K",    "2"))
OPTUNA_DB_URL   = os.getenv(
    "OPTUNA_DB_URL",
    f"sqlite:///{_REPO_ROOT / 'artifacts' / 'optuna.db'}",
)
