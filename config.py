import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://bapro:bapro_pass@postgres:5432/bapro_stress",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
PCA_COMPONENTS = 20
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/workspace/artifacts")
