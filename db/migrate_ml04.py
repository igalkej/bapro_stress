"""
Migration ML-04: Add trial_number to training_predictions.

Changes:
  - training_predictions: add `trial_number` column
  - Change UNIQUE from (date, split, horizon) to (date, split, horizon, trial_number)
  - Clear training_predictions (retrain to repopulate with per-trial rows)

Idempotent: safe to run multiple times.

Usage:
    python db/migrate_ml04.py
    docker compose run --rm app python db/migrate_ml04.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from db.connection import get_engine


def migrate_postgres(conn):
    print("PostgreSQL migration ML-04...")

    # Add trial_number column
    conn.execute(text(
        "ALTER TABLE training_predictions "
        "ADD COLUMN IF NOT EXISTS trial_number INT NOT NULL DEFAULT 0"
    ))
    print("  training_predictions.trial_number: OK")

    # Drop old unique constraint (name may vary)
    for old_name in [
        "training_predictions_date_split_horizon_key",
        "uq_training_predictions_date_split_horizon",
    ]:
        try:
            conn.execute(text(
                f"ALTER TABLE training_predictions "
                f"DROP CONSTRAINT IF EXISTS {old_name}"
            ))
        except Exception:
            pass

    # Add new unique constraint
    conn.execute(text(
        "ALTER TABLE training_predictions "
        "DROP CONSTRAINT IF EXISTS training_predictions_date_split_horizon_trial_key"
    ))
    conn.execute(text(
        "ALTER TABLE training_predictions "
        "ADD CONSTRAINT training_predictions_date_split_horizon_trial_key "
        "UNIQUE (date, split, horizon, trial_number)"
    ))
    print("  training_predictions UNIQUE(date, split, horizon, trial_number): OK")

    # Clear stale data — retrain will repopulate with trial_number per finalist
    conn.execute(text("DELETE FROM training_predictions"))
    print("  training_predictions cleared (retrain to repopulate per trial)")


def migrate_sqlite(conn):
    print("SQLite migration ML-04...")

    # Check if trial_number column already exists
    cols = [
        row[1] for row in conn.execute(
            text("PRAGMA table_info(training_predictions)")
        ).fetchall()
    ]
    if "trial_number" not in cols:
        conn.execute(text(
            "ALTER TABLE training_predictions RENAME TO training_predictions_old"
        ))
        conn.execute(text("""
            CREATE TABLE training_predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                date          TEXT    NOT NULL,
                fsi_actual    REAL,
                fsi_pred      REAL    NOT NULL,
                split         TEXT    NOT NULL,
                horizon       INTEGER NOT NULL DEFAULT 1,
                trial_number  INTEGER NOT NULL DEFAULT 0,
                model_version TEXT    NOT NULL,
                UNIQUE (date, split, horizon, trial_number)
            )
        """))
        # Migrate existing rows with trial_number=0
        conn.execute(text(
            "INSERT INTO training_predictions "
            "(date, fsi_actual, fsi_pred, split, horizon, trial_number, model_version) "
            "SELECT date, fsi_actual, fsi_pred, split, horizon, 0, model_version "
            "FROM training_predictions_old"
        ))
        conn.execute(text("DROP TABLE training_predictions_old"))
        print("  training_predictions migrated with trial_number column")

        # Clear stale rows
        conn.execute(text("DELETE FROM training_predictions"))
        print("  training_predictions cleared (retrain to repopulate per trial)")
    else:
        print("  training_predictions.trial_number already exists: skip")


def main():
    engine = get_engine()
    is_pg = not engine.url.drivername.startswith("sqlite")

    with engine.begin() as conn:
        if is_pg:
            migrate_postgres(conn)
        else:
            migrate_sqlite(conn)

    print("Migration ML-04 complete.")
    print("Next: retrain with  python training/train.py")


if __name__ == "__main__":
    main()
