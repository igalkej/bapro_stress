"""
Training pipeline — Darts TiDE with Optuna hyperparameter optimisation.

Workflow:
  1. Optuna runs N_TRIALS, each trained on TRAIN (70%), evaluated on VAL (15%).
  2. Top-K trials by MAPE_val advance to test evaluation.
  3. Each finalist is retrained on TRAIN+VAL and evaluated on TEST (15%).
  4. The trial with best MAPE_test becomes the production model.
  5. Production model is retrained on the full dataset (TRAIN+VAL+TEST).
  6. Out-of-sample val/test predictions written to training_predictions table.
  7. Trial results written to optuna_trials table.

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
from config import (
    ARTIFACTS_DIR, TIDE_MODEL_PATH,
    OPTUNA_N_TRIALS, OPTUNA_TOP_K,
    FORECAST_HORIZON,
)

from darts import TimeSeries
from darts.models import TiDEModel
from darts.metrics import mae, rmse, mape as darts_mape

# ---------------------------------------------------------------------------
# Split fractions
# ---------------------------------------------------------------------------
TRAIN_PCT = 0.70
VAL_PCT   = 0.15

INPUT_CHUNK  = 2
OUTPUT_CHUNK = 1
STUDY_NAME   = "tide_bapro"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(engine):
    """
    Join articles + article_embeddings + fsi_target on date.
    Mean-pool embeddings per business day.
    Returns DataFrame: date (datetime), vec (np.array), fsi_value (float).
    """
    with engine.connect() as conn:
        fsi_rows = conn.execute(
            text("SELECT date, fsi_value FROM fsi_target ORDER BY date")
        ).fetchall()
        emb_rows = conn.execute(
            text(
                "SELECT a.date, ae.embedding "
                "FROM article_embeddings ae "
                "JOIN articles a ON a.id = ae.id "
                "ORDER BY a.date"
            )
        ).fetchall()

    if not fsi_rows:
        raise RuntimeError("fsi_target table is empty. Run seed_fsi first.")
    if not emb_rows:
        raise RuntimeError("article_embeddings table is empty. Run embed first.")

    date_vecs: dict[str, list] = {}
    for row in emb_rows:
        date_str = str(row[0])[:10]
        emb_raw  = row[1]
        vec = np.array(
            json.loads(emb_raw) if isinstance(emb_raw, str) else list(emb_raw),
            dtype=np.float32,
        )
        date_vecs.setdefault(date_str, []).append(vec)

    mean_vecs = {d: np.mean(vecs, axis=0) for d, vecs in date_vecs.items()}
    emb_dim   = next(iter(mean_vecs.values())).shape[0] if mean_vecs else 384
    zero_vec  = np.zeros(emb_dim, dtype=np.float32)

    records = []
    for row in fsi_rows:
        date_str = str(row[0])[:10]
        records.append({
            "date":      date_str,
            "vec":       mean_vecs.get(date_str, zero_vec),
            "fsi_value": float(row[1]),
        })

    if not records:
        raise RuntimeError("No overlap between fsi_target and article_embeddings.")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# TimeSeries builders
# ---------------------------------------------------------------------------

def build_target_series(df):
    return TimeSeries.from_dataframe(
        df, time_col="date", value_cols="fsi_value",
        fill_missing_dates=True, freq="B",
    )


def build_covariate_series(df):
    dim = len(df.iloc[0]["vec"])
    mat = np.stack(df["vec"].values).astype(np.float32)
    cov_df = pd.DataFrame(mat, index=df["date"].values,
                          columns=[f"emb_{i}" for i in range(dim)])
    cov_df.index = pd.DatetimeIndex(cov_df.index)
    ts = TimeSeries.from_dataframe(cov_df, fill_missing_dates=True, freq="B")
    return TimeSeries.from_dataframe(ts.to_dataframe().fillna(0.0), freq="B")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(params: dict, work_dir: str, model_name: str,
                 progress: bool = False) -> TiDEModel:
    """Instantiate a TiDE model with given hyperparams and optional CSVLogger."""
    input_chunk = params["input_chunk_length"]

    pl_kwargs: dict = {"enable_progress_bar": progress}
    try:
        from pytorch_lightning.loggers import CSVLogger
        log_dir = Path(work_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        pl_kwargs["logger"] = CSVLogger(save_dir=str(log_dir), name="", version="")
    except ImportError:
        pass

    return TiDEModel(
        input_chunk_length=input_chunk,
        output_chunk_length=OUTPUT_CHUNK,
        num_encoder_layers=params.get("num_encoder_layers", 1),
        num_decoder_layers=params.get("num_decoder_layers", 1),
        decoder_output_dim=params.get("decoder_output_dim", 8),
        hidden_size=params["hidden_size"],
        temporal_width_past=min(4, input_chunk),
        temporal_width_future=min(4, input_chunk),
        use_layer_norm=True,
        dropout=params["dropout"],
        n_epochs=params["n_epochs"],
        batch_size=params["batch_size"],
        optimizer_kwargs={"lr": params["lr"]},
        model_name=model_name,
        work_dir=work_dir,
        save_checkpoints=False,
        force_reset=True,
        pl_trainer_kwargs=pl_kwargs,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _historical_forecasts(model, target, covariates, start):
    """
    Rolling-window backtesting with last_points_only=False so that every
    window contributes FORECAST_HORIZON prediction points.  Windows whose
    horizon would extend beyond the end of `target` are excluded by Darts
    (overlap_end=False default), which handles the boundary-truncation case.
    """
    return model.historical_forecasts(
        series=target,
        past_covariates=covariates,
        future_covariates=covariates,
        start=start,
        forecast_horizon=FORECAST_HORIZON,
        stride=1,
        retrain=False,
        last_points_only=False,
        verbose=False,
    )


def _eval_metrics(target, preds):
    """
    Compute metrics across the list of per-window forecasts returned by
    _historical_forecasts (last_points_only=False).  Average MAPE/MAE/RMSE
    over all windows that have at least one matching actual value.
    """
    mape_vals, mae_vals, rmse_vals = [], [], []
    for p in preds:
        actual = target.slice_intersect(p)
        if len(actual) == 0:
            continue
        try:
            mape_vals.append(float(darts_mape(actual, p)))
            mae_vals.append(float(mae(actual, p)))
            rmse_vals.append(float(rmse(actual, p)))
        except Exception:
            continue
    return {
        "mape": float(np.mean(mape_vals)) if mape_vals else float("nan"),
        "mae":  float(np.mean(mae_vals))  if mae_vals  else float("nan"),
        "rmse": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
    }


# ---------------------------------------------------------------------------
# training_predictions table writer
# ---------------------------------------------------------------------------

def write_training_predictions(engine, df, preds_list, split, model_version):
    """
    Write per-window forecast results to training_predictions, preserving
    every horizon step (1..FORECAST_HORIZON) for each rolling window.

    preds_list is a list[TimeSeries] (one per rolling window).  Each window's
    TimeSeries contains up to FORECAST_HORIZON time steps, stored with their
    horizon index so all predictions are visible in the dashboard fan chart.
    """
    is_pg = not engine.url.drivername.startswith("sqlite")

    actual_lookup = {
        row["date"].strftime("%Y-%m-%d"): float(row["fsi_value"])
        for _, row in df.iterrows()
    }

    if is_pg:
        upsert_sql = text(
            "INSERT INTO training_predictions "
            "(date, fsi_actual, fsi_pred, split, horizon, model_version) "
            "VALUES (:date, :actual, :pred, :split, :horizon, :version) "
            "ON CONFLICT (date, split, horizon) DO UPDATE SET "
            "fsi_actual=EXCLUDED.fsi_actual, fsi_pred=EXCLUDED.fsi_pred, "
            "model_version=EXCLUDED.model_version"
        )
    else:
        upsert_sql = text(
            "INSERT OR REPLACE INTO training_predictions "
            "(date, fsi_actual, fsi_pred, split, horizon, model_version) "
            "VALUES (:date, :actual, :pred, :split, :horizon, :version)"
        )

    with engine.begin() as conn:
        for window_ts in preds_list:
            window_df = window_ts.to_dataframe().reset_index()
            window_df.columns = ["time", "fsi_pred"]
            window_df["time"] = pd.to_datetime(window_df["time"])
            window_df = window_df.dropna(subset=["fsi_pred"])
            for horizon_step, (_, row) in enumerate(window_df.iterrows(), start=1):
                date_str   = row["time"].strftime("%Y-%m-%d")
                pred_val   = float(row["fsi_pred"])
                actual_val = actual_lookup.get(date_str)
                conn.execute(upsert_sql, {
                    "date":    date_str,
                    "actual":  actual_val,
                    "pred":    pred_val,
                    "split":   split,
                    "horizon": horizon_step,
                    "version": model_version,
                })


# ---------------------------------------------------------------------------
# optuna_trials table writer
# ---------------------------------------------------------------------------

def write_optuna_trials(engine, study_name, trial_results, model_version):
    is_pg = not engine.url.drivername.startswith("sqlite")
    sql = text(
        "INSERT INTO optuna_trials "
        "(study_name, trial_number, rank_val, mape_val, mape_test, "
        " mae_test, rmse_test, is_production, hyperparams, model_version) "
        "VALUES (:study, :tnum, :rank, :mval, :mtest, "
        " :maet, :rmset, :isprod, :hp, :ver)"
    )
    with engine.begin() as conn:
        for res in trial_results:
            hp_json = json.dumps(res["params"])
            conn.execute(sql, {
                "study":  study_name,
                "tnum":   res["trial_number"],
                "rank":   res["rank_val"],
                "mval":   res["mape_val"],
                "mtest":  res.get("mape_test"),
                "maet":   res.get("mae_test"),
                "rmset":  res.get("rmse_test"),
                "isprod": bool(res.get("is_production")),
                "hp":     hp_json,
                "ver":    model_version,
            })


# ---------------------------------------------------------------------------
# models table writer
# ---------------------------------------------------------------------------

def write_models(engine, eval_results, model_version,
                 train_size, val_size, test_size):
    """
    Persist each finalist model to the models table.

    eval_results is the list of top-K finalist dicts.  The one marked
    is_production=True has its artifact_path set; the rest get NULL.
    """
    is_pg = not engine.url.drivername.startswith("sqlite")
    if is_pg:
        sql = text(
            "INSERT INTO models "
            "(model_version, trial_number, rank_val, is_production, "
            " hyperparams, architecture, train_samples, val_samples, test_samples, "
            " mape_val, mape_test, mae_test, rmse_test, artifact_path) "
            "VALUES (:ver, :tnum, :rank, :isprod, "
            " :hp, :arch, :tr, :va, :te, "
            " :mval, :mtest, :maet, :rmset, :apath) "
            "ON CONFLICT DO NOTHING"
        )
    else:
        sql = text(
            "INSERT OR IGNORE INTO models "
            "(model_version, trial_number, rank_val, is_production, "
            " hyperparams, architecture, train_samples, val_samples, test_samples, "
            " mape_val, mape_test, mae_test, rmse_test, artifact_path) "
            "VALUES (:ver, :tnum, :rank, :isprod, "
            " :hp, :arch, :tr, :va, :te, "
            " :mval, :mtest, :maet, :rmset, :apath)"
        )
    with engine.begin() as conn:
        for res in eval_results:
            params = res["params"]
            # Tuning hyperparams (what Optuna searched over)
            hp_json = json.dumps({
                k: params[k]
                for k in ("hidden_size", "num_encoder_layers", "num_decoder_layers",
                          "dropout", "lr", "n_epochs", "batch_size", "decoder_output_dim")
                if k in params
            })
            # Fixed architecture (not searched by Optuna)
            arch_json = json.dumps({
                "input_chunk_length":  params.get("input_chunk_length", INPUT_CHUNK),
                "output_chunk_length": OUTPUT_CHUNK,
                "forecast_horizon":    FORECAST_HORIZON,
                "use_layer_norm":      True,
                "temporal_width_past": min(4, params.get("input_chunk_length", INPUT_CHUNK)),
                "temporal_width_future": min(4, params.get("input_chunk_length", INPUT_CHUNK)),
            })
            is_prod   = bool(res.get("is_production"))
            art_path  = TIDE_MODEL_PATH if is_prod else None
            conn.execute(sql, {
                "ver":   model_version,
                "tnum":  res["trial_number"],
                "rank":  res["rank_val"],
                "isprod": int(is_prod) if not is_pg else is_prod,
                "hp":    hp_json,
                "arch":  arch_json,
                "tr":    train_size,
                "va":    val_size,
                "te":    test_size,
                "mval":  res.get("mape_val"),
                "mtest": res.get("mape_test"),
                "maet":  res.get("mae_test"),
                "rmset": res.get("rmse_test"),
                "apath": art_path,
            })


# ---------------------------------------------------------------------------
# training_loss table writer
# ---------------------------------------------------------------------------

def write_training_loss(engine, model_version, trial_number, rank_val, work_dir):
    """
    Read the CSVLogger metrics.csv from work_dir/logs/ and persist epoch-level
    train/val loss rows to the training_loss table.
    """
    import csv as _csv

    csv_path = Path(work_dir) / "logs" / "metrics.csv"
    if not csv_path.exists():
        print(f"  [loss] metrics.csv not found at {csv_path}, skipping")
        return

    rows = []
    try:
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for record in reader:
                epoch_val = record.get("epoch")
                if epoch_val is None or epoch_val == "":
                    continue
                try:
                    epoch_int = int(float(epoch_val))
                except ValueError:
                    continue
                # Detect train_loss and val_loss column names flexibly
                train_loss = None
                val_loss   = None
                for col, val in record.items():
                    col_l = col.lower()
                    if val in ("", None):
                        continue
                    try:
                        fval = float(val)
                    except ValueError:
                        continue
                    if "train" in col_l and "loss" in col_l:
                        train_loss = fval
                    elif "val" in col_l and "loss" in col_l:
                        val_loss = fval
                rows.append((epoch_int, train_loss, val_loss))
    except Exception as exc:
        print(f"  [loss] could not parse {csv_path}: {exc}")
        return

    if not rows:
        return

    is_pg = not engine.url.drivername.startswith("sqlite")
    if is_pg:
        sql = text(
            "INSERT INTO training_loss "
            "(model_version, trial_number, rank_val, epoch, train_loss, val_loss) "
            "VALUES (:ver, :tnum, :rank, :ep, :tl, :vl) "
            "ON CONFLICT DO NOTHING"
        )
    else:
        sql = text(
            "INSERT OR IGNORE INTO training_loss "
            "(model_version, trial_number, rank_val, epoch, train_loss, val_loss) "
            "VALUES (:ver, :tnum, :rank, :ep, :tl, :vl)"
        )
    with engine.begin() as conn:
        for ep, tl, vl in rows:
            conn.execute(sql, {
                "ver":  model_version,
                "tnum": trial_number,
                "rank": rank_val,
                "ep":   ep,
                "tl":   tl,
                "vl":   vl,
            })
    print(f"  [loss] trial {trial_number}: {len(rows)} epoch rows written")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    engine = get_engine()
    print("Loading dataset...")
    df = load_dataset(engine)
    n = len(df)
    print(f"Loaded {n} samples: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    TRAIN_SIZE = int(n * TRAIN_PCT)
    VAL_SIZE   = int(n * VAL_PCT)
    TEST_SIZE  = n - TRAIN_SIZE - VAL_SIZE
    print(f"Split: {TRAIN_SIZE} train / {VAL_SIZE} val / {TEST_SIZE} test  (70/15/15%)")

    if n < 5:
        raise RuntimeError(f"Need at least 5 samples, got {n}.")

    target     = build_target_series(df)
    covariates = build_covariate_series(df)

    target_train = target[:TRAIN_SIZE]
    target_val   = target[:TRAIN_SIZE + VAL_SIZE]
    target_test  = target
    cov_full     = covariates

    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    trials_dir = Path(ARTIFACTS_DIR) / "optuna_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Capture val preds per trial (to avoid retraining later)
    _trial_val_preds: dict[int, object] = {}
    _trial_params:    dict[int, dict]   = {}

    # ── Optuna study ─────────────────────────────────────────────────────────
    def objective(trial):
        params = {
            "input_chunk_length": INPUT_CHUNK,
            "hidden_size":        trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 2),
            "num_decoder_layers": trial.suggest_int("num_decoder_layers", 1, 2),
            "dropout":            trial.suggest_float("dropout", 0.05, 0.3),
            "lr":                 trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "n_epochs":           trial.suggest_categorical("n_epochs", [100, 300, 500]),
            "batch_size":         trial.suggest_categorical("batch_size", [4, 8, 16]),
            "decoder_output_dim": 8,
        }
        work = str(trials_dir / f"trial_{trial.number}")
        Path(work).mkdir(parents=True, exist_ok=True)
        model = _build_model(params, work, f"tide_trial_{trial.number}")
        model.fit(
            series=target_train,
            past_covariates=cov_full[:TRAIN_SIZE],
            future_covariates=cov_full[:TRAIN_SIZE],
            val_series=target_val,
            val_past_covariates=cov_full,
            val_future_covariates=cov_full,
            verbose=False,
        )
        val_preds = _historical_forecasts(
            model, target_val, cov_full[:TRAIN_SIZE + VAL_SIZE],
            TRAIN_SIZE + params["input_chunk_length"],
        )
        metrics = _eval_metrics(target_val, val_preds)
        _trial_val_preds[trial.number] = val_preds
        _trial_params[trial.number]    = params
        print(f"  Trial {trial.number}: MAPE_val={metrics['mape']:.4f}")
        return metrics["mape"]

    print(f"Running Optuna study ({OPTUNA_N_TRIALS} trials)...")
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
    )
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)

    # ── Select top-K trials ──────────────────────────────────────────────────
    completed = [t for t in study.trials if t.value is not None]
    completed.sort(key=lambda t: t.value)
    top_trials = completed[:OPTUNA_TOP_K]
    print(f"Top {len(top_trials)} trials selected for test evaluation:")
    for rank, t in enumerate(top_trials, 1):
        print(f"  Rank {rank}: trial {t.number}  MAPE_val={t.value:.4f}")

    # ── Evaluate top-K on TEST ───────────────────────────────────────────────
    eval_results = []
    for rank, trial in enumerate(top_trials, start=1):
        params = _trial_params.get(trial.number, trial.params)
        work = str(trials_dir / f"rank_{rank}_eval")
        Path(work).mkdir(parents=True, exist_ok=True)

        print(f"  Evaluating rank {rank} on TEST (retrain on TRAIN+VAL)...")
        model_eval = _build_model(params, work, f"tide_eval_rank{rank}")
        model_eval.fit(
            series=target_val,  # = target[:TRAIN_SIZE + VAL_SIZE]
            past_covariates=cov_full[:TRAIN_SIZE + VAL_SIZE],
            future_covariates=cov_full[:TRAIN_SIZE + VAL_SIZE],
            val_series=target_test,
            val_past_covariates=cov_full,
            val_future_covariates=cov_full,
            verbose=False,
        )
        test_preds = _historical_forecasts(
            model_eval, target_test, cov_full,
            TRAIN_SIZE + VAL_SIZE + params["input_chunk_length"],
        )
        test_metrics = _eval_metrics(target_test, test_preds)
        print(
            f"    MAPE_test={test_metrics['mape']:.4f}  "
            f"MAE={test_metrics['mae']:.4f}  "
            f"RMSE={test_metrics['rmse']:.4f}"
        )

        eval_results.append({
            "rank_val":     rank,
            "trial_number": trial.number,
            "params":       params,
            "mape_val":     trial.value,
            "mape_test":    test_metrics["mape"],
            "mae_test":     test_metrics["mae"],
            "rmse_test":    test_metrics["rmse"],
            "model_eval":   model_eval,
            "val_preds":    _trial_val_preds.get(trial.number),
            "test_preds":   test_preds,
            "is_production": False,
        })

    # ── Best by MAPE_test ────────────────────────────────────────────────────
    best = min(eval_results, key=lambda x: x["mape_test"])
    best["is_production"] = True
    best_params = best["params"]
    print(f"Production model: trial {best['trial_number']}  MAPE_test={best['mape_test']:.4f}")

    # ── Train production model on full dataset ───────────────────────────────
    prod_work = str(Path(ARTIFACTS_DIR) / "prod")
    Path(prod_work).mkdir(parents=True, exist_ok=True)
    print("Training production model on full dataset (TRAIN+VAL+TEST)...")
    model_prod = _build_model(best_params, prod_work, "tide_bapro", progress=True)
    model_prod.fit(
        series=target,
        past_covariates=cov_full,
        future_covariates=cov_full,
        verbose=True,
    )
    model_prod.save(TIDE_MODEL_PATH)
    print(f"Production model saved: {TIDE_MODEL_PATH}")

    # ── Version stamp (used by all DB writers below) ──────────────────────────
    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ── Write training predictions (all horizons, out-of-sample) ─────────────
    if best["val_preds"] is not None:
        write_training_predictions(engine, df, best["val_preds"], "val", model_version)
    write_training_predictions(engine, df, best["test_preds"], "test", model_version)
    print(f"Training predictions written (all horizons). Version: {model_version}")

    # ── Write Optuna trial results to DB ─────────────────────────────────────
    write_optuna_trials(engine, STUDY_NAME, eval_results, model_version)
    print(f"Optuna trial results written to DB ({len(eval_results)} rows)")

    # ── Write finalist models to models table ─────────────────────────────────
    write_models(engine, eval_results, model_version, TRAIN_SIZE, VAL_SIZE, TEST_SIZE)
    print(f"Finalist models written to models table ({len(eval_results)} rows)")

    # ── Write epoch loss curves to training_loss table ────────────────────────
    for res in eval_results:
        rank  = res["rank_val"]
        tnum  = res["trial_number"]
        work  = str(trials_dir / f"rank_{rank}_eval")
        write_training_loss(engine, model_version, tnum, rank, work)
    print("Training loss curves written to DB")

    # ── Save metadata ─────────────────────────────────────────────────────────
    metadata = {
        "model_version":   model_version,
        "train_samples":   TRAIN_SIZE,
        "val_samples":     VAL_SIZE,
        "test_samples":    TEST_SIZE,
        # Primary metrics (MAPE)
        "mape_val_best":   best["mape_val"],
        "mape_test_prod":  best["mape_test"],
        # Secondary metrics
        "mae_test_prod":   best["mae_test"],
        "rmse_test":       best["rmse_test"],
        # Legacy key for CI computation in dashboard
        "test_rmse":       best["rmse_test"],
        # Optuna
        "optuna_n_trials": OPTUNA_N_TRIALS,
        "optuna_top_k":    OPTUNA_TOP_K,
        "best_trial":      best["trial_number"],
        "best_params":     best_params,
        # All finalist results
        "eval_results": [
            {k: v for k, v in r.items()
             if k not in ("model_eval", "val_preds", "test_preds")}
            for r in eval_results
        ],
    }
    meta_path = Path(ARTIFACTS_DIR) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {meta_path}")
    print(f"Done. MAPE_val_best={best['mape_val']:.4f}  MAPE_test_prod={best['mape_test']:.4f}")


if __name__ == "__main__":
    main()
