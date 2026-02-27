"""
Build the Financial Stress Index (FSI) target from market data.

Uses yfinance to download four proxy series, z-scores each, applies PCA
to extract the first principal component, validates sign using PASO 2019
as a known stress peak, and saves the result to data/fsi_target.csv.

Components:
  ^MERV   - Merval index 30-day rolling volatility (primary stress signal)
  GD30.BA - GD30 bond price inverted (sovereign debt stress proxy)
  ARS=X   - USD/ARS exchange rate (FX pressure proxy)
  EMB     - iShares EM Bond ETF inverted (EM stress proxy)

Usage:
    python src/data/build_fsi_target.py --start 2023-01-01 --end 2024-12-31
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import FSI_CSV
from db.connection import get_engine
from src.utils.log import get_logger
from sqlalchemy import text as _sa_text

log = get_logger(__name__)

# Known stress event for sign validation: PASO 2019 (Aug 12, 2019)
# This date saw a sharp market crash — FSI should be near its maximum
SIGN_VALIDATION_DATE = "2019-08-12"


def _download_series(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for multiple tickers via yfinance."""
    log.info("Downloading tickers: %s", tickers)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    return prices


def _zscore(series: pd.Series) -> pd.Series:
    """Z-score normalise a series, dropping NaN first."""
    clean = series.dropna()
    if clean.std() == 0:
        return series * 0
    return (series - clean.mean()) / clean.std()


def _save_components_to_db(components_df: pd.DataFrame) -> None:
    """Persist normalised FSI components to the fsi_components table."""
    try:
        engine = get_engine()
        is_pg = not engine.url.drivername.startswith("sqlite")
        with engine.begin() as conn:
            for _, row in components_df.iterrows():
                if is_pg:
                    sql = _sa_text(
                        """
                        INSERT INTO fsi_components (date, merv_vol, argt_spread, usd_ars, emb_spread)
                        VALUES (:date, :merv_vol, :argt_spread, :usd_ars, :emb_spread)
                        ON CONFLICT (date) DO UPDATE SET
                            merv_vol    = EXCLUDED.merv_vol,
                            argt_spread = EXCLUDED.argt_spread,
                            usd_ars     = EXCLUDED.usd_ars,
                            emb_spread  = EXCLUDED.emb_spread
                        """
                    )
                else:
                    sql = _sa_text(
                        """
                        INSERT OR REPLACE INTO fsi_components
                            (date, merv_vol, argt_spread, usd_ars, emb_spread)
                        VALUES (:date, :merv_vol, :argt_spread, :usd_ars, :emb_spread)
                        """
                    )
                conn.execute(sql, {
                    "date":        row["date"],
                    "merv_vol":    float(row["merv_vol"]),
                    "argt_spread": float(row["argt_spread"]),
                    "usd_ars":     float(row["usd_ars"]),
                    "emb_spread":  float(row["emb_spread"]),
                })
        log.info("Saved %d rows to fsi_components", len(components_df))
    except Exception as exc:
        log.warning("Could not save components to DB: %s", exc)


def build_fsi(start: str, end: str) -> pd.DataFrame:
    """
    Download market data, run PCA, save data/fsi_target.csv.

    Returns the resulting DataFrame with columns [date, fsi_value].
    """
    log.info("start... build_fsi", ts=datetime.now().strftime("%Y/%m/%d %H:%M"),
             start=start, end=end)
    # Extend start by 60 days to compute rolling vol for early dates
    start_extended = (pd.Timestamp(start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    # --- Download ---
    merv_raw = _download_series(["^MERV"], start_extended, end)
    others_raw = _download_series(["GD30.BA", "ARS=X", "EMB"], start_extended, end)

    # --- Component 1: Merval 30-day rolling volatility ---
    merv_close = merv_raw.squeeze()
    merv_ret = merv_close.pct_change()
    merv_vol = merv_ret.rolling(30).std() * np.sqrt(252)

    # --- Component 2: GD30 price (inverted — falling price = rising stress) ---
    # Fallback to ARGT if GD30.BA is unavailable (delisted / data gaps)
    gd30 = others_raw.get("GD30.BA")
    if gd30 is None or gd30.dropna().empty:
        log.warning("GD30.BA not available, trying ARGT as fallback")
        argt_raw = _download_series(["ARGT"], start_extended, end)
        gd30 = argt_raw.squeeze()
    gd30_stress = -gd30  # invert: lower price = higher stress

    # --- Component 3: USD/ARS rate (higher = more FX pressure = more stress) ---
    ars = others_raw.get("ARS=X")

    # --- Component 4: EMB inverted (lower EM bond price = higher EM stress) ---
    emb = others_raw.get("EMB")
    emb_stress = -emb  # invert

    # --- Align to business-day index, forward-fill gaps, trim to [start, end] ---
    df = pd.DataFrame({
        "merv_vol": merv_vol,
        "gd30_stress": gd30_stress,
        "ars_rate": ars,
        "emb_stress": emb_stress,
    })
    df = df.resample("B").last().ffill()
    df = df.loc[start:end]
    df = df.dropna()

    assert not df.empty, "No valid data after alignment and dropna -- check tickers and date range"

    log.info("Aligned DataFrame shape: %s", df.shape)

    # --- Z-score normalise ---
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # --- Save normalised components to DB ---
    components_df = pd.DataFrame(
        X,
        index=df.index,
        columns=["merv_vol", "argt_spread", "usd_ars", "emb_spread"],
    )
    components_df.index.name = "date"
    components_df = components_df.reset_index()
    components_df["date"] = components_df["date"].dt.strftime("%Y-%m-%d")
    _save_components_to_db(components_df)

    # --- PCA: first component = FSI ---
    pca = PCA(n_components=1)
    fsi_values = pca.fit_transform(X).squeeze()
    log.info("PCA explained variance ratio: %.3f", pca.explained_variance_ratio_[0])

    # --- Sign validation using PASO 2019 ---
    fsi_series = pd.Series(fsi_values, index=df.index, name="fsi_value")

    # Find the closest available date to PASO event
    paso_ts = pd.Timestamp(SIGN_VALIDATION_DATE)
    available = fsi_series.index
    if paso_ts < available[0] or paso_ts > available[-1]:
        log.warning(
            "PASO validation date %s is outside series range [%s, %s]. Skipping sign check.",
            SIGN_VALIDATION_DATE, available[0].date(), available[-1].date(),
        )
    else:
        closest_idx = available.get_indexer([paso_ts], method="nearest")[0]
        paso_fsi = fsi_series.iloc[closest_idx]
        # Check if PASO is in top 10% of stress values
        p90 = fsi_series.quantile(0.90)
        if paso_fsi < p90:
            log.info(
                "PASO FSI value %.3f is below 90th pct (%.3f) -- flipping sign", paso_fsi, p90
            )
            fsi_series = -fsi_series
        else:
            log.info("PASO FSI value %.3f >= 90th pct (%.3f) -- sign OK", paso_fsi, p90)

    # --- Save to CSV ---
    FSI_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = fsi_series.reset_index()
    out.columns = ["date", "fsi_value"]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(FSI_CSV, index=False)
    log.info("finish... build_fsi", ts=datetime.now().strftime("%Y/%m/%d %H:%M"),
             rows=len(out), start=out["date"].iloc[0], end=out["date"].iloc[-1])

    return out


def main():
    parser = argparse.ArgumentParser(description="Build FSI target from market data")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    df = build_fsi(args.start, args.end)
    log.info("build_fsi_main_done", rows=len(df),
             start=df["date"].iloc[0], end=df["date"].iloc[-1])


if __name__ == "__main__":
    main()
