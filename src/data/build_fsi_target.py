"""
Build the Financial Stress Index (FSI) target from market data.

Uses yfinance to download four Argentine market proxy series plus the OFR
Financial Stress Index (global context), z-scores each, applies PCA to
extract the first principal component, validates sign using PASO 2019 as
a known stress peak, and writes results directly to the database.

Components:
  ^MERV   - Merval index 30-day rolling volatility (primary stress signal)
  GD30.BA - GD30 bond price inverted (sovereign debt stress proxy)
  ARS=X   - USD/ARS exchange rate (FX pressure proxy)
  EMB     - iShares EM Bond ETF inverted (EM stress proxy)
  OFR FSI - OFR Financial Stress Index (global context, financialresearch.gov)

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


OFR_FSI_URL = "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"


OFR_COL_MAP = {
    "OFR FSI":                   "ofr_fsi",
    "Credit":                    "ofr_credit",
    "Equity valuation":          "ofr_equity",
    "Safe assets":               "ofr_safe_assets",
    "Funding":                   "ofr_funding",
    "Volatility":                "ofr_volatility",
    "United States":             "ofr_us",
    "Other advanced economies":  "ofr_other_adv",
    "Emerging markets":          "ofr_em",
}

# Sub-indicators used as PCA inputs (composite excluded to avoid multicollinearity)
OFR_PCA_COLS = [
    "ofr_credit", "ofr_equity", "ofr_safe_assets", "ofr_funding",
    "ofr_volatility", "ofr_us", "ofr_other_adv", "ofr_em",
]


def _download_ofr_fsi(start: str, end: str) -> pd.DataFrame:
    """
    Download all OFR FSI indicator columns and return as a DataFrame.

    Columns returned: ofr_fsi (composite) + 8 sub-indicators.
    The composite is stored as reference; sub-indicators feed the PCA.
    """
    log.info("Downloading OFR FSI from financialresearch.gov")
    try:
        raw = pd.read_csv(OFR_FSI_URL, parse_dates=["Date"], index_col="Date")
        df = raw.rename(columns=OFR_COL_MAP)[list(OFR_COL_MAP.values())]
        df = df.loc[start:end]
        if df.empty:
            raise ValueError(f"OFR FSI returned no data for range [{start}, {end}]")
        log.info("OFR FSI: %d rows (%s to %s), %d indicators",
                 len(df), df.index[0].date(), df.index[-1].date(), len(df.columns))
        return df
    except Exception as exc:
        log.error("ofr_fsi_download_failed", reason=str(exc))
        raise


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
        data_cols = [c for c in components_df.columns if c != "date"]
        col_list = ", ".join(data_cols)
        val_list = ", ".join(f":{c}" for c in data_cols)
        update_list = ", ".join(f"{c} = EXCLUDED.{c}" for c in data_cols)

        if is_pg:
            sql = _sa_text(
                f"INSERT INTO fsi_components (date, {col_list}) "
                f"VALUES (:date, {val_list}) "
                f"ON CONFLICT (date) DO UPDATE SET {update_list}"
            )
        else:
            sql = _sa_text(
                f"INSERT OR REPLACE INTO fsi_components (date, {col_list}) "
                f"VALUES (:date, {val_list})"
            )

        with engine.begin() as conn:
            for _, row in components_df.iterrows():
                params = {"date": row["date"]}
                params.update({c: float(row[c]) for c in data_cols})
                conn.execute(sql, params)
        log.info("Saved %d rows to fsi_components (%d columns)", len(components_df), len(data_cols))
    except Exception as exc:
        log.warning("Could not save components to DB: %s", exc)


def _save_fsi_to_db(fsi_df: pd.DataFrame) -> None:
    """Upsert [date, fsi_value] rows into the fsi_target table."""
    try:
        engine = get_engine()
        is_pg = not engine.url.drivername.startswith("sqlite")
        if is_pg:
            sql = _sa_text(
                "INSERT INTO fsi_target (date, fsi_value) VALUES (:date, :v) "
                "ON CONFLICT (date) DO UPDATE SET fsi_value = EXCLUDED.fsi_value"
            )
        else:
            sql = _sa_text(
                "INSERT OR REPLACE INTO fsi_target (date, fsi_value) VALUES (:date, :v)"
            )
        with engine.begin() as conn:
            for _, row in fsi_df.iterrows():
                conn.execute(sql, {"date": row["date"], "v": float(row["fsi_value"])})
        log.info("Saved %d rows to fsi_target", len(fsi_df))
    except Exception as exc:
        log.warning("Could not save FSI to DB: %s", exc)


def build_fsi(start: str, end: str) -> pd.DataFrame:
    """
    Download market data, run PCA, write results to DB.

    Upserts fsi_components and fsi_target tables directly from memory.
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

    # --- Components 5-13: OFR FSI sub-indicators + composite (global context) ---
    ofr_df = _download_ofr_fsi(start_extended, end)

    # --- Align to business-day index, forward-fill gaps, trim to [start, end] ---
    arg_df = pd.DataFrame({
        "merv_vol":   merv_vol,
        "gd30_stress": gd30_stress,
        "ars_rate":   ars,
        "emb_stress": emb_stress,
    })
    df = arg_df.join(ofr_df, how="left")
    df = df.resample("B").last().ffill()
    df = df.loc[start:end]
    df = df.dropna()

    assert not df.empty, "No valid data after alignment and dropna -- check tickers and date range"

    log.info("Aligned DataFrame shape: %s", df.shape)

    # --- Z-score normalise all columns ---
    all_cols = list(df.columns)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(df)

    # --- Save all normalised components to DB (composite stored as reference) ---
    db_col_names = ["merv_vol", "argt_spread", "usd_ars", "emb_spread"] + \
                   [c for c in all_cols if c.startswith("ofr_")]
    components_df = pd.DataFrame(X_all, index=df.index, columns=db_col_names)
    components_df.index.name = "date"
    components_df = components_df.reset_index()
    components_df["date"] = components_df["date"].dt.strftime("%Y-%m-%d")
    _save_components_to_db(components_df)

    # --- PCA: 4 Argentine + 8 OFR sub-indicators (composite excluded) ---
    pca_col_names = ["merv_vol", "gd30_stress", "ars_rate", "emb_stress"] + OFR_PCA_COLS
    pca_indices = [all_cols.index(c) for c in pca_col_names]
    X_pca = X_all[:, pca_indices]

    pca = PCA(n_components=1)
    fsi_values = pca.fit_transform(X_pca).squeeze()
    log.info("PCA explained variance ratio: %.3f (12 inputs: 4 AR + 8 OFR sub-indicators)",
             pca.explained_variance_ratio_[0])

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

    # --- Save to DB ---
    out = fsi_series.reset_index()
    out.columns = ["date", "fsi_value"]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    _save_fsi_to_db(out)
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
