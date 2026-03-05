#!/usr/bin/env python3
"""
One-file data pipeline + macro sensitivity regressions (VIX + 10Y yield proxy).

Inputs:
  - A holdings CSV exported from Google Sheets, with a column named "Ticker/CUSIP"

Data sources:
  - yfinance:
      - stocks: Adj Close
      - VIX: ^VIX
      - 10Y proxy: ^TNX (quoted ~ yield*10). Converted to yield% via TNX/10.

Model:
  r_i,t = alpha_i + beta_VIX * dVIX_t + beta_10Y * d10Y_t + eps_i,t
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

import yfinance as yf
import statsmodels.api as sm


# -------------------------
# Config
# -------------------------

@dataclass(frozen=True)
class Config:
    holdings_csv: str
    ticker_col: str = "Ticker/CUSIP"

    years_back: int = 10
    use_log_returns: bool = True

    vix_symbol: str = "^VIX"
    tnx_symbol: str = "^TNX"

    out_dir: str = "data/processed"
    regression_frame_name: str = "regression_frame.parquet"
    betas_name: str = "macro_betas.csv"
    download_report_name: str = "ticker_download_report.csv"


# -------------------------
# Helpers
# -------------------------

def today_utc_date() -> date:
    # Good enough for daily market data; avoids tz complexity.
    return pd.Timestamp.utcnow().date()

def compute_start_end(years_back: int) -> tuple[str, str]:
    end_dt = today_utc_date()
    # Use pandas offset to be robust to leap years.
    start_dt = (pd.Timestamp(end_dt) - pd.DateOffset(years=years_back)).date()
    return start_dt.isoformat(), end_dt.isoformat()

def is_plausible_ticker(x: str) -> bool:
    """
    Keep it simple:
      - uppercase letters, digits, dot, hyphen
      - length 1..10
    Excludes obvious non-tickers.
    """
    x = x.strip().upper()
    if not (1 <= len(x) <= 10):
        return False
    return bool(re.fullmatch(r"[A-Z0-9.\-]+", x))

def extract_tickers_from_holdings(path: str, ticker_col: str) -> list[str]:
    df = pd.read_csv(path)
    if ticker_col not in df.columns:
        raise ValueError(f"Holdings CSV is missing required column: {ticker_col!r}. "
                         f"Found columns: {list(df.columns)}")

    raw = df[ticker_col].dropna().astype(str).str.strip().str.upper()
    tickers = [t for t in raw if is_plausible_ticker(t)]

    # De-dupe preserving order
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    if not uniq:
        raise ValueError("No tickers extracted from holdings CSV. Check the ticker column content.")

    return uniq

def yf_download_adj_close(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Returns a DataFrame of Adj Close with Date index and columns = symbols.
    Handles yfinance shape differences.
    """
    data = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned no data. Check internet / symbols / date range.")

    # MultiIndex columns: (Field, Ticker) for multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" not in data.columns.get_level_values(0):
            raise RuntimeError(f"Expected 'Adj Close' in yfinance data fields. Got fields: "
                               f"{sorted(set(data.columns.get_level_values(0)))}")
        adj = data["Adj Close"].copy()
    else:
        # Single symbol returns single-level columns: Open/High/Low/Close/Adj Close/Volume
        if "Adj Close" not in data.columns:
            raise RuntimeError(f"Expected 'Adj Close' in yfinance columns. Got: {list(data.columns)}")
        adj = data[["Adj Close"]].copy()
        # If single, name column as the symbol itself
        if len(symbols) == 1:
            adj.columns = [symbols[0]]
        else:
            # Unexpected but handle
            adj.columns = ["Adj Close"]

    adj.index = pd.to_datetime(adj.index).tz_localize(None)
    adj = adj.sort_index()
    return adj

def make_regression_frame(
    adj_close: pd.DataFrame,
    vix_level: pd.Series,
    y10_level_pct: pd.Series,
    use_log_returns: bool
) -> pd.DataFrame:
    # Align on common dates
    base = pd.concat(
        [adj_close, vix_level.rename("VIX"), y10_level_pct.rename("Y10")],
        axis=1,
        join="inner"
    ).sort_index()

    # Stock returns
    stock_cols = [c for c in base.columns if c not in ("VIX", "Y10")]
    prices = base[stock_cols]

    if use_log_returns:
        stock_rets = np.log(prices / prices.shift(1))
    else:
        stock_rets = prices.pct_change()

    stock_rets = stock_rets.add_prefix("r_")

    # Macro features
    dVIX = base["VIX"].diff().rename("dVIX")
    dY10 = base["Y10"].diff().rename("dY10")  # percentage points

    reg = pd.concat([stock_rets, dVIX, dY10], axis=1).dropna()
    return reg

def run_ols_per_ticker(reg_frame: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    out = []
    X = reg_frame[["dVIX", "dY10"]].copy()
    X = sm.add_constant(X)

    for t in tickers:
        col = f"r_{t}"
        if col not in reg_frame.columns:
            continue

        y = reg_frame[col]
        model = sm.OLS(y, X, missing="drop").fit()

        out.append({
            "ticker": t,
            "alpha": float(model.params.get("const", np.nan)),
            "beta_vix": float(model.params.get("dVIX", np.nan)),
            "beta_10y": float(model.params.get("dY10", np.nan)),
            "t_beta_vix": float(model.tvalues.get("dVIX", np.nan)),
            "t_beta_10y": float(model.tvalues.get("dY10", np.nan)),
            "p_beta_vix": float(model.pvalues.get("dVIX", np.nan)),
            "p_beta_10y": float(model.pvalues.get("dY10", np.nan)),
            "r2": float(model.rsquared),
            "resid_std": float(np.std(model.resid, ddof=1)),
            "n_obs": int(model.nobs),
        })

    if not out:
        raise RuntimeError("No regressions were run. Likely no valid return columns after downloads/alignment.")

    return pd.DataFrame(out).set_index("ticker").sort_index()

def build_download_report(requested: list[str], adj_close: pd.DataFrame) -> pd.DataFrame:
    got = set(map(str, adj_close.columns))
    rows = []
    for t in requested:
        rows.append({"ticker": t, "downloaded": t in got})
    return pd.DataFrame(rows)


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Holdings-driven macro beta pipeline (VIX + 10Y).")
    parser.add_argument("--holdings_csv", required=True, help="Path to holdings CSV exported from Google Sheets.")
    parser.add_argument("--ticker_col", default="Ticker/CUSIP", help="Column name containing tickers.")
    parser.add_argument("--years", type=int, default=10, help="Lookback window in years.")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory.")
    args = parser.parse_args()

    cfg = Config(
        holdings_csv=args.holdings_csv,
        ticker_col=args.ticker_col,
        years_back=args.years,
        out_dir=args.out_dir,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    start, end = compute_start_end(cfg.years_back)

    tickers = extract_tickers_from_holdings(cfg.holdings_csv, cfg.ticker_col)

    # Download asset prices
    adj_close = yf_download_adj_close(tickers, start, end)

    # Download macro series
    vix_adj = yf_download_adj_close([cfg.vix_symbol], start, end)[cfg.vix_symbol]
    tnx_adj = yf_download_adj_close([cfg.tnx_symbol], start, end)[cfg.tnx_symbol]

    # Convert ^TNX (yield*10) -> yield in %
    y10 = (tnx_adj / 10.0).rename("Y10")

    # Build regression frame
    reg_frame = make_regression_frame(adj_close, vix_adj, y10, use_log_returns=cfg.use_log_returns)

    # Regressions
    betas = run_ols_per_ticker(reg_frame, tickers)

    # Reports + outputs
    report = build_download_report(tickers, adj_close)

    reg_path = os.path.join(cfg.out_dir, cfg.regression_frame_name)
    betas_path = os.path.join(cfg.out_dir, cfg.betas_name)
    report_path = os.path.join(cfg.out_dir, cfg.download_report_name)

    reg_frame.to_parquet(reg_path)
    betas.to_csv(betas_path)
    report.to_csv(report_path, index=False)

    print(f"Saved:\n  {reg_path}\n  {betas_path}\n  {report_path}")
    print(f"Tickers requested: {len(tickers)} | downloaded columns: {adj_close.shape[1]} | regression rows: {len(reg_frame)}")


if __name__ == "__main__":
    main()
