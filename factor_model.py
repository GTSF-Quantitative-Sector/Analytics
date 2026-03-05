from __future__ import annotations

import os
import re
import sys
import time
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.api_client import APIClient
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import requests
import requests_cache
from dotenv import load_dotenv

# =========================
# Config
# =========================
START_DATE = date(2020, 12, 15) # user input
END_DATE = date(2025, 12, 15)  # or up to today date.today()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FUND_DIR = Path("fundamentals")
FUND_DIR.mkdir(exist_ok=True)


CACHE_DIR = Path(".http_cache")
CACHE_DIR.mkdir(exist_ok=True)
FIN_CACHE_DIR = Path(".cache_financials")
FIN_CACHE_DIR.mkdir(exist_ok=True)
TICKER_CACHE_PATH = Path(".cache_sp500_tickers.json")
TICKER_LIST_PATH = Path("tickers.txt") #change to our portfolio holdings
SPY_PATH = DATA_DIR / "SPY.csv"

# Rate limit knobs
REQUESTS_PER_MINUTE = 300
SLEEP_BETWEEN_CALLS = max(0.02, 60.0 / REQUESTS_PER_MINUTE)

BASE_V2 = "https://api.polygon.io/v2"
BASE_V3 = "https://api.polygon.io/v3"
BASE_VX = "https://api.polygon.io"

# =========================
# Environment / Session
# =========================
load_dotenv()
API_KEY = '4cR_irLDgivxae1WO4y0Wb30VYxXRkQj' # os.getenv("POLYGON_API_KEY")
if not API_KEY:
    print("ERROR: POLYGON_API_KEY missing. Add it to your .env file.", file=sys.stderr)
    sys.exit(1)

SESSION = requests_cache.CachedSession(
    cache_name=str(CACHE_DIR / "polygon_cache"),
    backend="sqlite",
    allowable_methods=("GET",),
    expire_after=timedelta(days=3),
    stale_if_error=True,
)
SESSION.params = SESSION.params or {}
SESSION.params.update({"apiKey": API_KEY})

def _sleep():
    time.sleep(SLEEP_BETWEEN_CALLS)

def _iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")

def _get(
    url: str, params: Optional[Dict[str, Any]] = None, retries: int = 5
) -> Dict[str, Any]:
    """GET with simple retry/backoff and rate-limit awareness."""
    params = params or {}
    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            if not resp.from_cache:
                _sleep()
            if resp.status_code == 429:
                time.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == retries:
                raise
            time.sleep(backoff)
            backoff = min(30.0, backoff * 2.0)
    # Should not reach here
    return {}

def _fin_cache_path(ticker: str) -> Path:
    return FIN_CACHE_DIR / f"{ticker}.json"

# =========================
# Universe (S&P 500) NEED TO FIX CURRENTLY GATHERING TOO MANY TICKERS
# =========================
def load_tickers_from_file(path: Path) -> Optional[List[str]]:
    """Return tickers specified in a plain-text file; None if file missing."""
    if not path.exists():
        return None

    try:
        contents = path.read_text().splitlines()
    except OSError as exc:
        print(f"WARN unable to read {path}: {exc}", file=sys.stderr)
        return None

    tickers: List[str] = []
    for raw_line in contents:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in re.split(r"[\s,]+", line) if p.strip()]
        tickers.extend(parts)

    deduped = sorted({t.upper() for t in tickers})
    if not deduped:
        print(f"WARN {path} exists but contains no tickers.", file=sys.stderr)
    return deduped


def get_sp500_tickers() -> List[str]:
    """
    Fetch S&P 500 tickers via Polygon reference API.
    Caches results to .cache_sp500_tickers.json.
    """
    if TICKER_CACHE_PATH.exists():
        try:
            return sorted(set(json.loads(TICKER_CACHE_PATH.read_text())))
        except Exception:
            pass  # fall through to refetch

    tickers: List[str] = []
    url = f"{BASE_V3}/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "limit": 1000,
        "index": "sp500",
        "sort": "ticker",
    }

    while True:
        data = _get(url, params=params)
        results = data.get("results") or []
        for r in results:
            sym = r.get("ticker")
            # Filter to plain equity tickers
            if sym and all(ch.isalnum() or ch in {".", "-", "/"} for ch in sym):
                tickers.append(sym)

        next_url = data.get("next_url")
        if next_url:
            url = next_url  # already includes query params; API key auto-attached by session
            params = {}
        else:
            break

    tickers = sorted(set(tickers))
    if not tickers:
        raise RuntimeError(
            "Could not fetch S&P 500 membership via Polygon. "
            "Your plan may not include index membership. "
            "Workaround: hardcode tickers = ['AAPL','MSFT',...]"
        )

    TICKER_CACHE_PATH.write_text(json.dumps(tickers, indent=2))
    return tickers

# =========================
# Prices (daily aggregates)
# =========================
def get_daily_bars(api: APIClient, ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    /v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?adjusted=true
    Returns DataFrame: date, open, close, high, low, volume, daily_change, daily_change_pct
    """
    # url = f"{BASE_V2}/aggs/ticker/{ticker}/range/1/day/{_iso(start)}/{_iso(end)}"
    # params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    data = api.get_daily_bars(ticker, start, end) # get(url, params=params)
    if data.empty:
        print(f"[WARN]: Data is empty for {ticker}")
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "daily_change",
                "daily_change_pct",
            ]
        )

    data = data.sort_values("date").reset_index(drop=True)

    data["daily_change"] = data["close"] - data["open"]
    with np.errstate(divide="ignore", invalid="ignore"):
        data["daily_change_pct"] = np.where(
            data["open"].abs() > 0,
            data["daily_change"] / data["open"],
            np.nan,
        )

    return data

# =========================
# Financials
# =========================

def get_quarterly_financials(api: APIClient, ticker: str) -> pd.DataFrame:
    ref_cache_path = _fin_cache_path(ticker)

    if ref_cache_path.exists():
        ref_raw = json.loads(ref_cache_path.read_text())
    else:
        ref_url = f"{BASE_VX}/vX/reference/financials"
        ref_params = {
            "ticker": ticker,
            "timeframe": "quarterly",
            "limit": 100,
            "sort": "period_of_report_date",
            "order": "asc",
        }
        ref_raw = _get(ref_url, params=ref_params)
        ref_cache_path.write_text(json.dumps(ref_raw))

    cf_url = f"{BASE_VX}/stocks/financials/v1/cash-flow-statements"
    cf_params = {
        "tickers": ticker,
        "timeframe": "quarterly",
        "limit": 100,
    }

    cf_raw = _get(cf_url, params=cf_params)

    cf_by_period = {}
    for blk in cf_raw.get("results", []):
        key = (blk.get("filing_date"))
        cf_by_period[key] = blk

    rows = []

    for blk in ref_raw.get("results", []):
        fin = blk.get("financials", {}) or {}
        is_ = fin.get("income_statement", {}) or {}
        bs = fin.get("balance_sheet", {}) or {}

        filing_date = blk.get("filing_date")

        if not filing_date:
            continue

        cf_blk = cf_by_period.get(filing_date, {}) or {}

        revenue = is_.get("revenues", {}).get("value")
        cogs = is_.get("cost_of_revenue", {}).get("value")
        net_income = is_.get("net_income_loss", {}).get("value")

        total_assets = bs.get("assets", {}).get("value")
        equity = bs.get("equity", {}).get("value")
        liabilities = bs.get("liabilities", {}).get("value")
        shares = is_.get("diluted_average_shares", {}).get("value")

        operating_cf = cf_blk.get("net_cash_from_operating_activities")

        market_cap = 0
        if equity is not None and shares is not None:
            bvps = equity / shares
            market_cap = bvps * shares

        rows.append({
            "period_end": pd.to_datetime(filing_date),

            # "market_cap": market_cap,
            "revenue": revenue,
            "cogs": cogs,
            "total_assets": total_assets,
            "net_income": net_income,
            "equity": equity,
            "liabilities": liabilities,
            "operating_cf": operating_cf,
            "shares": shares,

            # "pb_ratio": (
            #     market_cap / equity
            #     if market_cap and equity
            #     else np.nan
            # ),
            # "earnings_yield": (
            #     net_income / market_cap
            #     if net_income and market_cap
            #     else np.nan
            # ),
            # "cf_yield": (
            #     operating_cf / market_cap
            #     if operating_cf and market_cap
            #     else np.nan
            # ),
            "liabilities_to_equity": (
                liabilities / equity
                if liabilities and equity
                else np.nan
            ),
        })

    fin_df = pd.DataFrame(rows)

    if fin_df.empty:
        return fin_df

    return fin_df.sort_values("period_end").reset_index(drop=True)

def attach_fin(daily: pd.DataFrame, fin: pd.DataFrame):
    daily_df = daily.copy()
    quarterly_df = fin.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    quarterly_df['period_end'] = pd.to_datetime(quarterly_df['period_end'])

    daily_df = daily_df.sort_values('date')
    quarterly_df = quarterly_df.sort_values('period_end')
    merged_df = pd.merge_asof(
        daily_df,
        quarterly_df,
        left_on='date',
        right_on='period_end',
        direction='backward'  
    )

    merged_df['market_cap'] = merged_df['close'] * merged_df['shares']

    merged_df['pb_ratio'] = np.where(
        (merged_df['market_cap'].notna()) & (merged_df['equity'] > 0),
        merged_df['market_cap'] / merged_df['equity'],
        np.nan
    )

    merged_df['earnings_yield'] = np.where(
        (merged_df['net_income'].notna()) & (merged_df['market_cap'] > 0),
        merged_df['net_income'] / merged_df['market_cap'],
        np.nan
    )

    merged_df['cf_yield'] = np.where(
        (merged_df['operating_cf'].notna()) & (merged_df['market_cap'] > 0),
        merged_df['operating_cf'] / merged_df['market_cap'],
        np.nan
    )

    return merged_df.reset_index(drop=True)

def run_ticker(api: APIClient, ticker: str, start: date, end: date) -> pd.DataFrame:
    prices = get_daily_bars(api, ticker, start, end)
    fin = get_quarterly_financials(api, ticker)

    final = attach_fin(prices, fin)
    final['ticker'] = ticker   # 🔑 REQUIRED
    return final

def quarterly_agg(df: pd.DataFrame):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Convert to quarterly period
    df['quarter'] = df['date'].dt.to_period('Q')

    exclude_cols = [
        'date', 'open', 'high', 'low', 'volume', 'vwap',
        'daily_change', 'daily_change_pct', 'quarter', 'period_end'
    ]
    fundamentals_cols = [col for col in df.columns if col not in exclude_cols]

    # Aggregation logic
    agg_dict = {'close': 'last'}
    for col in fundamentals_cols:
        agg_dict[col] = 'last'

    # Aggregate
    quarterly_df = df.groupby('quarter').agg(agg_dict).reset_index()

    # 🔑 Use QUARTER START date (MM-01-YYYY)
    quarterly_df['date'] = (
        quarterly_df['quarter']
        .dt.start_time              # YYYY-MM-01
        .dt.strftime('%m-01-%Y')    # MM-01-YYYY
    )

    quarterly_df = quarterly_df.drop(columns='quarter')

    # Sort chronologically
    quarterly_df['date_sort'] = pd.to_datetime(quarterly_df['date'], format='%m-%d-%Y')
    quarterly_df = quarterly_df.sort_values('date_sort').reset_index(drop=True)
    quarterly_df = quarterly_df.drop(columns='date_sort')

    # Quarterly return
    quarterly_df['quarterly_return'] = quarterly_df['close'].pct_change()
    quarterly_df = quarterly_df[
        quarterly_df['quarterly_return'].notna()
    ].reset_index(drop=True)

    # Reorder columns
    cols = quarterly_df.columns.tolist()
    cols.remove('date')
    cols.remove('close')
    cols.remove('quarterly_return')
    quarterly_df = quarterly_df[['date', 'close', 'quarterly_return'] + cols]

    return quarterly_df

def zscore(x):
    return (x - x.mean()) / x.std(ddof=0)

def calc_factors(df:pd.DataFrame):          #**z score at the very end
    df = df.copy()
    
    # ---------------------------
    # 1. Value
    # ---------------------------
    df['bp'] = 1/df['pb_ratio']

    # z score before combining
    df['value_raw'] = 0.4 * df['bp'] + 0.3 * df['earnings_yield'] + 0.3 * df['cf_yield']
    # df['value_z'] = df['value_raw'].transform(zscore) # df.groupby('date')['value_raw'].transform(zscore)
    
    # ---------------------------
    # 2. Quality factor
    # ---------------------------

    df['gpoa'] = (df['revenue'] - df['cogs']) / df['total_assets']
    df['roe'] = df['net_income'] / df['equity']
    df['de_ratio'] = df['liabilities'] / df['equity']
    df = df.sort_values('date').reset_index(drop=True)
    df['evol'] = df['quarterly_return'].rolling(12, min_periods=1).std().shift(1).reset_index(level=0, drop=True)
    # df.groupby('shares')['monthly_return'].rolling(12, min_periods=1).std().shift(1).reset_index(level=0, drop=True)
    
    # df['gpoa_z'] = df.groupby('date')['gpoa'].transform(zscore)
    # df['roe_z'] = df.groupby('date')['roe'].transform(zscore)
    # df['de_z'] = df.groupby('date')['de_ratio'].transform(lambda x: zscore(-x))
    # df['evol_z'] = df.groupby('date')['evol'].transform(lambda x: zscore(-x))
    
    # df['quality_raw'] = df['gpoa_z'] + df['roe_z'] + df['de_z'] + df['evol_z']
    # df['quality_z'] = df.groupby('date')['quality_raw'].transform(zscore)
    
    # ---------------------------
    # 3. Momentum factor
    # ---------------------------
    df['close_1m'] = df['close'].shift(1)
    df['close_12m'] = df['close'].shift(4)
    
    df['momentum_raw'] = (df['close_1m'] - df['close_12m']) / df['close_12m']
    # df['momentum_z'] = df.groupby('date')['momentum_raw'].transform(zscore)
    
    # ---------------------------
    # 4. Size factor
    # ---------------------------
    df['size_raw'] = -np.log(df['market_cap'])
    # df['size_z'] = df.groupby('date')['size_raw'].transform(zscore)

    df = df.drop(columns=['bp'])
    df = df[df['momentum_raw'].notna()].reset_index(drop=True)
    return df

def main():
    api = APIClient(API_KEY)
    file_tickers = load_tickers_from_file(TICKER_LIST_PATH)

    if file_tickers is not None:
        if not file_tickers:
            print(
                f"ERROR: {TICKER_LIST_PATH} exists but no tickers were found.",
                file=sys.stderr,
            )
            return
        tickers = file_tickers
        print(f"Using {len(tickers)} tickers from {TICKER_LIST_PATH}.")
    else:
        try:
            tickers = get_sp500_tickers()
        except Exception as e:
            print(f"ERROR fetching S&P 500 tickers: {e}", file=sys.stderr)
            print(
                "Fallback: set DO_FULL_RUN=False and use DRY_RUN_TICKERS.",
                file=sys.stderr,
            )
            return
        
    
    print(f"Processing {len(tickers)} tickers...")
    all_dfs = []

    # ------------------------------------
    # Uncomment below for testing purposes
    # ------------------------------------

    # tickers = ['AAPL','SFM']

    for i,tkr in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] Adding {tkr}")
            df = run_ticker(api, tkr, START_DATE, END_DATE)
            df = quarterly_agg(df)
            df = calc_factors(df)
            all_dfs.append(df)
        except Exception as e:
            print(f"[WARN] {tkr} failed: {e}")
    all_dfs = pd.concat(all_dfs,ignore_index=True)
    print(f"Done. Intial Calculation Complete")

    all_dfs['value_z'] = all_dfs['value_raw'].transform(zscore)

    all_dfs['gpoa_z'] = all_dfs['gpoa'].transform(zscore)
    all_dfs['roe_z'] = all_dfs['roe'].transform(zscore)
    all_dfs['de_z'] = all_dfs['de_ratio'].transform(lambda x: zscore(-x))
    all_dfs['evol_z'] = all_dfs['evol'].transform(lambda x: zscore(-x))
    
    all_dfs['quality_raw'] = all_dfs['gpoa_z'] + all_dfs['roe_z'] + all_dfs['de_z'] + all_dfs['evol_z']
    all_dfs['quality_z'] = all_dfs['quality_raw'].transform(zscore)

    all_dfs['momentum_z'] = all_dfs['momentum_raw'].transform(zscore)

    all_dfs['size_z'] = all_dfs['size_raw'].transform(zscore)

    all_results = {}

    print(f'Running regression for {len(tickers)}tickers')
    for i,tkr in enumerate(tickers, 1):
        df = all_dfs[all_dfs['ticker'].str.contains(tkr)]
        df = df[['quarterly_return','value_z','quality_z','momentum_z','size_z']].dropna()

        y = df['quarterly_return']
        X = df[['value_z', 'quality_z', 'momentum_z', 'size_z']]

        model = LinearRegression(fit_intercept=True)
        model.fit(X, y) 
        print(f"[{i}/{len(tickers)}] Regression for {tkr} complete")

        results = {
            'intercept': float(model.intercept_),
            'beta_value': float(model.coef_[0]),
            'beta_quality': float(model.coef_[1]),
            'beta_momentum': float(model.coef_[2]),
            'beta_size': float(model.coef_[3]),
            'r_squared': model.score(X, y)
        }

        all_results[tkr] = results

    # ticker = 'AAPL'
    # df = run_ticker(api,ticker,START_DATE, END_DATE)
    # df = quarterly_agg(df)
    # df = calc_factors(df)

    # print(all_dfs.columns.tolist())
    # print(all_dfs.head(20))
    return all_results

if __name__ == "__main__":
    main()




