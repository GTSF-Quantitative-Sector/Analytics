# get_data_polygon.py
# Polygon-only S&P 500 pipeline (daily since 2020-01-01), one CSV per ticker in ./data

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

import numpy as np
import pandas as pd
import requests
import requests_cache
from dotenv import load_dotenv

# =========================
# Config
# =========================
START_DATE = date(2021, 12, 15)
END_DATE = date(2026, 12, 15)  # or up to today date.today()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FUND_DIR = Path("fundamentals")
FUND_DIR.mkdir(exist_ok=True)


CACHE_DIR = Path(".http_cache")
CACHE_DIR.mkdir(exist_ok=True)
FIN_CACHE_DIR = Path(".cache_financials")
FIN_CACHE_DIR.mkdir(exist_ok=True)
TICKER_CACHE_PATH = Path(".cache_sp500_tickers.json")
TICKER_LIST_PATH = Path("tickers.txt")
SPY_PATH = DATA_DIR / "SPY.csv"

# Rate limit knobs
REQUESTS_PER_MINUTE = 300
SLEEP_BETWEEN_CALLS = max(0.02, 60.0 / REQUESTS_PER_MINUTE)

BASE_V2 = "https://api.polygon.io/v2"
BASE_V3 = "https://api.polygon.io/v3"
BASE_VX = "https://api.polygon.io"


# ****Toggle for testing vs full run (SET DO_FULL_RUN TO TRUE)****
DO_FULL_RUN = True
DRY_RUN_TICKERS = ["AAPL", "MSFT"]

# =========================
# Environment / Session
# =========================
load_dotenv()
API_KEY = '4cR_irLDgivxae1WO4y0Wb30VYxXRkQj' # os.getenv("POLYGON_API_KEY")
if not API_KEY:
    print("ERROR: POLYGON_API_KEY missing. Add it to your .env file.", file=sys.stderr)
    sys.exit(1)

# Cached session: avoid re-hitting Polygon
# 3-day cache expiration (May want to expand)
# SESSION = requests_cache.CachedSession(
#     cache_name=str(CACHE_DIR / "polygon_cache"),
#     backend="sqlite",
#     allowable_methods=("GET",),
#     expire_after=timedelta(days=3),
#     stale_if_error=True,
# )
# SESSION.params = SESSION.params or {}
# SESSION.params.update({"apiKey": API_KEY})


# =========================
# Helpers
# =========================
# def _sleep():
#     time.sleep(SLEEP_BETWEEN_CALLS)

# def _get(
#     url: str, params: Optional[Dict[str, Any]] = None, retries: int = 5
# ) -> Dict[str, Any]:
#     """GET with simple retry/backoff and rate-limit awareness."""
#     params = params or {}
#     backoff = 1.0
#     for attempt in range(1, retries + 1):
#         try:
#             resp = SESSION.get(url, params=params, timeout=30)
#             if not resp.from_cache:
#                 _sleep()
#             if resp.status_code == 429:
#                 time.sleep(backoff)
#                 backoff = min(30.0, backoff * 2.0)
#                 continue
#             resp.raise_for_status()
#             return resp.json()
#         except requests.RequestException as e:
#             if attempt == retries:
#                 raise
#             time.sleep(backoff)
#             backoff = min(30.0, backoff * 2.0)
#     # Should not reach here
#     return {}


# def _iso(d: date) -> str:
#     return d.strftime("%Y-%m-%d")


def _fin_cache_path(ticker: str) -> Path:
    return FIN_CACHE_DIR / f"{ticker}.json"

def extract_growth_fundamentals(fin_json):
    """
    Returns list of dicts:
    date, revenue, net_income, operating_cf, capex, fcf
    """
    rows = []

    for blk in fin_json.get("results", []):
        fin = blk.get("financials", {})
        is_ = fin.get("income_statement", {}) or {}
        cf = fin.get("cash_flow_statement", {}) or {}

        try:
            revenue = is_.get("revenues", {}).get("value")
            net_income = is_.get("net_income_loss", {}).get("value")
            op_cf = cf.get("net_cash_flow_from_operating_activities", {}).get("value")
            capex = cf.get("capital_expenditures", {}).get("value")
        except AttributeError:
            continue

        if revenue is None and net_income is None and op_cf is None:
            continue

        rows.append({
            "period_end": blk.get("end_date"),
            "revenue": revenue,
            "net_income": net_income,
            "operating_cf": op_cf,
            "capex": capex,
            "fcf": (
                op_cf - capex
                if op_cf is not None and capex is not None
                else None
            ),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end")

    return df


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
# Financial Helpers
# =========================
def extract_financial_periods(fin_json):
    """
    Returns a list of dicts:
      {"period_end": date, "equity": float or None, "shares": float or None}
    Accepts multiple key names, and may leave shares None to be filled by roll-forward.
    """
    periods = []
    for blk in fin_json.get("results", []):
        fin = blk.get("financials", {})
        bs = fin.get("balance_sheet", {}) or {}
        is_ = fin.get("income_statement", {}) or {}

        # Equity candidates (first non-null wins)
        equity_candidates = [
            bs.get("equity"),
            bs.get("equity_attributable_to_parent"),
            bs.get("total_shareholders_equity"),
            bs.get("total_equity_gross_minority_interest"),
        ]
        equity = None
        for c in equity_candidates:
            if isinstance(c, dict) and "value" in c and c["value"] is not None:
                equity = float(c["value"])
                break

        # Shares candidates (first valid positive wins)
        shares_candidates = [
            is_.get("common_stock_shares_outstanding"),
            bs.get("common_stock_shares_outstanding"),
            bs.get("common_stock_shares_issued"),
            is_.get("diluted_average_shares"),
            is_.get("basic_average_shares"),
        ]
        shares = None
        for c in shares_candidates:
            if isinstance(c, dict) and "value" in c and c["value"]:
                try:
                    v = float(c["value"])
                    if v > 0:
                        shares = v
                        break
                except Exception:
                    pass

        periods.append(
            {
                "period_end": (
                    blk.get("end_date")
                    or blk.get("period_of_report_date")
                    or blk.get("fiscal_period_end")
                    or blk.get("filing_date")
                ),
                "equity": equity,
                "shares": shares,
            }
        )
    # sort by period_end ascending
    periods.sort(key=lambda d: d["period_end"] or "")
    return periods


def roll_forward_shares(periods):
    """
    Fill missing 'shares' by carrying forward the most recent prior valid shares.
    """
    last = None
    for p in periods:
        if p["shares"] and p["shares"] > 0:
            last = p["shares"]
        elif last is not None:
            p["shares"] = last
    return periods


def enrich_row_with_financials(row, fin_periods):
    # row["date"] is 'YYYY-MM-DD'
    d = row["date"]
    # choose most recent period_end <= d
    candidates = [p for p in fin_periods if p["period_end"] and p["period_end"] <= d]
    if not candidates:
        return row  # leave blanks

    p = candidates[-1]  # last in time

    equity = p.get("equity")
    shares = p.get("shares")

    if equity and shares and shares > 0:
        bvps = equity / shares
        row["pb_ratio"] = (
            row.get("close") / bvps if (row.get("close") and bvps) else None
        )
        row["market_cap"] = (row.get("close") * shares) if row.get("close") else None
        row["fin_period_used"] = p["period_end"]
        row["shares_outstanding_used"] = shares
    else:
        # leave blanks but record the period we tried, if any
        row["fin_period_used"] = p.get("period_end")
    return row


# =========================
# Financials
# =========================
def get_quarterly_financials(api: APIClient, ticker: str) -> pd.DataFrame:

    # Will need to add PB ratio (market cap/book equity), earnings yield (earnings/market cap), cash flow yield (cash flow/market cap), 
    # revenue, cost of goods sold (cost of revenue), total assets, net income, shareholder's equity, debt to equity (liabilities/equity)
    # earning volatilty:
    # earnings_vol = (
    #     fin_df
    #     .groupby("ticker")["net_income"]
    #     .pct_change()
    #     .rolling(8)   # 2 years quarterly
    #     .std()
    # )

    # cache_path = _fin_cache_path(ticker)
    # if cache_path.exists():
    #     raw = json.loads(cache_path.read_text())
    # else:
    #     url = f"{BASE_VX}/vX/reference/financials"
    #     params = {
    #         "ticker": ticker,
    #         "timeframe": "quarterly",
    #         "limit": 100,
    #         "sort": "period_of_report_date",
    #         "order": "asc",
    #     }
    #     raw = _get(url, params=params)
    #     cache_path.write_text(json.dumps(raw))

    # # Existing balance-sheet extraction (PB, shares)
    # periods = extract_financial_periods(raw)
    # periods = roll_forward_shares(periods)

    # bs_rows = []
    # for p in periods:
    #     if p["equity"] and p["shares"]:
    #         bs_rows.append({
    #             "period_end": pd.to_datetime(p["period_end"]),
    #             "equity": p["equity"],
    #             "diluted_shares": p["shares"],
    #             "bvps": p["equity"] / p["shares"],
    #         })

    # bs_df = pd.DataFrame(bs_rows)

    # growth_df = extract_growth_fundamentals(raw)

    # if bs_df.empty:
    #     return growth_df

    # if growth_df.empty:
    #     return bs_df

    # fin = pd.merge(bs_df, growth_df, on="period_end", how="outer")

    # for col in ["bvps", "diluted_shares"]:
    #     if col not in fin.columns:
    #         fin[col] = np.nan

    # return fin.sort_values("period_end").reset_index(drop=True)

    cache_path = _fin_cache_path(ticker)

    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
    else:
        url = f"{BASE_VX}/vX/reference/financials"
        params = {
            "ticker": ticker,
            "timeframe": "quarterly",
            "limit": 100,
            "sort": "period_of_report_date",
            "order": "asc",
        }
        raw = _get(url, params=params)
        cache_path.write_text(json.dumps(raw))

    rows = []

    for blk in raw.get("results", []):
        fin = blk.get("financials", {}) or {}
        is_ = fin.get("income_statement", {}) or {}
        bs = fin.get("balance_sheet", {}) or {}
        cf = fin.get("cash_flow_statement", {}) or {}

        period_end = blk.get("period_of_report_date")
        if not period_end:
            continue

        # -------- Raw fundamentals --------
        revenue = is_.get("revenues", {}).get("value")
        cogs = is_.get("cost_of_revenue", {}).get("value")
        net_income = is_.get("net_income_loss", {}).get("value")

        total_assets = bs.get("assets", {}).get("value")
        equity = bs.get("equity", {}).get("value")
        liabilities = bs.get("liabilities", {}).get("value")
        shares = bs.get("weighted_average_shares_diluted", {}).get("value")

        operating_cf = cf.get(
            "net_cash_flow_from_operating_activities", {}
        ).get("value")

        # -------- Market cap (point-in-time safe) --------
        market_cap = None
        if equity is not None and shares:
            bvps = equity / shares
            market_cap = bvps * shares

        rows.append({
            "ticker": ticker,
            "period_end": pd.to_datetime(period_end),

            # raw
            "market_cap": market_cap,
            "revenue": revenue,
            "cogs": cogs,
            "total_assets": total_assets,
            "net_income": net_income,
            "equity": equity,
            "liabilities": liabilities,
            "operating_cf": operating_cf,
            "shares": shares,

            # derived
            "pb_ratio": (
                market_cap / equity
                if market_cap and equity
                else np.nan
            ),
            "earnings_yield": (
                net_income / market_cap
                if net_income and market_cap
                else np.nan
            ),
            "cf_yield": (
                operating_cf / market_cap
                if operating_cf and market_cap
                else np.nan
            ),
            "liabilities_to_equity": (
                liabilities / equity
                if liabilities and equity
                else np.nan
            ),
        })

    fin_df = pd.DataFrame(rows)

    if fin_df.empty:
        return fin_df

    fin_df = fin_df.sort_values("period_end").reset_index(drop=True)
    return fin_df

def attach_pb_and_mcap(daily: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
#     """
#     For each trading day, attach the most recent prior quarterly filing:
#       PB = close / BVPS
#       MarketCap = close * diluted_shares
#     """
#     out = daily.copy()

#     REQUIRED = {"period_end", "bvps", "diluted_shares"}
#     if fin.empty or not REQUIRED.issubset(fin.columns):
#         out["pb_ratio"] = np.nan
#         out["market_cap"] = np.nan
#         out["fin_period_used"] = pd.NaT
#         out["shares_outstanding_used"] = np.nan
#         return out

#     f = fin.rename(columns={"period_end": "fin_period_end"}).copy()

#     left = out.copy()
#     left["date_dt"] = pd.to_datetime(left["date"])
#     right = f.copy()
#     right["fin_dt"] = pd.to_datetime(right["fin_period_end"])

#     left = left.sort_values("date_dt").set_index("date_dt")
#     right = right.sort_values("fin_dt").set_index("fin_dt")

#     merged = pd.merge_asof(
#         left, right, left_index=True, right_index=True, direction="backward"
#     )

#     merged["pb_ratio"] = np.where(
#         merged["bvps"].notna() & (merged["bvps"] != 0),
#         merged["close"] / merged["bvps"],
#         np.nan,
#     )
#     merged["market_cap"] = merged["close"] * merged["diluted_shares"]

#     fin_end_dt = pd.to_datetime(merged["fin_period_end"], errors="coerce")
#     merged["fin_period_used"] = fin_end_dt.dt.date
#     merged["shares_outstanding_used"] = merged["diluted_shares"]

#     keep = [
#         "date",
#         "open",
#         "close",
#         "high",
#         "low",
#         "volume",
#         "daily_change",
#         "daily_change_pct",
#         "pb_ratio",
#         "market_cap",
#         "fin_period_used",
#         "shares_outstanding_used",
#     ]
#     merged.reset_index(drop=True, inplace=True)
#     return merged[keep]
    """
    For each trading day, attach the most recent prior quarterly filing and compute:
      - Market cap
      - PB ratio
      - Earnings yield
      - Cash flow yield

    Pass through raw quarterly fundamentals unchanged.
    """

    out = daily.copy()

    REQUIRED = {
        "period_end",
        "equity",
        "shares",
        "net_income",
        "operating_cf",
        "liabilities",
    }

    if fin.empty or not REQUIRED.issubset(fin.columns):
        for col in [
            "market_cap",
            "pb_ratio",
            "earnings_yield",
            "cf_yield",
            "liabilities_to_equity",
            "fin_period_used",
            "shares_outstanding_used",
        ]:
            out[col] = np.nan
        return out

    f = fin.rename(columns={"period_end": "fin_period_end"}).copy()

    left = out.copy()
    left["date_dt"] = pd.to_datetime(left["date"])
    right = f.copy()
    right["fin_dt"] = pd.to_datetime(right["fin_period_end"])

    left = left.sort_values("date_dt").set_index("date_dt")
    right = right.sort_values("fin_dt").set_index("fin_dt")

    merged = pd.merge_asof(
        left, right, left_index=True, right_index=True, direction="backward"
    )

    # -----------------------
    # Market cap (daily price × quarterly shares)
    # -----------------------
    merged["market_cap"] = merged["close"] * merged["shares"]

    # -----------------------
    # Price-to-book
    # -----------------------
    merged["pb_ratio"] = np.where(
        merged["equity"].notna() & (merged["equity"] != 0),
        merged["market_cap"] / merged["equity"],
        np.nan,
    )

    # -----------------------
    # Earnings yield
    # -----------------------
    merged["earnings_yield"] = np.where(
        merged["net_income"].notna() & (merged["market_cap"] > 0),
        merged["net_income"] / merged["market_cap"],
        np.nan,
    )

    # -----------------------
    # Cash flow yield
    # -----------------------
    merged["cf_yield"] = np.where(
        merged["operating_cf"].notna() & (merged["market_cap"] > 0),
        merged["operating_cf"] / merged["market_cap"],
        np.nan,
    )

    # -----------------------
    # Leverage
    # -----------------------
    merged["liabilities_to_equity"] = np.where(
        merged["equity"].notna() & (merged["equity"] != 0),
        merged["liabilities"] / merged["equity"],
        np.nan,
    )

    # Metadata
    merged["fin_period_used"] = pd.to_datetime(
        merged["fin_period_end"], errors="coerce"
    ).dt.date
    merged["shares_outstanding_used"] = merged["shares"]

    keep = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "daily_change",
        "daily_change_pct",

        # pricing & valuation
        "market_cap",
        "pb_ratio",
        "earnings_yield",
        "cf_yield",

        # balance-sheet
        "liabilities_to_equity",

        # fundamentals (passed through)
        "revenue",
        "cogs",
        "total_assets",
        "net_income",
        "equity",

        # metadata
        "fin_period_used",
        "shares_outstanding_used",
    ]

    merged.reset_index(drop=True, inplace=True)
    return merged[keep]

def get_spy_data_polygon(api: APIClient, start: date, end: date) -> pd.DataFrame:
    """
    Fetch SPY daily bars from Polygon and save locally.
    Returns a DataFrame with PB ratio, market cap, daily_change, etc. like other tickers.
    """
    
    if SPY_PATH.exists():
        return pd.read_csv(SPY_PATH, parse_dates=["date"])

    prices = get_daily_bars(api, "SPY", start, end)

    try:
        fin = get_quarterly_financials("SPY")
    except Exception:
        fin = pd.DataFrame(columns=["period_end_date", "equity", "diluted_shares", "bvps"])

    final = attach_pb_and_mcap(prices, fin)
    final.to_csv(SPY_PATH, index=False)
    return final


def run_ticker(api: APIClient, ticker: str, start: date, end: date) -> None:
    prices = get_daily_bars(api, ticker, start, end)
    fin = get_quarterly_financials(api, ticker)

    if not fin.empty:
        fin_out = FUND_DIR / f"{ticker}_fundamentals.csv"
        fin.to_csv(fin_out, index=False)

    final = attach_pb_and_mcap(prices, fin)

    out_path = DATA_DIR / f"{ticker}.csv"
    final.to_csv(out_path, index=False)

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
    elif DO_FULL_RUN:
        try:
            tickers = get_sp500_tickers()
        except Exception as e:
            print(f"ERROR fetching S&P 500 tickers: {e}", file=sys.stderr)
            print(
                "Fallback: set DO_FULL_RUN=False and use DRY_RUN_TICKERS.",
                file=sys.stderr,
            )
            return
    else:
        tickers = DRY_RUN_TICKERS

    print(f"Processing {len(tickers)} tickers... (cache active)")
    for i, tkr in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {tkr}")
        run_ticker(api, tkr, START_DATE, END_DATE)
    print(f"Done. CSVs written to: {DATA_DIR.resolve()}")

    spy_df = get_spy_data_polygon(api, START_DATE, END_DATE)
    print(f"SPY data fetched: {len(spy_df)} rows, saved at {SPY_PATH}")



if __name__ == "__main__":
    main()
