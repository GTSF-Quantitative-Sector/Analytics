from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional
import re

import pandas as pd
import numpy as np
import json

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path(".http_cache")
CACHE_DIR.mkdir(exist_ok=True)
FIN_CACHE_DIR = Path(".cache_financials")
FIN_CACHE_DIR.mkdir(exist_ok=True)
TICKER_CACHE_PATH = Path(".cache_sp500_tickers.json")
TICKER_LIST_PATH = Path("tickers.txt")
SIZE_BREAKPOINTS = [0.0, 50.0, 100.0]
BM_BREAKPOINTS = [0.0, 30.0, 70.0, 100.0]
PORTFOLIO_RETURNS_PATH = Path("portfolio_returns.csv")
FACTOR_RETURNS_PATH = Path("factor_returns.csv")
SPY_PATH = DATA_DIR / "SPY.csv"

def extract_growth_fundamentals(fin_json):
    """
    Returns DataFrame with:
    period_end, revenue, net_income, operating_cf, capex, fcf
    """
    rows = []

    for blk in fin_json.get("results", []):
        fin = blk.get("financials", {}) or {}
        is_ = fin.get("income_statement", {}) or {}
        cf = fin.get("cash_flow_statement", {}) or {}

        try:
            revenue = is_.get("revenues", {}).get("value")
            net_income = is_.get("net_income_loss", {}).get("value")
            operating_cf = (
                cf.get("net_cash_flow_from_operating_activities", {}).get("value")
            )
            capex = cf.get("capital_expenditures", {}).get("value")
        except AttributeError:
            continue

        # Skip rows with no usable fundamentals
        if revenue is None and net_income is None and operating_cf is None:
            continue

        rows.append({
            "period_end": blk.get("end_date"),
            "revenue": revenue,
            "net_income": net_income,
            "operating_cf": operating_cf,
            "capex": capex,
            "fcf": (
                operating_cf - capex
                if operating_cf is not None and capex is not None
                else None
            ),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end").reset_index(drop=True)

    return df


def load_tickers_from_file(path: Path) -> Optional[List[str]]:
    """load the tickers from the tickers.txt file"""
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


def _build_characteristic_labels(
    df: pd.DataFrame,
    value_column: str,
    snapshot_month: int,
    assignment_year_offset: int,
    label_name: str,
    percentiles: List[float],
    positive_only: bool = True,
    breakpoint_universe: Optional[set[str]] = None,
) -> pd.DataFrame:
    snapshots = df[df["month"] == snapshot_month].copy()
    snapshots["formation_year"] = snapshots["date"].dt.year + assignment_year_offset
    snapshots = snapshots.dropna(subset=[value_column])
    if positive_only:
        snapshots = snapshots[snapshots[value_column] > 0]

    records: List[pd.DataFrame] = []
    bucket_count = len(percentiles) - 1
    if bucket_count <= 0:
        return pd.DataFrame(columns=["ticker", "formation_year", label_name])

    for year, group in snapshots.groupby("formation_year"):
        if group.empty:
            continue

        base = group
        if breakpoint_universe is not None:
            base = group[group["ticker"].isin(breakpoint_universe)]
        values = base[value_column].dropna()
        if positive_only:
            values = values[values > 0]
        if values.size < bucket_count:
            continue

        with np.errstate(invalid="ignore"):
            bins = np.nanpercentile(values.to_numpy(), percentiles)
        bins = pd.to_numeric(bins, errors="coerce")
        if np.isnan(bins).any():
            continue

        bins = bins.astype(float)
        bins[0] = -float("inf")
        bins[-1] = float("inf")
        for idx in range(1, len(bins)):
            if bins[idx] <= bins[idx - 1]:
                bins[idx] = bins[idx - 1] + 1e-9

        eligible_mask = (
            group[value_column] > 0 if positive_only else group[value_column].notna()
        )
        eligible_idx = group.index[eligible_mask]

        assigned = pd.Series(pd.NA, index=group.index, dtype="Int64")
        bin_labels = pd.cut(
            group.loc[eligible_idx, value_column],
            bins=bins,
            labels=False,
            include_lowest=True,
        )
        assigned.loc[eligible_idx] = pd.Series(bin_labels, index=eligible_idx).astype(
            "Int64"
        )

        records.append(
            pd.DataFrame(
                {
                    "ticker": group["ticker"],
                    "formation_year": year,
                    label_name: assigned,
                }
            )
        )

    if not records:
        return pd.DataFrame(columns=["ticker", "formation_year", label_name])

    result = pd.concat(records, ignore_index=True)
    return result.dropna(subset=[label_name])


def get_spy_monthly_returns() -> pd.DataFrame:
    if not SPY_PATH.exists():
        print(f"SPY data not found at {SPY_PATH}. Run get_data_polygon.py first.", file=sys.stderr)
        return pd.DataFrame()

    df = pd.read_csv(SPY_PATH, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    
    monthly_first_close = df["close"].resample("ME").first()
    monthly_last_close = df["close"].resample("ME").last()
    monthly_return = (monthly_last_close - monthly_first_close) / monthly_first_close
    monthly_return = monthly_return.replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame({"date": monthly_return.index, "MKT": monthly_return.values})

def _compute_mom_growth_factors(monthly_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute momentum and growth factor returns: value-weighted returns of top-minus-bottom buckets.
    Assumes `mom_bucket` and `growth_bucket` are already assigned.
    """
    if "mom_bucket" not in monthly_df or "growth_bucket" not in monthly_df:
        return None

    if monthly_df["mom_bucket"].nunique() < 2 or monthly_df["growth_bucket"].nunique() < 2:
        return None

    monthly_df["weight"] = monthly_df["ME"].clip(lower=0)
    monthly_df["weighted_return"] = monthly_df["monthly_return"] * monthly_df["weight"]

    mom_agg = (
        monthly_df.groupby(["date", "mom_bucket"], as_index=False)
        .agg(weight_sum=("weight", "sum"), weighted_return_sum=("weighted_return", "sum"))
    )

    top_mom = mom_agg[mom_agg["mom_bucket"] == mom_agg["mom_bucket"].max()]
    bottom_mom = mom_agg[mom_agg["mom_bucket"] == mom_agg["mom_bucket"].min()]

    mom_factor = pd.merge(top_mom, bottom_mom, on="date", suffixes=("_top", "_bot"))
    mom_factor["mom_factor"] = np.where(
        (mom_factor["weight_sum_top"] > 0) & (mom_factor["weight_sum_bot"] > 0),
        mom_factor["weighted_return_sum_top"] / mom_factor["weight_sum_top"] -
        mom_factor["weighted_return_sum_bot"] / mom_factor["weight_sum_bot"],
        np.nan
    )

    growth_agg = (
        monthly_df.groupby(["date", "growth_bucket"], as_index=False)
        .agg(weight_sum=("weight", "sum"), weighted_return_sum=("weighted_return", "sum"))
    )

    top_growth = growth_agg[growth_agg["growth_bucket"] == growth_agg["growth_bucket"].max()]
    bottom_growth = growth_agg[growth_agg["growth_bucket"] == growth_agg["growth_bucket"].min()]

    growth_factor = pd.merge(top_growth, bottom_growth, on="date", suffixes=("_top", "_bot"))
    growth_factor["growth_factor"] = np.where(
        (growth_factor["weight_sum_top"] > 0) & (growth_factor["weight_sum_bot"] > 0),
        growth_factor["weighted_return_sum_top"] / growth_factor["weight_sum_top"] -
        growth_factor["weighted_return_sum_bot"] / growth_factor["weight_sum_bot"],
        np.nan
    )

    factors = pd.DataFrame({
        "date": mom_factor["date"],
        "mom_factor": mom_factor["mom_factor"],
        "growth_factor": growth_factor["growth_factor"]
    }).set_index("date")

    return factors



def _compute_ff_factors(portfolio_returns: pd.DataFrame) -> Optional[pd.DataFrame]:
    required_size_levels = {0, 1}
    required_bm_levels = {0, 1, 2}

    size_levels = set(portfolio_returns["size_bucket"].dropna().unique())
    bm_levels = set(portfolio_returns["bm_bucket"].dropna().unique())

    if not required_size_levels.issubset(size_levels) or not required_bm_levels.issubset(bm_levels):
        return None

    pivot = portfolio_returns.pivot_table(
        index="date",
        columns=["size_bucket", "bm_bucket"],
        values="value_weighted_return",
    )

    try:
        smb = (
            pivot[(0, 0)] + pivot[(0, 1)] + pivot[(0, 2)]
            - pivot[(1, 0)] - pivot[(1, 1)] - pivot[(1, 2)]
        ) / 3.0
        hml = (pivot[(0, 2)] + pivot[(1, 2)]) / 2.0 - ((pivot[(0, 0)] + pivot[(1, 0)]) / 2.0)
    except KeyError:
        return None

    factor_df = pd.DataFrame({"SMB": smb, "HML": hml})

    spy_df = get_spy_monthly_returns()
    if not spy_df.empty:
        factor_df = factor_df.reset_index().merge(spy_df, on="date", how="left").set_index("date")

    return factor_df.dropna()


def main():
    tickers_path = Path("tickers.txt")
    tickers = load_tickers_from_file(tickers_path)
    if not tickers:
        print("No tickers available for processing", file=sys.stderr)
        return

    monthly_frames: List[pd.DataFrame] = []

    for ticker in tickers:
        file_path = Path(f"data/{ticker}.csv")
        if not file_path.exists():
            print(f"missing price file for {ticker}", file=sys.stderr)
            continue

        df = pd.read_csv(file_path)
        if "date" not in df:
            print(f"price file for {ticker} lacks date column", file=sys.stderr)
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")

        monthly_first_close = df["close"].resample("ME").first()
        monthly_last_close = df["close"].resample("ME").last()
        monthly_return = (
            monthly_last_close - monthly_first_close
        ) / monthly_first_close
        monthly_return = monthly_return.replace([np.inf, -np.inf], np.nan)

        monthly_me = (df["close"] * df["shares_outstanding_used"]).resample("ME").last()
        monthly_be = (df["market_cap"] / df["pb_ratio"]).resample("ME").last()

        monthly_frame = pd.DataFrame(
            {
                "date": monthly_return.index,
                "ticker": ticker,
                "monthly_return": monthly_return.values,
                "ME": monthly_me.values,
                "BE": monthly_be.values,
            }
        )
        monthly_frame["B/M"] = monthly_frame["BE"] / monthly_frame["ME"]
        monthly_frames.append(monthly_frame)

    if not monthly_frames:
        print("No monthly data constructed", file=sys.stderr)
        return
    
    fundamental_frames = []

    for ticker in tickers:
        fin_path = FIN_CACHE_DIR / f"{ticker}.json"
        if not fin_path.exists():
            continue

        fin_raw = json.loads(fin_path.read_text())
        fin_df = extract_growth_fundamentals(fin_raw)
        if fin_df.empty:
            continue

        fin_df["ticker"] = ticker
        fundamental_frames.append(fin_df)

    fund_df = pd.concat(fundamental_frames, ignore_index=True)
    fund_df = fund_df.sort_values(["ticker", "period_end"])

    for col in ["revenue", "net_income", "fcf"]:
        fund_df[f"{col}_growth"] = (
            fund_df
            .groupby("ticker")[col]
            .pct_change(4)
        )

    fund_df["growth_score"] = fund_df[
        ["revenue_growth", "net_income_growth", "fcf_growth"]
    ].mean(axis=1)

    fund_df = fund_df.rename(columns={"period_end": "fin_date"})

    monthly_df = pd.concat(monthly_frames, ignore_index=True)
    monthly_df = monthly_df.dropna(subset=["date"]).reset_index(drop=True)

    monthly_df["date"] = pd.to_datetime(monthly_df["date"])
    monthly_df["year"] = monthly_df["date"].dt.year
    monthly_df["month"] = monthly_df["date"].dt.month
    monthly_df["formation_year"] = monthly_df.apply(
        lambda row: row["year"] if row["month"] >= 7 else row["year"] - 1,
        axis=1,
    )

    monthly_df["mom_6m"] = (
        monthly_df
        .groupby("ticker")["monthly_return"]
        .transform(lambda x: x.rolling(6).sum())
    )

    monthly_df = monthly_df.sort_values("date")

    monthly_df = pd.merge_asof(
        monthly_df.sort_values("date"),
        fund_df.sort_values("fin_date"),
        left_on="date",
        right_on="fin_date",
        by="ticker",
        direction="backward",
    )

    QUINTILES = [0, 20, 40, 60, 80, 100]

    mom_labels = _build_characteristic_labels(
        monthly_df,
        value_column="mom_6m",
        snapshot_month=6,
        assignment_year_offset=0,
        label_name="mom_bucket",
        percentiles=QUINTILES,
        positive_only=False,
    )

    growth_labels = _build_characteristic_labels(
        monthly_df,
        value_column="growth_score",
        snapshot_month=12,
        assignment_year_offset=1,
        label_name="growth_bucket",
        percentiles=QUINTILES,
        positive_only=False,
    )

    monthly_df = monthly_df.merge(
        mom_labels,
        how="left",
        on=["ticker", "formation_year"],
    )

    monthly_df = monthly_df.merge(
        growth_labels,
        how="left",
        on=["ticker", "formation_year"],
    )

    breakpoint_universe = tickers = load_tickers_from_file(Path("tickers.txt"))

    size_labels = _build_characteristic_labels(
        monthly_df,
        value_column="ME",
        snapshot_month=6,
        assignment_year_offset=0,
        label_name="size_bucket",
        percentiles=SIZE_BREAKPOINTS,
        positive_only=True,
        breakpoint_universe=breakpoint_universe,
    )

    bm_labels = _build_characteristic_labels(
        monthly_df,
        value_column="B/M",
        snapshot_month=12,
        assignment_year_offset=1,
        label_name="bm_bucket",
        percentiles=BM_BREAKPOINTS,
        positive_only=True,
        breakpoint_universe=breakpoint_universe,
    )

    monthly_df = monthly_df.merge(
        size_labels,
        how="left",
        on=["ticker", "formation_year"],
    )
    monthly_df = monthly_df.merge(
        bm_labels,
        how="left",
        on=["ticker", "formation_year"],
    )

    valid_portfolios = monthly_df.dropna(
        subset=["monthly_return", "ME", "size_bucket", "bm_bucket"]
    ).copy()

    if valid_portfolios.empty:
        print("No valid portfolio assignments available", file=sys.stderr)
        return

    valid_portfolios["weight"] = valid_portfolios["ME"].clip(lower=0)
    valid_portfolios["weighted_return"] = (
        valid_portfolios["monthly_return"] * valid_portfolios["weight"]
    )

    aggregated = valid_portfolios.groupby(
        ["date", "size_bucket", "bm_bucket"], as_index=False
    ).agg(
        weight_sum=("weight", "sum"),
        weighted_return_sum=("weighted_return", "sum"),
        equal_weight_return=("monthly_return", "mean"),
    )

    aggregated["value_weighted_return"] = np.where(
        aggregated["weight_sum"] > 0,
        aggregated["weighted_return_sum"] / aggregated["weight_sum"],
        aggregated["equal_weight_return"],
    )

    portfolio_returns = (
        aggregated[
            [
                "date",
                "size_bucket",
                "bm_bucket",
                "value_weighted_return",
            ]
        ]
        .sort_values(["date", "size_bucket", "bm_bucket"])
        .reset_index(drop=True)
    )

    portfolio_returns.to_csv(PORTFOLIO_RETURNS_PATH, index=False)

    # factor_df = _compute_ff_factors(portfolio_returns)
    factor_df = _compute_mom_growth_factors(monthly_df)

    if factor_df is not None:
        factor_df.to_csv(FACTOR_RETURNS_PATH, index=True)
        print(factor_df.tail(12))
    else:
        print(
            "Insufficient data to compute momentum/growth factors; check bucket coverage.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
