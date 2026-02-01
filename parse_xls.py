import pandas as pd
from pathlib import Path

HOLDINGS_PATH = Path('2026S Portfolio Holdings and Analytics.xlsx')

df = pd.read_excel(HOLDINGS_PATH, header=0, sheet_name='Sector Holdings')
tickers = df["Ticker/CUSIP"].dropna().astype(str).str.strip()
tickers = tickers[tickers!=""]
stock_tickers = sorted(
    tickers[tickers.str.match(r"^[A-Z]{1,5}$")].unique()
)
bond_cusips = sorted(
    tickers[tickers.str.match(r"^[A-Z0-9]{9}$")].unique()
)
