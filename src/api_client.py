from datetime import date
import pandas as pd
import requests

#Class for getting financial data
#Goal is to use Massive wherever possible
class APIClient:
    #Store credentials and set up requests session with auth headers
    def __init__(self, api_key: str, base_url: str = "https://api.massive.com") -> None:
        pass

    #Fetch daily OHLCV bars. Returns DataFrame: date, open, high, low, close, volume, vwap
    def get_daily_bars(self, ticker: str, start: date, end: date, adjusted: bool = True) -> pd.DataFrame:
        pass

    #Fetch financial ratios (P/E, P/B, ROE, D/E, etc.). Returns dict: ratio name -> value
    def get_ratios(self, ticker: str) -> dict:
        pass

    #Fetch income statements (revenue, net income, EPS). Returns DataFrame
    def get_income_statements(self, ticker: str, period: str = "quarterly", limit: int = 20) -> pd.DataFrame:
        pass

    #Fetch balance sheets (assets, liabilities, equity). Returns DataFrame
    def get_balance_sheets(self, ticker: str, period: str = "quarterly", limit: int = 20) -> pd.DataFrame:
        pass

    #Fetch cash flow statements (operating, investing, financing). Returns DataFrame
    def get_cash_flow_statements(self, ticker: str, period: str = "quarterly", limit: int = 20) -> pd.DataFrame:
        pass

    #Fetch treasury yields for a date. Returns dict: tenor -> yield value
    def get_treasury_yields(self, as_of: date) -> dict:
        pass
