from datetime import date, timedelta
import requests_cache
import pandas as pd

#Class for getting financial data
#Goal is to use Massive wherever possible
class APIClient:
    #Store credentials and set up requests session with auth headers
    def __init__(self, api_key: str,) -> None:
        self.base_url = "https://api.massive.com"
        self.session = requests_cache.CachedSession(
            cache_name = ".cache/massive",
            expire_after = timedelta(days=3)
        )
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}'
        })

    #Fetch daily OHLCV bars. Returns DataFrame: date, open, high, low, close, volume, vwap
    def get_daily_bars(self, ticker: str, start: date, end: date, adjusted: bool = True) -> pd.DataFrame:
        url = f'{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}'
        params = {'adjusted': str(adjusted).lower(), 'sort': 'asc', 'limit': 50000}

        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if data.get('resultsCount', 0) == 0:
            raise ValueError(f"No daily bars found for {ticker} between {start} and {end}")

        df = pd.DataFrame(data['results'])
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'vw': 'vwap', 't': 'date'})
        df['date'] = pd.to_datetime(df['date'], unit='ms').dt.date

        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'vwap']]


    #Fetch financial ratios (P/E, P/B, ROE, D/E, etc.). Returns dict: ratio name -> value
    def get_ratios(self, ticker: str) -> dict:
        url = f'{self.base_url}/stocks/financials/v1/ratios'
        params = {'ticker': ticker}

        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if data.get('resultsCount', 0) == 0:
            raise ValueError(f"No ratios data found for {ticker}")

        return data['results'][0]

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
