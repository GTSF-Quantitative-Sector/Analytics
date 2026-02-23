import pandas as pd
from datetime import date, timedelta
from pathlib import Path

#Builds portfolio historical prices
def build_portfolio(api_client, portfolio, start_date):
    stock_tickers = portfolio.holdings.loc[
        portfolio.holdings['Stock/Bond'] == 'Stock', 'Ticker'
    ]

    today = date.today()
    price_series = {}
    for ticker in stock_tickers:
        try:
            bars = api_client.get_daily_bars(ticker, start_date, today)
            price_series[ticker] = bars.set_index('date')['close']
        except Exception:
            continue

    portfolio.historical_prices = pd.DataFrame(price_series)
    portfolio.historical_prices.index.name = 'date'

#Updates metrics with current data
#Meant to be called once a day after close
#build_portfolio must be called first
def update_portfolio(portfolio, api_client):
    holdings = portfolio.holdings
    today = date.today()
    start = today - timedelta(days=5)

    prices = []
    for _, row in holdings.iterrows():
        ticker = row['Ticker']
        if row['Stock/Bond'] != 'Stock':
            prices.append(None)
            continue

        try:
            bars = api_client.get_daily_bars(ticker, start, today)
            prices.append(bars['close'].iloc[-1])
        except Exception:
            prices.append(None)

    holdings['Price'] = prices
    holdings['Value'] = holdings['Share Count'] * holdings['Price']
    holdings['% Gain to Date'] = (holdings['Price'] - holdings['Purchase Price']) / holdings['Purchase Price'] * 100
    portfolio.holdings = holdings

    if today not in portfolio.historical_prices.index:
        new_row = pd.Series(
            {ticker: price for ticker, price in zip(holdings['Ticker'], prices) if price is not None},
            name=today,
        )
        portfolio.historical_prices = pd.concat([portfolio.historical_prices, new_row.to_frame().T])

#Parses the holdings xlsx file to build the holdings df
def parse_xlsx(path):
    HOLDINGS_PATH = Path(path)
    df = pd.read_excel(HOLDINGS_PATH, header=0, sheet_name='Sector Holdings')
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    ticker_col = 'Ticker/CUSIP'
    current_sector = None
    processed_data = []
    cash_row = None

    for _, row in df.iterrows():
        ticker_val = row[ticker_col]
        first_col_val = row.iloc[0]
        if pd.isna(ticker_val) and pd.notna(first_col_val):
            label = str(first_col_val).strip()

            if 'Total' not in label and 'Strategy' not in label:
                current_sector = label
            continue

        if pd.notna(ticker_val) and str(ticker_val).strip() != '':
            ticker_str = str(ticker_val).strip()

            if 'Total' in ticker_str:
                continue
            if 'Strategy' in ticker_str:
                continue

            value = row.get('Value', None)

            if pd.notna(value) and value == 0:
                continue

            if ticker_str == 'Cash':
                cash_row = {
                    'Ticker': 'Cash',
                    'Description': 'Cash',
                    'Share Count': row.get('Value', None),
                    'Purchase Date': None,
                    'Purchase Price': None,
                    'Sector': current_sector,
                    'Stock/Bond': 'Bond',
                }
                continue

            is_stock = bool(pd.Series([ticker_str]).str.match(r'^[A-Z]{1,5}$').iloc[0])
            processed_data.append({
                'Ticker': ticker_str,
                'Description': row.get('Description', None),
                'Share Count': row.get('Share Count', None),
                'Purchase Date': row.get('Purchase Date', None),
                'Purchase Price': row.get('Purchase Price', None),
                'Sector': current_sector,
                'Stock/Bond': 'Stock' if is_stock else 'Bond',
            })

    if cash_row:
        processed_data.append(cash_row)

    portfolio_df = pd.DataFrame(processed_data)
    return portfolio_df