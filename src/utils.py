import pandas as pd
from pathlib import Path

def build_portfolio(api_client, holdings, start_date, end_date):
    pass

def parse_xls(path):
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