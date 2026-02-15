import pandas as pd
import numpy as np
from pathlib import Path

HOLDINGS_PATH = Path('2026S Portfolio Holdings and Analytics.xlsx')
df = pd.read_excel(HOLDINGS_PATH, header=0, sheet_name='Sector Holdings')
df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
ticker_col = 'Ticker/CUSIP'
price_col = 'Current Price'
sectors = []
current_sector = None
processed_data = []
for index, row in df.iterrows():
    ticker_val = row[ticker_col]
    first_col_val = row.iloc[0]
    if pd.isna(ticker_val) and pd.notna(first_col_val):
        current_sector = str(first_col_val).strip()
        continue
        
    if pd.notna(ticker_val) and str(ticker_val).strip() != '':
        if 'Total' in str(ticker_val):
            continue
        clean_row = row.to_dict()
        clean_row['Sector'] = current_sector
        processed_data.append(clean_row)
portfolio_df = pd.DataFrame(processed_data)
portfolio_df.set_index(ticker_col, inplace=True)
cols_to_clean = [
    'Last 1Y Performance', 'Current Price', 'Share Count', 'Value', 
    '% of Porfolio', 'Target % of Portfolio', 'Purchase Price', 
    '% Return to Date', 'Beta', '% Premium Over Thesis', 'Thesis Price Target'
]
#function to make clean all of the N/A or #REF strings on the overall file
def clean_col(data):
    if isinstance(data, str):
        clean_str = data.replace('$', '').replace('%', '').replace(',', '').replace('data', '').strip()
        if clean_str in ['', '-', 'N/A', 'NA', '#REF!']:
            return np.nan
        try:
            if '%' in data:
                return float(clean_str) / 100
            return float(clean_str)
        except ValueError:
            return data
    return data

for col in portfolio_df.columns:
    if col in cols_to_clean:
        portfolio_df[col] = portfolio_df[col].apply(clean_col)      
portfolio_df.dropna(axis=1, how='all', inplace=True)
portfolio_df.to_csv("cleaned_portfolio_dataframe.csv")
def get_portfolio_df():
    return portfolio_df
print(get_portfolio_df())
