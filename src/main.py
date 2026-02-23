import os
from datetime import date
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from utils import *
from api_client import APIClient
from portfolio import Portfolio

def main():
    load_dotenv()
    api_key = os.getenv("MASSIVE_API_KEY")
    api_client = APIClient(api_key)

    holdings_df = parse_xlsx("data/holdings.xlsx")
    portfolio = Portfolio(holdings_df)

    time_frame_years = 1
    start_date = date.today() - relativedelta(years=time_frame_years)
    build_portfolio(api_client, portfolio, start_date)
    update_portfolio(portfolio, api_client)

    var = portfolio.find_var(start_date, date.today())

    # portfolio.holdings.to_csv('data/holdings_output.csv')
    # portfolio.historical_prices.to_csv('data/historical_prices_output.csv')

if __name__ == "__main__":
    main()