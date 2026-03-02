import os
from datetime import date
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from utils import *
from api_client import APIClient
from portfolio import Portfolio
from visualizations import (
    plot_cumulative_returns,
    plot_rolling_volatility,
    plot_sector_rolling_volatility,
    plot_drawdown,
    plot_return_distribution,
    plot_monte_carlo,
    plot_component_risk,
    plot_correlation_heatmap,
    plot_sector_returns,
)

def main():
    load_dotenv()
    api_key = os.getenv("MASSIVE_API_KEY")
    api_client = APIClient(api_key)

    time_frame_years = 5
    portfolio = build_portfolio("data/holdings.xlsx", api_client, time_frame_years)

    start_date = date.today() - relativedelta(years=time_frame_years)
    var = portfolio.find_var(start_date, date.today())
    monte_carlo = portfolio.run_monte_carlo(start_date, date.today(), horizon_days=100)
    marginal_risk = portfolio.find_risk_contribution(start_date, date.today())

    print(marginal_risk)

if __name__ == "__main__":
    main()