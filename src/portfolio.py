from datetime import date
import numpy as np
import pandas as pd

#Portfolio class
#Used to hold stocks and run all analysis
#holdings df has columns: ticker, shares, sector
#Passing sector to any method filters to just that sector
class Portfolio:

    #holdings: DataFrame with columns [Ticker, Description, Share Count, Purchase Date, Purchase Price, Sector, Stock/Bond]
    def __init__(self, holdings: pd.DataFrame) -> None:
        self.holdings = holdings

    #Per-asset return matrix over a date range. Returns DataFrame (dates x tickers)
    def get_returns(self, start: date, end: date, freq: str = "daily", sector: str | None = None) -> pd.DataFrame:
        df = self._sector_filter(sector)

        pass

    #Weight-adjusted portfolio return series
    def portfolio_returns(self, start: date, end: date, freq: str = "daily", sector: str | None = None) -> pd.Series:
        df = self._sector_filter(sector)

        pass

    #Covariance matrix of asset returns. method: 'sample', 'ledoit_wolf', or 'ewma'
    def covariance_matrix(self, start: date, end: date, method: str = "ledoit_wolf", sector: str | None = None) -> pd.DataFrame:
        df = self._sector_filter(sector)

        pass

    #Score each holding on Quality, Value, Growth, Momentum
    #ratios: {ticker: {pe, pb, roe, ...}}. price_history: daily prices for momentum
    #Returns DataFrame (tickers x factors)
    def calculate_factors(self, ratios: dict[str, dict], price_history: pd.DataFrame, fundamentals: dict | None = None, sector: str | None = None) -> pd.DataFrame:
        df = self._sector_filter(sector)

        pass

    #OLS regression: r - rf = alpha + b1*Q + b2*V + b3*G + b4*M + eps
    #Returns dict: alpha, betas, t_stats, p_values, r_squared
    def fit_factor_model(self, factor_returns: pd.DataFrame, start: date, end: date, sector: str | None = None) -> dict:
        df = self._sector_filter(sector)

        pass

    #Per-stock and aggregate factor betas. Returns DataFrame (tickers + 'Portfolio' x factors)
    def factor_exposure(self, factor_returns: pd.DataFrame, start: date, end: date, sector: str | None = None) -> pd.DataFrame:
        df = self._sector_filter(sector)

        pass

    #Decompose portfolio return into factor contributions + alpha
    #This is important for alpha attribution. That is the main thing we want to get from this
    #Returns dict: {factor: contribution, 'alpha': float, 'residual': float, 'total': float}
    def attribute_returns(self, factor_returns: pd.DataFrame, start: date, end: date, sector: str | None = None) -> dict:
        df = self._sector_filter(sector)

        pass

    #Value at Risk. Let's use parametric method for now
    #Returns dict: var, dollar_var, expected_shortfall, confidence, method
    def find_var(self, start: date, end: date, confidence: float = 0.95, horizon_days: int = 1, sector: str | None = None) -> dict:
        df = self._sector_filter(sector)

        pass

    #Per-position risk contribution. Returns DataFrame: marginal, component, pct_contribution
    def find_marginal_risk(self, start: date, end: date, cov_method: str = "ledoit_wolf", sector: str | None = None) -> pd.DataFrame:
        df = self._sector_filter(sector)

        pass

    #Split portfolio variance into factor risk vs idiosyncratic risk
    #Returns dict: total_var, factor_var, idio_var, pct_factor, pct_idio, per_factor
    def decompose_risk(self, factor_returns: pd.DataFrame, start: date, end: date, sector: str | None = None) -> dict:
        df = self._sector_filter(sector)

        pass

    #Simulate portfolio return paths
    #Returns dict: simulated_returns, mean, median, var_95, expected_shortfall
    def run_monte_carlo(self, start: date, end: date, n_simulations: int = 10_000, horizon_days: int = 21, seed: int | None = 42, sector: str | None = None) -> dict:
        df = self._sector_filter(sector)

        pass

    #We will worry about this later
    def run_scenario(self, factor_returns: pd.DataFrame, shocks: dict[str, float], start: date, end: date) -> dict:
        pass


    #Helper method for applying sector filters when necessary
    def _sector_filter(self, sector: str | None) -> pd.DataFrame:
        if sector is None:
            return self.holdings
        return self.holdings[self.holdings['Sector'] == sector]
