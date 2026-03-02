from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

#Portfolio class
#Used to hold stocks and run all analysis
#Passing sector to any method filters to just that sector
class Portfolio:

    #holdings: DataFrame with columns [Ticker, Description, Share Count, Purchase Date, Purchase Price, Sector, Stock/Bond]
    def __init__(self, holdings: pd.DataFrame) -> None:
        self.holdings = holdings

        #Built using API client
        self.historical_prices = None

    #Per-asset return matrix over a date range. Returns DataFrame (dates x tickers)
    def get_returns(self, start: date, end: date, sector: str | None = None) -> pd.DataFrame:
        df = self._sector_filter(sector)
        tickers = [t for t in df['Ticker'] if t in self.historical_prices.columns]
        prices = self.historical_prices[tickers].loc[start:end]
        return prices.pct_change(fill_method=None).iloc[1:].dropna(how='all')

    #Weight-adjusted portfolio return series using dynamic daily weights
    def weighted_portfolio_returns(self, start: date, end: date, sector: str | None = None) -> pd.Series:
        df = self._sector_filter(sector)
        tickers = [t for t in df['Ticker'] if t in self.historical_prices.columns]
        shares = df.set_index('Ticker').loc[tickers, 'Share Count']

        prices = self.historical_prices[tickers].loc[start:end]
        market_values = prices.mul(shares, axis=1)
        daily_weights = market_values.div(market_values.sum(axis=1), axis=0)

        returns = self.get_returns(start, end, sector=sector)
        daily_weights = daily_weights.shift(1).loc[returns.index].dropna(how='all')
        returns = returns.loc[daily_weights.index]
        return (returns[tickers] * daily_weights).sum(axis=1)

    #Covariance matrix of asset returns. method: 'ledoit_wolf' or 'ewma'
    #Returns DataFrame (tickers x tickers)
    def covariance_matrix(self, start: date, end: date, method: str = "lw", ewma_span: int = 33, sector: str | None = None) -> pd.DataFrame:
        if (method not in ["lw", "ewma"]):
            raise ValueError("Method must be ledoit wolf (lw) or exponentially weighted moving average (ewma)")

        returns = self.get_returns(start, end, sector=sector)
        returns = returns.dropna(axis=1, how='all').ffill().bfill()

        tickers = returns.columns

        if method == "lw":
            lw = LedoitWolf().fit(returns.values)
            cov = pd.DataFrame(lw.covariance_, index=tickers, columns=tickers)
        elif method == "ewma":
            ewm_cov = returns.ewm(span=ewma_span).cov(pairwise=True)
            last_date = ewm_cov.index.get_level_values(0)[-1]
            cov = ewm_cov.loc[last_date].copy()
            cov.columns = tickers

        return cov

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

    #Parametric Value at Risk
    #Returns dict: var, dollar_var, expected_shortfall, confidence, method
    def find_var(self, start: date, end: date, confidence: float = 0.95, horizon_days: int = 1, sector: str | None = None) -> dict:
        p_ret = self.weighted_portfolio_returns(start, end, sector)
        mu = p_ret.mean()
        sigma = p_ret.std(ddof=1)

        z_score = norm.ppf(1 - confidence)
        var = -(mu * horizon_days + z_score * sigma * np.sqrt(horizon_days))
        es = -(mu * horizon_days - sigma * np.sqrt(horizon_days) * norm.pdf(z_score) / (1 - confidence))

        return {
            "var_pct": var,
            "expected_shortfall": es,
            "confidence": confidence,
            "horizon": horizon_days
        }

    #Per-position risk contribution. Returns DataFrame indexed by ticker: marginal, component, pct_contribution
    def find_risk_contribution(self, start: date, end: date, cov_method: str = "lw", sector: str | None = None) -> pd.DataFrame:
        cov_matrix = self.covariance_matrix(start, end, method=cov_method, sector=sector)
        tickers = list(cov_matrix.columns)

        holdings = self._sector_filter(sector)
        shares = holdings.set_index('Ticker').loc[tickers, 'Share Count']
        first_prices = self.historical_prices[tickers].loc[start:end].dropna(how='all').iloc[0]

        valid = first_prices.dropna().index
        tickers = [t for t in tickers if t in valid]
        shares = shares.loc[tickers]
        first_prices = first_prices.loc[tickers]
        cov_matrix = cov_matrix.loc[tickers, tickers]

        market_values = shares * first_prices
        w = (market_values / market_values.sum()).values

        cov = cov_matrix.values
        sigma_w = cov @ w
        port_var = w @ sigma_w
        port_std = np.sqrt(port_var)

        mctr = sigma_w / port_std
        cctr = w * mctr
        pct_contribution = cctr / cctr.sum()

        return pd.DataFrame({
            "marginal": mctr,
            "component": cctr,
            "pct_contribution": pct_contribution,
        }, index=tickers)

    #Split portfolio variance into factor risk vs idiosyncratic risk
    #Returns dict: total_var, factor_var, idio_var, pct_factor, pct_idio, per_factor
    def decompose_risk(self, factor_returns: pd.DataFrame, start: date, end: date, sector: str | None = None) -> dict:
        df = self._sector_filter(sector)

        pass

    #Simulate portfolio return paths
    #Returns dict: simulated_returns, mean, median, var_95, expected_shortfall
    def run_monte_carlo(self, start: date, end: date, n_simulations: int = 10_000, horizon_days: int = 21, seed: int | None = 42, sector: str | None = None) -> dict:
        rng = np.random.default_rng(seed)

        port = self.weighted_portfolio_returns(start, end, sector=sector)
        port = pd.Series(port).dropna()

        mu = float(port.mean())
        sigma = float(port.std(ddof=1))

        sim_daily = rng.normal(loc=mu, scale=sigma, size=(n_simulations, horizon_days))
        simulated_returns = np.prod(1.0 + sim_daily, axis=1) - 1.0

        mean = float(np.mean(simulated_returns))
        median = float(np.median(simulated_returns))
        var_95 = float(np.quantile(simulated_returns, 0.05))
        tail = simulated_returns[simulated_returns <= var_95]
        expected_shortfall = float(np.mean(tail)) if tail.size else var_95

        return {
            "simulated_returns": simulated_returns,
            "mean": mean,
            "median": median,
            "var_95": var_95,
            "expected_shortfall": expected_shortfall,
        }

    #We will worry about this later
    def run_scenario(self, factor_returns: pd.DataFrame, shocks: dict[str, float], start: date, end: date) -> dict:
        pass


    #Helper method for applying sector filters when necessary
    def _sector_filter(self, sector: str | None) -> pd.DataFrame:
        if sector is None:
            return self.holdings
        return self.holdings[self.holdings['Sector'] == sector]
