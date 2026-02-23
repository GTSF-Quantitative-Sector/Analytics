import numpy as np
import pandas as pd
from datetime import date

def run_monte_carlo(self, start: date, end: date, n_simulations: int = 10_000, horizon_days: int = 21, seed: int | None = 42,
    sector: str | None = None,
) -> dict:

    if n_simulations <= 0:
        raise ValueError("n_simulations must be > 0")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0")

    rng = np.random.default_rng(seed)

    # Pull historical daily portfolio returns (preferred: uses teammates' weighting logic)
    port = self.portfolio_returns(start, end, freq="daily", sector=sector)

    # Clean
    port = pd.Series(port).dropna()
    if port.shape[0] < 2:
        raise ValueError("Not enough daily portfolio returns in [start, end] to estimate mu/sigma")

    mu = float(port.mean())
    sigma = float(port.std(ddof=1))
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Computed portfolio return volatility is invalid")

    # Simulate daily returns: shape (n_simulations, horizon_days)
    sim_daily = rng.normal(loc=mu, scale=sigma, size=(n_simulations, horizon_days))

    # Compound to horizon simple return: Π(1+r) - 1
    simulated_returns = np.prod(1.0 + sim_daily, axis=1) - 1.0

    mean = float(np.mean(simulated_returns))
    median = float(np.median(simulated_returns))
    var_95 = float(np.quantile(simulated_returns, 0.05))  # 5% left-tail return
    tail = simulated_returns[simulated_returns <= var_95]
    expected_shortfall = float(np.mean(tail)) if tail.size else var_95

    return {
        "simulated_returns": simulated_returns,
        "mean": mean,
        "median": median,
        "var_95": var_95,
        "expected_shortfall": expected_shortfall,
    }