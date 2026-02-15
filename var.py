import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import norm

def find_var(self, start: date, end: date, confidence: float = 0.95, horizon_days: int = 1, sector: str | None = None) -> dict:
        p_ret = self.portfolio_returns(start, end, sector=sector)
        mu = p_ret.mean()
        sigma = p_ret.std()

        # Parametric var
        z_score = norm.ppf(1 - confidence)
        var = -(mu * horizon_days + z_score * sigma * np.sqrt(horizon_days))
        return {
            "var_pct": var,
            "expected_shortfall": es,
            "confidence": confidence,
            "horizon": horizon_days
        }