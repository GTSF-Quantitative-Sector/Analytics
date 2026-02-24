def find_marginal_risk(self, start: date, end: date, cov_method: str = "ledoit_wolf", sector: str | None = None) -> pd.DataFrame:
       
        df = self._sector_filter(sector)
        data_slice = df.loc[start:end]
        returns = np.log(data_slice / data_slice.shift(1)).dropna()

        if cov_method == "ledoit_wolf":
            lw = LedoitWolf().fit(returns)
            cov_matrix = lw.covariance_
        else:
            cov_matrix = returns.cov().values
            
        num_assets = returns.shape[1]
        w = np.array([1 / num_assets] * num_assets)
        
        port_variance = np.dot(w.T, np.dot(cov_matrix, w))
        port_std = np.sqrt(port_variance)
        
        mctr_values = w * np.dot(cov_matrix, w) / port_std  # weighted by position size
        
        mctr_df = pd.DataFrame({
            "ticker": returns.columns,
            "mctr": mctr_values
        }).set_index("ticker")
        
        return mctr_df