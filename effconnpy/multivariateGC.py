import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import scipy.stats as stats
from .utils import validate_input

class MultivariateGrangerCausality:
    def __init__(self, data: Union[np.ndarray, pd.DataFrame, List[np.ndarray]]):
        self.data = validate_input(data)
        
        if len(self.data.shape) == 1:
            self.num_series = 1
            self.data = self.data.reshape(-1, 1)
        else:
            self.num_series = self.data.shape[1]
        
        self.df = pd.DataFrame(self.data)
    
    def _compute_restricted_ssr(self, var_results, target, predictor, max_lag):
        """
        Compute Sum of Squared Residuals for restricted model
        """
        # Get the data
        data = var_results.model.endog
        
        # Create lagged data matrix
        Z = np.zeros((len(data) - max_lag, max_lag * self.num_series))
        for i in range(max_lag):
            Z[:, i*self.num_series:(i+1)*self.num_series] = data[max_lag-i-1:-i-1]
        
        # Add constant
        Z = sm.add_constant(Z)
        
        # Get target variable
        y = data[max_lag:, target]
        
        # Create mask for restricted model (excluding predictor lags)
        mask = np.ones(Z.shape[1], dtype=bool)
        for i in range(max_lag):
            mask[1 + predictor + i*self.num_series] = False
        
        # Fit restricted model
        Z_restricted = Z[:, mask]
        beta_restricted = np.linalg.lstsq(Z_restricted, y, rcond=None)[0]
        
        # Compute residuals
        resid = y - Z_restricted @ beta_restricted
        
        return np.sum(resid**2)

    def multivariate_granger_causality(self, max_lag: int = 1, verbose: bool = False) -> Dict:
        """
        Compute Granger causality between all pairs of time series.
        
        Args:
            max_lag: Maximum number of lags to include
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with results in format: {'source → target': {'p_value': p, 'f_statistic': f}}
        """
        if self.num_series == 1:
            return {"error": "Granger Causality requires multiple time series"}
        
        causality_results = {}
        
        # Fit VAR model
        var_model = VAR(self.df)
        var_results = var_model.fit(maxlags=max_lag)
        
        # Get unrestricted SSR for each target
        unrestricted_ssr = {}
        for target in range(self.num_series):
            # Convert residuals to numpy array if they're not already
            resid = var_results.resid.values[:, target] if isinstance(var_results.resid, pd.DataFrame) else var_results.resid[:, target]
            unrestricted_ssr[target] = np.sum(resid**2)
        
        # Perform Granger causality test for each pair of variables
        for target in range(self.num_series):
            for predictor in range(self.num_series):
                if target != predictor:
                    # Get restricted SSR
                    restricted_ssr = self._compute_restricted_ssr(var_results, target, predictor, max_lag)
                    
                    # Calculate F-statistic
                    T = len(var_results.resid)
                    n_params = max_lag
                    
                    f_stat = ((restricted_ssr - unrestricted_ssr[target]) / n_params) / (unrestricted_ssr[target] / (T - self.num_series * max_lag - 1))
                    
                    # Calculate p-value
                    p_value = 1 - stats.f.cdf(f_stat, n_params, T - self.num_series * max_lag - 1)
                    
                    # Store results in format compatible with create_connectivity_matrix
                    key = f"{predictor} → {target}"
                    result = {
                        "f_statistic": f_stat,
                        "p_value": p_value
                    }
                    
                    causality_results[key] = result
                    
                    if verbose:
                        print(f"Granger Causality from Series {predictor} to Series {target}:")
                        print(f"F-statistic: {f_stat}")
                        print(f"p-value: {p_value}")
                        print(f"Significant: {p_value < 0.05}\n")
        
        return causality_results

    def get_connectivity_matrix(self, max_lag: int = 1, threshold: float = 0.05) -> np.ndarray:
        """
        Compute and return the connectivity matrix directly.
        
        Args:
            max_lag: Maximum number of lags to include
            threshold: P-value threshold for significance
            
        Returns:
            numpy array: Connectivity matrix where entry [i,j] represents causality from i to j
        """
        results = self.multivariate_granger_causality(max_lag=max_lag)
        
        # Initialize connectivity matrix
        connectivity_matrix = np.zeros((self.num_series, self.num_series))
        
        # Fill matrix based on results
        for connection, value in results.items():
            if isinstance(connection, str) and '→' in connection:
                source, target = map(int, connection.split(' → '))
                connectivity_matrix[source, target] = 1 if value['p_value'] < threshold else 0
        
        return connectivity_matrix
