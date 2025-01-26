import numpy as np
import pandas as pd
from typing import Union, List, Optional
from .utils import validate_input
import statsmodels.api as sm
import scipy.stats as stats

class CausalityAnalyzer:
    def __init__(self, data: Union[np.ndarray, pd.DataFrame, List[np.ndarray]]):
        """
        Initialize CausalityAnalyzer with input time series
        
        Args:
            data: Input time series data
        """
        self.data = validate_input(data)
        
        # Detect number of time series
        if len(self.data.shape) == 1:
            self.num_series = 1
            self.data = self.data.reshape(-1, 1)
        else:
            self.num_series = self.data.shape[1]
    
    def granger_causality(self, lag: int = 1, verbose: bool = False) -> dict:
        """
        Perform Granger Causality test
        
        Args:
            lag: Number of lags to use
            verbose: Whether to print detailed results
        
        Returns:
            Dictionary of Granger Causality test results
        """
        results = {}
        
        # Univariate case
        if self.num_series == 1:
            return {"error": "Granger Causality requires multiple time series"}
        
        # Multivariate Granger Causality
        for i in range(self.num_series):
            for j in range(self.num_series):
                if i != j:
                    # Prepare data
                    x = self.data[:, i]
                    y = self.data[:, j]
                    
                    # Model without X
                    model_restricted = sm.OLS(y[lag:], sm.add_constant(y[:-lag])).fit()
                    
                    # Model with X
                    X_extended = np.column_stack([y[:-lag], x[:-lag]])
                    model_unrestricted = sm.OLS(y[lag:], sm.add_constant(X_extended)).fit()
                    
                    # Calculate F-statistic
                    f_statistic = (model_restricted.ssr - model_unrestricted.ssr) / model_unrestricted.ssr
                    p_value = 1 - stats.f.cdf(f_statistic, 1, len(x) - 2*lag - 1)
                    
                    results[f"{j} → {i}"] = {
                        "f_statistic": f_statistic,
                        "p_value": p_value
                    }
                    
                    if verbose:
                        print(f"Granger Causality Test from Series {j} to Series {i}:")
                        print(f"F-statistic: {f_statistic}")
                        print(f"p-value: {p_value}\n")
        
        return results
    
    def causality_test(self, method: str = 'granger', lag: Optional[int] = None, verbose: bool = False) -> dict:
        """
        Perform causality test based on selected method
        
        Args:
            method: Causality test method ('granger')
            lag: Number of lags (default: 1)
            verbose: Whether to print detailed results
        
        Returns:
            Dictionary of causality test results
        """
        # Use default lag of 1 if not specified
        if lag is None:
            lag = 1
        
        # Select and run appropriate causality test
        methods = {
            'granger': self.granger_causality
        }
        
        if method.lower() not in methods:
            raise ValueError(f"Method {method} not supported. Choose from {list(methods.keys())}")
        
        return methods[method.lower()](lag=lag, verbose=verbose)
