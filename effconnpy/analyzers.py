import numpy as np
import pandas as pd
from typing import Union, List, Optional
from .utils import validate_input
import statsmodels.api as sm
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from copent import transent as te

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
        
        # Bivariate Granger Causality
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

    def transfer_entropy(self, lag: int = 1, verbose: bool = False) -> dict:
        """
        Perform Transfer Entropy test using copent library with time delay
        
        Args:
            lag: Number of time steps to delay the source series
            verbose: Whether to print detailed results
        
        Returns:
            Dictionary of Transfer Entropy test results
        """
        results = {}
        
        # Univariate case
        if self.num_series == 1:
            return {"error": "Transfer Entropy requires multiple time series"}
        
        # Bivariate Transfer Entropy
        for i in range(self.num_series):
            for j in range(self.num_series):
                if i != j:
                    # Create time-shifted version of the source series
                    source = self.data[:-lag, j]  # Earlier values
                    target = self.data[lag:, i]   # Later values
                    
                    # Calculate transfer entropy with the shifted data
                    entropy_value = te(source, target)
                    
                    results[f"{j} → {i}"] = entropy_value
                    
                    if verbose:
                        print(f"Transfer Entropy from Series {j} to Series {i}: {entropy_value}\n")
        
        return results
    
    def convergent_cross_mapping(self, lag: int = 1, verbose: bool = False) -> dict:
        """
        Perform Convergent Cross Mapping 
        
        Args:
            lag: Number of lags to use
            verbose: Whether to print detailed results
        
        Returns:
            Dictionary of Convergent Cross Mapping results
        """
        results = {}
        
        # Univariate case
        if self.num_series == 1:
            return {"error": "Convergent Cross Mapping requires multiple time series"}
        
        # Bivariate Cross Mapping
        for i in range(self.num_series):
            for j in range(self.num_series):
                if i != j:
                    # Embed time series
                    def embed(x, lag):
                        n = len(x)
                        return np.column_stack([x[k:-lag+k] for k in range(lag)])
                    
                    # Embed both series
                    x_embed = embed(self.data[:, i], lag)
                    y_embed = embed(self.data[:, j], lag)
                    
                    # Calculate distances
                    x_dist = squareform(pdist(x_embed))
                    y_dist = squareform(pdist(y_embed))
                    
                    # Calculate cross mapping skill
                    skill = np.corrcoef(x_dist.flatten(), y_dist.flatten())[0, 1]
                    
                    results[f"{j} → {i}"] = skill
                    
                    if verbose:
                        print(f"Convergent Cross Mapping from Series {j} to Series {i}: {skill}\n")
        
        return results
    
    def causality_test(self, method: str = 'granger', lag: Optional[int] = None, verbose: bool = False) -> dict:
        """
        Perform causality test based on selected method
        
        Args:
            method: Causality test method ('granger', 'transfer_entropy', 'ccm')
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
            'granger': self.granger_causality,
            'transfer_entropy': self.transfer_entropy,
            'ccm': self.convergent_cross_mapping
        }
        
        if method.lower() not in methods:
            raise ValueError(f"Method {method} not supported. Choose from {list(methods.keys())}")
        
        return methods[method.lower()](lag=lag, verbose=verbose)

def create_connectivity_matrix(results: dict, method: str = 'granger', threshold: float = None) -> np.ndarray:
    """
    Create a connectivity matrix from causality analysis results.
    
    Args:
        results: Dictionary of causality analysis results
        method: Type of causality analysis ('granger', 'te', or 'ccm')
        threshold: Threshold value for determining connections
            - For Granger: p-value threshold (default=0.05)
            - For Transfer Entropy: minimum TE value (default=0.1)
            - For CCM: minimum correlation threshold (default=0.5)
    
    Returns:
        numpy array: Connectivity matrix where entry [i,j] represents causality from i to j
    """
    # Set default thresholds based on method
    if threshold is None:
        thresholds = {
            'granger': 0.05,  # Standard statistical significance
            'te': 0.1,       # Common threshold for transfer entropy
            'ccm': 0.5       # Moderate correlation threshold
        }
        threshold = thresholds.get(method.lower(), 0.05)
    
    # Get number of nodes from the results
    nodes = set()
    for key in results.keys():
        if isinstance(key, str) and '→' in key:
            source, target = map(int, key.split(' → '))
            nodes.add(source)
            nodes.add(target)
    n_nodes = len(nodes)
    
    # Initialize connectivity matrix
    connectivity_matrix = np.zeros((n_nodes, n_nodes))
    
    # Fill matrix based on method
    for connection, value in results.items():
        if isinstance(connection, str) and '→' in connection:
            source, target = map(int, connection.split(' → '))
            
            if method.lower() == 'granger':
                # For Granger: check p-value against threshold
                connectivity_matrix[source, target] = 1 if value['p_value'] < threshold else 0
                
            elif method.lower() == 'te':
                # For Transfer Entropy: compare TE value against threshold
                connectivity_matrix[source, target] = 1 if value > threshold else 0
                
            elif method.lower() == 'ccm':
                # For CCM: compare correlation value against threshold
                connectivity_matrix[source, target] = 1 if value > threshold else 0
            
            else:
                raise ValueError(f"Method {method} not supported. Choose from: 'granger', 'te', or 'ccm'")
    
    return connectivity_matrix
