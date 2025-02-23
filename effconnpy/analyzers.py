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

def create_connectivity_matrix_GC(results, threshold=0.05, metric='p_value'):
    """
    Convert Granger causality results to connectivity matrix
    
    Args:
        results: Dictionary of Granger causality results
        threshold: Significance threshold for p-values (default=0.05)
        metric: Which metric to use ('p_value' or 'f_statistic')
    
    Returns:
        numpy array: Connectivity matrix where entry [i,j] represents causality from i to j
    """
    # Get number of nodes from the results
    nodes = set()
    for key in results.keys():
        source, target = map(int, key.split(' → '))
        nodes.add(source)
        nodes.add(target)
    n_nodes = len(nodes)
    
    # Initialize connectivity matrix
    connectivity_matrix = np.zeros((n_nodes, n_nodes))
    
    # Fill matrix based on selected metric
    for connection, stats in results.items():
        source, target = map(int, connection.split(' → '))
        
        if metric == 'p_value':
            # For p-values: connection exists if p < threshold
            connectivity_matrix[source, target] = 1 if stats['p_value'] < threshold else 0
        else:
            # For F-statistic: use the F-value directly
            connectivity_matrix[source, target] = stats['f_statistic']
    
    return connectivity_matrix
    
    
def create_connectivity_matrix_TE(results, threshold=0.1):
    """
    Convert Transfer Entropy results to a connectivity matrix.
    
    Args:
        results: Dictionary of Transfer Entropy results
        threshold: Minimum TE value required for connection (default=0.1)
    
    Returns:
        numpy array: Connectivity matrix where entry [i, j] represents TE from i to j
    """
    nodes = set()
    for key in results.keys():
        source, target = map(int, key.split(' → '))
        nodes.add(source)
        nodes.add(target)
    
    n_nodes = len(nodes)
    connectivity_matrix = np.zeros((n_nodes, n_nodes))
    
    for connection, te_value in results.items():
        source, target = map(int, connection.split(' → '))
        
        # Apply threshold to determine connectivity
        connectivity_matrix[source, target] = 1 if te_value > threshold else 0
    
    return connectivity_matrix
