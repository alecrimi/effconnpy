import numpy as np
import scipy.stats as stats
from scipy.spatial import distance
from typing import List, Dict, Any

class MultivariateTransferEntropy:
    def __init__(self, time_series: np.ndarray):
        """
        Initialize multivariate transfer entropy analysis
        
        Args:
            time_series: 2D numpy array where each column is a time series
        """
        self.time_series = time_series
        self.num_series = time_series.shape[1]
    
    def estimate_probability(self, data: np.ndarray, k: int = 3) -> float:
        """
        Estimate probability density using k-nearest neighbors
        
        Args:
            data: Input data points
            k: Number of nearest neighbors
        
        Returns:
            Probability estimate
        """
        n = len(data)
        distances = distance.cdist(data, data)
        np.fill_diagonal(distances, np.inf)
        
        # Find k-th nearest neighbor distances
        knn_distances = np.sort(distances, axis=1)[:, k]
        
        # Compute local volume estimate
        volumes = (2 * knn_distances) ** data.shape[1]
        
        return np.mean(1 / (volumes * n))
    
    def compute_conditional_transfer_entropy(self, 
                                             source: np.ndarray, 
                                             target: np.ndarray, 
                                             context: np.ndarray = None, 
                                             lag: int = 1) -> float:
        """
        Compute conditional transfer entropy
        
        Args:
            source: Source time series
            target: Target time series
            context: Contextual/conditioning time series
            lag: Time lag
        
        Returns:
            Conditional transfer entropy value
        """
        def create_joint_vector(source, target, context, lag):
            """Create joint state vector"""
            n = len(source) - lag
            joint_vector = np.zeros((n, 1 + (1 if context is not None else 0)))
            joint_vector[:, 0] = source[lag:]
            
            if context is not None:
                joint_vector[:, 1] = context[lag:]
            
            return joint_vector
        
        # Prepare history states
        source_history = source[:-lag]
        target_history = target[:-lag]
        
        # Create conditional joint vectors
        if context is not None:
            joint_past = create_joint_vector(target, source, context, lag)
            joint_future = create_joint_vector(target, source, context, 0)
        else:
            joint_past = source_history.reshape(-1, 1)
            joint_future = target[lag:].reshape(-1, 1)
        
        # Estimate probabilities
        p_past = self.estimate_probability(joint_past)
        p_future = self.estimate_probability(joint_future)
        
        # Transfer entropy computation
        log_ratio = np.log(p_future / p_past)
        
        return log_ratio
    
    def multivariate_transfer_entropy(self, 
                                      lag: int = 1, 
                                      context_series: List[int] = None) -> Dict[str, float]:
        """
        Compute multivariate transfer entropy across all series
        
        Args:
            lag: Time lag for analysis
            context_series: Indices of contextual series
        
        Returns:
            Dictionary of transfer entropy values
        """
        te_results = {}
        
        for source_idx in range(self.num_series):
            for target_idx in range(self.num_series):
                if source_idx != target_idx:
                    # Select source and target series
                    source = self.time_series[:, source_idx]
                    target = self.time_series[:, target_idx]
                    
                    # Compute transfer entropy with optional context
                    if context_series:
                        context_data = self.time_series[:, context_series]
                        te_value = self.compute_conditional_transfer_entropy(
                            source, target, context_data, lag
                        )
                    else:
                        te_value = self.compute_conditional_transfer_entropy(
                            source, target, lag=lag
                        )
                    
                    te_results[f'{source_idx} → {target_idx}'] = te_value
        
        return te_results

# Example usage
def main():
    # Generate sample multivariate time series
    np.random.seed(42)
    time_series = np.random.rand(200, 3)  # 3 time series of length 200
    
    # Perform multivariate transfer entropy
    te_analyzer = MultivariateTransferEntropy(time_series)
    te_results = te_analyzer.multivariate_transfer_entropy(lag=1)
    
    # Print transfer entropy results
    for connection, value in te_results.items():
        print(f"Transfer Entropy {connection}: {value}")

if __name__ == "__main__":
    main()
