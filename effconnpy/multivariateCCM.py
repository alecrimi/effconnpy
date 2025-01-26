import numpy as np
import scipy.spatial
import networkx as nx
from typing import List, Dict, Any

class MultivariateCCM:
    def __init__(self, time_series: np.ndarray):
        """
        Initialize multivariate CCM
        
        Args:
            time_series: 2D numpy array where each column is a time series
        """
        self.time_series = time_series
        self.num_series = time_series.shape[1]
    
    def embed_series(self, series: np.ndarray, lag: int, dim: int) -> np.ndarray:
        """
        Create time delay embedding
        
        Args:
            series: Input time series
            lag: Time lag
            dim: Embedding dimension
        
        Returns:
            Embedded time series matrix
        """
        N = len(series) - (dim - 1) * lag
        embedded = np.zeros((N, dim))
        
        for i in range(N):
            embedded[i, :] = [series[i + j * lag] for j in range(dim)]
        
        return embedded
    
    def nearest_neighbors(self, embedded_series: np.ndarray, 
                           target_point: np.ndarray, 
                           num_neighbors: int) -> List[int]:
        """
        Find nearest neighbors in embedded space
        
        Args:
            embedded_series: Embedded time series
            target_point: Target point for finding neighbors
            num_neighbors: Number of nearest neighbors
        
        Returns:
            Indices of nearest neighbors
        """
        # Compute Euclidean distances
        distances = scipy.spatial.distance.cdist([target_point], embedded_series)[0]
        return np.argsort(distances)[:num_neighbors]
    
    def predict_lib_size_convergence(self, 
                                     series_x: np.ndarray, 
                                     series_y: np.ndarray, 
                                     max_lib_size: int = 50, 
                                     num_steps: int = 10,
                                     lag: int = 1, 
                                     dim: int = 2) -> Dict[str, np.ndarray]:
        """
        Compute predictive skill across different library sizes
        
        Args:
            series_x: First time series
            series_y: Target time series
            max_lib_size: Maximum library size to test
            num_steps: Number of library size steps
            lag: Time delay for embedding
            dim: Embedding dimension
        
        Returns:
            Dictionary of library size and predictive skill
        """
        # Embed both time series
        x_embedded = self.embed_series(series_x, lag, dim)
        y_embedded = self.embed_series(series_y, lag, dim)
        
        # Compute library size steps
        lib_sizes = np.linspace(10, max_lib_size, num_steps).astype(int)
        predictive_skills = []
        
        for lib_size in lib_sizes:
            # Initialize skill tracking
            y_pred = []
            true_y = []
            
            # Predict for each point
            for i in range(len(x_embedded)):
                # Find nearest neighbors in X
                x_neighbors = self.nearest_neighbors(x_embedded, x_embedded[i], lib_size)
                
                # Get corresponding neighbors in Y
                y_neighbor_values = y_embedded[x_neighbors, -1]
                
                # Predict target series value
                y_pred.append(np.mean(y_neighbor_values))
                true_y.append(y_embedded[i, -1])
            
            # Compute correlation skill
            skill = np.corrcoef(y_pred, true_y)[0, 1]
            predictive_skills.append(skill)
        
        return {
            'library_sizes': lib_sizes,
            'predictive_skills': np.array(predictive_skills)
        }
    
    def multivariate_ccm_analysis(self, 
                                   max_lib_size: int = 50, 
                                   num_steps: int = 10, 
                                   lag: int = 1, 
                                   dim: int = 2) -> nx.DiGraph:
        """
        Perform multivariate CCM across all series
        
        Args:
            max_lib_size: Maximum library size
            num_steps: Number of library size steps
            lag: Time delay for embedding
            dim: Embedding dimension
        
        Returns:
            Directed graph of causal relationships
        """
        # Create causal network
        causal_network = nx.DiGraph()
        
        # Compute CCM for all pairs
        for i in range(self.num_series):
            for j in range(self.num_series):
                if i != j:
                    # Compute predictive skill
                    result = self.predict_lib_size_convergence(
                        self.time_series[:, i], 
                        self.time_series[:, j], 
                        max_lib_size, 
                        num_steps, 
                        lag, 
                        dim
                    )
                    
                    # Check convergence and causality
                    final_skill = result['predictive_skills'][-1]
                    causal_strength = final_skill if final_skill > 0 else 0
                    
                    # Add edge to causal network
                    if causal_strength > 0:
                        causal_network.add_edge(f'Series_{i}', f'Series_{j}', weight=causal_strength)
        
        return causal_network

# Example usage
def main():
    # Generate sample multivariate time series
    np.random.seed(42)
    time_series = np.random.rand(200, 3)  # 3 time series of length 200
    
    # Perform multivariate CCM
    ccm_analyzer = MultivariateCCM(time_series)
    causal_network = ccm_analyzer.multivariate_ccm_analysis()
    
    # Print causal relationships
    print("Causal Network Edges:")
    for edge in causal_network.edges(data=True):
        print(f"{edge[0]} → {edge[1]}: Strength {edge[2]['weight']}")

if __name__ == "__main__":
    main()
