import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple
import networkx as nx
from scipy import stats
from collections import Counter
from .utils import validate_input

class EffectiveInformation:
    """
    Effective Information calculation using KL divergence as defined in
    Hoel et al., PNAS 2013. This class supports network construction
    from time series and computes EI and related measures.
    """

    def __init__(self, data: Union[np.ndarray, pd.DataFrame, nx.Graph, List[np.ndarray]]):
        if isinstance(data, nx.Graph):
            self.G = data
            self.adjacency_matrix = nx.to_numpy_array(self.G)
            self.num_nodes = self.adjacency_matrix.shape[0]
            self.time_series = None
        else:
            self.time_series = data
            if len(self.time_series.shape) == 1:
                self.num_nodes = 1
                self.time_series = self.time_series.reshape(-1, 1)
            else:
                self.num_nodes = self.time_series.shape[1]
            self.G = None
            self.adjacency_matrix = None

    def construct_network_from_timeseries(self, binarize=True, n_bins=2):
        """
        Constructs a transition graph from time series data suitable for Effective Information (EI) calculation.

        Parameters:
        - binarize (bool): Whether to binarize the time series data.
        - n_bins (int): Number of bins to discretize the data if not binarizing.

        Returns:
        - G (nx.DiGraph): Directed graph representing state transitions.
        """
        if self.time_series is None:
            raise ValueError("No time series data available to construct network")

        # Step 1: Discretize time series data
        if binarize:
            # Binarize each node's time series based on median
            discretized = (self.time_series > np.median(self.time_series, axis=0)).astype(int)
        else:
            # Discretize into n_bins using quantiles
            discretized = np.zeros_like(self.time_series, dtype=int)
            for i in range(self.time_series.shape[1]):
                bins = np.quantile(self.time_series[:, i], q=np.linspace(0, 1, n_bins + 1))
                discretized[:, i] = np.digitize(self.time_series[:, i], bins[1:-1])

        # Step 2: Create state representations
        # Each state is a tuple representing the discretized values of all nodes at a time point
        states = [tuple(row) for row in discretized]

        # Step 3: Count transitions between consecutive states
        transitions = Counter()
        for i in range(len(states) - 1):
            transitions[(states[i], states[i + 1])] += 1

        # Step 4: Build transition probability matrix
        transition_counts = {}
        for (from_state, to_state), count in transitions.items():
            if from_state not in transition_counts:
                transition_counts[from_state] = Counter()
            transition_counts[from_state][to_state] += count

        # Normalize to get probabilities
        transition_probs = {}
        for from_state, to_states in transition_counts.items():
            total = sum(to_states.values())
            transition_probs[from_state] = {to_state: cnt / total for to_state, cnt in to_states.items()}

        # Step 5: Construct directed graph
        G = nx.DiGraph()
        for from_state, to_states in transition_probs.items():
            for to_state, prob in to_states.items():
                G.add_edge(from_state, to_state, weight=prob)

        self.G = G
        self.transition_probs = transition_probs
        return G

    def _mutual_information_from_joint(self, joint_probability: np.ndarray) -> float:
        joint_probability = joint_probability / np.sum(joint_probability)
        p_x = np.sum(joint_probability, axis=1)
        p_y = np.sum(joint_probability, axis=0)
        independent = np.outer(p_x, p_y)
        mask = (joint_probability > 0) & (independent > 0)
        mutual_info = np.sum(joint_probability[mask] * np.log2(joint_probability[mask] / independent[mask]))
        return mutual_info

    def _calculate_transition_matrix(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        transition_matrix = adjacency_matrix.copy()
        out_degrees = np.sum(transition_matrix, axis=1)
        for i in range(transition_matrix.shape[0]):
            if out_degrees[i] > 0:
                transition_matrix[i, :] /= out_degrees[i]
        return transition_matrix

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        mask = (p > 0) & (q > 0)
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

    def construct_channel_network(self, method: str = 'mutual_info', threshold: float = 0.1) -> nx.DiGraph:
        """
        Constructs a channel-based network using mutual information or correlation between channels.

        Parameters:
        - method (str): 'mutual_info' or 'correlation'
        - threshold (float): threshold for edge inclusion

        Returns:
        - G (nx.DiGraph): directed graph
        """
        if self.time_series is None:
            raise ValueError("No time series data available to construct network")

        n = self.num_nodes
        rel_matrix = np.zeros((n, n))

        if method == 'correlation':
            rel_matrix = np.corrcoef(self.time_series.T)
        elif method == 'mutual_info':
            for i in range(n):
                for j in range(n):
                    if i != j:
                        bins = min(10, len(self.time_series) // 5)
                        c_xy = np.histogram2d(self.time_series[:, i], self.time_series[:, j], bins=bins)[0]
                        rel_matrix[i, j] = self._mutual_information_from_joint(c_xy)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create adjacency matrix by thresholding
        adj = np.zeros_like(rel_matrix)
        adj[rel_matrix > threshold] = 1
        np.fill_diagonal(adj, 0)

        self.adjacency_matrix = adj
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        self.G = G
        return G


    def effective_information(self, source: Union[int, List[int]],
                              target: Union[int, List[int]],
                              adjacency_matrix: Optional[np.ndarray] = None) -> float:
        if adjacency_matrix is None:
            if self.adjacency_matrix is None:
                raise ValueError("No adjacency matrix available. Construct network first.")
            adjacency_matrix = self.adjacency_matrix

        if isinstance(source, int):
            source = [source]
        if isinstance(target, int):
            target = [target]

        transition_matrix = self._calculate_transition_matrix(adjacency_matrix)

        intervention_dist = np.zeros(adjacency_matrix.shape[0])
        intervention_dist[source] = 1.0 / len(source)

        effect_dist = intervention_dist @ np.linalg.matrix_power(transition_matrix, 1)
        target_effect_dist = effect_dist[target]

        if np.sum(target_effect_dist) > 0:
            target_effect_dist /= np.sum(target_effect_dist)
        else:
            target_effect_dist = np.full(len(target), 1.0 / len(target))

        uniform_dist = np.full(len(target), 1.0 / len(target))
        ei = self._kl_divergence(target_effect_dist, uniform_dist)
        return ei

    def effective_information_matrix(self, path_length: int = 1) -> np.ndarray:
        if self.adjacency_matrix is None:
            raise ValueError("No adjacency matrix available. Construct network first.")

        n = self.adjacency_matrix.shape[0]
        ei_matrix = np.zeros((n, n))
        transition_matrix = self._calculate_transition_matrix(self.adjacency_matrix)
        transition_power = np.linalg.matrix_power(transition_matrix, path_length)

        for source in range(n):
            for target in range(n):
                if source != target:
                    intervention_dist = np.zeros(n)
                    intervention_dist[source] = 1.0
                    effect_dist = intervention_dist @ transition_power
                    effect = effect_dist[target]
                    if effect > 0:
                        ei_matrix[source, target] = -np.log2(effect) * effect
        return ei_matrix

    def integrated_information(self, partition_method: str = 'bipartition') -> Tuple[float, List]:
        if self.adjacency_matrix is None:
            raise ValueError("No adjacency matrix available. Construct network first.")

        n = self.adjacency_matrix.shape[0]
        whole_system_ei = self.effective_information(list(range(n)), list(range(n)))
        min_phi = float('inf')
        min_partition = None

        if partition_method == 'bipartition':
            if n <= 20:
                for k in range(1, 2**(n - 1)):
                    binary = format(k, f'0{n}b')
                    part1 = [i for i in range(n) if binary[i] == '1']
                    part2 = [i for i in range(n) if binary[i] == '0']
                    if not part1 or not part2:
                        continue
                    ei_1to2 = self.effective_information(part1, part2)
                    ei_2to1 = self.effective_information(part2, part1)
                    normalization = min(len(part1), len(part2))
                    phi = (whole_system_ei - (ei_1to2 + ei_2to1)) / normalization if normalization > 0 else float('inf')
                    if phi < min_phi:
                        min_phi = phi
                        min_partition = [part1, part2]
            else:
                for _ in range(1000):
                    indicator = np.random.randint(0, 2, size=n)
                    part1 = [i for i in range(n) if indicator[i] == 1]
                    part2 = [i for i in range(n) if indicator[i] == 0]
                    if not part1 or not part2:
                        continue
                    ei_1to2 = self.effective_information(part1, part2)
                    ei_2to1 = self.effective_information(part2, part1)
                    normalization = min(len(part1), len(part2))
                    phi = (whole_system_ei - (ei_1to2 + ei_2to1)) / normalization if normalization > 0 else float('inf')
                    if phi < min_phi:
                        min_phi = phi
                        min_partition = [part1, part2]
        else:
            raise ValueError(f"Unknown partition method: {partition_method}")

        return min_phi, min_partition

    def network_complexity(self) -> float:
        if self.adjacency_matrix is None:
            raise ValueError("No adjacency matrix available. Construct network first.")

        n = self.adjacency_matrix.shape[0]
        ei_matrix = self.effective_information_matrix()
        total_ei = np.sum(ei_matrix)

        if total_ei > 0:
            flat_ei = ei_matrix.flatten()
            flat_ei = flat_ei[flat_ei > 0]
            normalized_ei = flat_ei / np.sum(flat_ei)
            entropy = -np.sum(normalized_ei * np.log2(normalized_ei))
            max_entropy = np.log2(len(flat_ei)) if len(flat_ei) > 0 else 0
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
                complexity = 4 * normalized_entropy * (1 - normalized_entropy)
            else:
                complexity = 0
        else:
            complexity = 0

        return complexity

if __name__ == "__main__":
    # Example use
    time_series = np.random.randn(1000, 5)
    ei = EffectiveInformation(time_series)
 
    # Build a network over the channels (not states)
    network = ei.construct_channel_network(method='mutual_info', threshold=0.1)

    ei_value = ei.effective_information(source=0, target=1)
    print(f"Effective Information from node 0 to node 1: {ei_value}")

    ei_matrix = ei.effective_information_matrix()
    print(ei_matrix)
