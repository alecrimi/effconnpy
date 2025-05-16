import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple
import networkx as nx
from scipy import stats
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
            self.time_series =  validate_input(data) #data
            if len(self.time_series.shape) == 1:
                self.num_nodes = 1
                self.time_series = self.time_series.reshape(-1, 1)
            else:
                self.num_nodes = self.time_series.shape[1]
            self.G = None
            self.adjacency_matrix = None

    def construct_network_from_timeseries(self, method: str = 'mutual_info',
                                          threshold: float = 0.5,
                                          absolute: bool = True) -> nx.Graph:
        if self.time_series is None:
            raise ValueError("No time series data available to construct network")

        relationship_matrix = np.zeros((self.num_nodes, self.num_nodes))

        if method == 'correlation':
            corr_matrix = np.corrcoef(self.time_series.T)
            relationship_matrix = corr_matrix

        elif method == 'mutual_info':
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j:
                        bins = min(10, len(self.time_series) // 5)
                        c_xy = np.histogram2d(self.time_series[:, i], self.time_series[:, j], bins=bins)[0]
                        mi = self._mutual_information_from_joint(c_xy)
                        relationship_matrix[i, j] = mi
        else:
            raise ValueError(f"Unknown method: {method}")

        adjacency_matrix = np.zeros_like(relationship_matrix)
        mask = np.abs(relationship_matrix) > threshold if absolute else relationship_matrix > threshold
        adjacency_matrix[mask] = 1
        np.fill_diagonal(adjacency_matrix, 0)
        G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

        self.G = G
        self.adjacency_matrix = adjacency_matrix
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
    network = ei.construct_network_from_timeseries(threshold=0.3)

    ei_value = ei.effective_information(source=0, target=1)
    print(f"Effective Information from node 0 to node 1: {ei_value}")

    ei_matrix = ei.effective_information_matrix()
    print(ei_matrix)
