from effconnpy import CausalityAnalyzer  , create_connectivity_matrix
import numpy as np
# Generate sample time series
data = np.random.rand(100, 3)
analyzer = CausalityAnalyzer(data)
results = analyzer.causality_test(method='transfer_entropy')
print(results)
binary_matrix =  create_connectivity_matrix(results, method="TE", threshold=0.1)
print(binary_matrix)
