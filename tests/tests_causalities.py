
import numpy as np
from causal_analysis import CausalityAnalyzer

# Create sample data
data = np.random.rand(100, 3)  # 3 time series

# Initialize analyzer
analyzer = CausalityAnalyzer(data)

# Initialize with time series data
analyzer = ExtendedCausalityAnalyzer(data)

# Dynamic Bayesian Network
dbn_result = analyzer.causality_test(method='dbn')

# Structural Equation Modeling
sem_result = analyzer.causality_test(method='sem')

# Causal Discovery
dowhy_result = analyzer.causality_test(method='dowhy', 
                                       treatment_var=0, 
                                       outcome_var=1)

# Dynamic Causal Modeling
dcm_result = analyzer.causality_test(method='dcm')


# Perform Granger Causality test
results = analyzer.causality_test(method='granger', lag=1, verbose=True)
