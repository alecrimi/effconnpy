from multivariateGC import MultivariateGrangerCausality  
from effconnpy import create_connectivity_matrix 
import numpy as np

# Generate sample time series 
# Parameters
T = 100  # Number of time steps
x = np.zeros(T)
y = np.zeros(T)
# Initial conditions
x[0] = 0.5
y[0] = 0.5
# Iterate the system
for t in range(T - 1):
    x[t + 1] = x[t] * (3.8 - 3.8 * x[t])
    y[t + 1] = y[t] * (3.1 - 3.1 * y[t] - 0.1 * x[t])

z = np.random.rand(100)  
data = np.vstack((x, y, z)).T   

analyzer = MultivariateGrangerCausality(data)
results = analyzer.multivariate_granger_causality()
print(results)

binary_matrix =  create_connectivity_matrix(results, method="granger")
print(binary_matrix)
