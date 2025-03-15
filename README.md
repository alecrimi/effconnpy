# effconnpy

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alecrimi/effconnpy/main)

![Help Wanted](https://img.shields.io/badge/help%20wanted-issues-orange?style=for-the-badge)

## Overview
 
<img src="https://raw.githubusercontent.com/alecrimi/effconnpy/main/logo.png" alt="logo" width="200"/>


`Effconnpy` is a Python library for advanced causal inference and connectivity analysis in time series data, offering both bivariate and multivariate approaches.
The toolbox assumes that neuroimging data (e.g. from Nifti files) have been already pre-processed e.g. with fMRI-prep, and parcellated, therefore the time series have been saved in text files as .tsv
and can easily be loaded into a dataframe.

## Bivariate Causality Analysis
<img src="https://upload.wikimedia.org/wikipedia/commons/7/7d/GrangerCausalityIllustration.svg" alt="GCwikipedia" width="400"/>

Two core classes provide bivariate causal inference methods:

### 1. CausalityAnalyzer
Basic methods include:
- Bivariate Granger Causality
- Bivariate Transfer Entropy
- Bivariate Convergent Cross Mapping 

### 2. ExtendedCausalityAnalyzer
Extended methods include:
- Dynamic Bayesian Network
- Structural Equation Modeling
- DoWhy Causal Discovery
- Dynamic Causal Modeling

## Multivariate Causality Analysis

Three specialized multivariate approaches:

### 1. Multivariate Granger Causality
- Based on methodology by Barnett & Seth, Journal of Neuroscience Methods 2014
- VAR model-based causality inference
- Log-likelihood ratio testing

### 2. Multivariate Convergent Cross-Mapping (CCM)
- Inspired by Nithya & Tangirala, ICC 2019
- Nonlinear causality detection
- Network-based causal relationship visualization

### 3. Multivariate Transfer Entropy
- Methodology from Duan et al. 2022
- Information-theoretic causality measure
- Supports conditional transfer entropy

N.B. The multivariate implementations are not considered state-of-the-art and are not fully tested, please report any error or bug.

## Visualization of effective connectivity

- Plotting of directed graphs, also overlapping with tractography

- Visualization of time series over structural connections (currently not working properly)

<img src="https://raw.githubusercontent.com/alecrimi/effconnpy/main/vis_eff_conn.png" alt="visualization" width="400"/>


## Installation

```bash
pip install effconnpy
```

## Quick Example

```python
from effconnpy import CausalityAnalyzer  , create_connectivity_matrix 
import numpy as np
# Generate sample time series
data = np.random.rand(100, 3)
analyzer = CausalityAnalyzer(data)
results = analyzer.causality_test(method='granger')
print(results)

binary_matrix =  create_connectivity_matrix(results, method = 'granger') 
print(binary_matrix)
```


## Multivariate GC Example using causal logistic maps

```python
from effconnpy.multivariateGC import MultivariateGrangerCausality  
from effconnpy import create_connectivity_matrix 
import numpy as np

# Generate sample time series 
#The first 2 time series are related by a logistic causal mapping, 
#while the third is random and has no causal relationship with the others
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
    y[t + 1] = y[t] * (3.1 - 3.1 * y[t] - 0.8 * x[t])
z = np.random.rand(100)  
data = np.vstack((x, y,z)).T   

analyzer = MultivariateGrangerCausality(data)
results = analyzer.multivariate_granger_causality()
print(results)
```
Indeed, there is also a causal time series generator script to generate ground-truth time series which can be complemented by random or other time series:

```python
    from effconnpy import timeseriesgenerator
    # Create an instance of the generator
    generator = TimeSeriesGenerator()    
    # Generate some example data
    n_points = 1000
    
    # Lorenz system
    t, x, y, z = generator.lorenz_system(n_points)
    
    # Coupled logistic map
    logistic_x, logistic_y = generator.coupled_logistic_map(n_points)
    
    # Kuramoto oscillators
    kura1, kura2 = generator.kuramoto_oscillators(n_points)
    
    # AR process
    ar_x, ar_y = generator.ar_process(n_points)
    
    # Nonlinear coupled system
    nl_x, nl_y = generator.nonlinear_coupled_system(n_points)
```



## Visualization of effective connectivity example

```python
from effconnpy import  vis_effconn 

node_file = "Node_AAL90.node"
edge_file = "my_effconn.csv"
    
vis_effconn(node_file, edge_file, show_tractography=True)
```
This will show the overlay of a tractography from the HCP dataset over a connectome according to an atlas with given nodes coordinates, and the effective connectivity computed and saved in a csv file.
If a structural connectivity matrix (non-effective) is passed, the function whill show arrows on both directions for all edges. 


## To be done
1. Automatic selection of lags
2. Extension with own work as Structurally constrained Granger causality A. Crimi Neuroimage 2021
and Reservoir Computing Causality (End-to-End Stroke Imaging Analysis Using Effective Connectivity and Interpretable Artificial Intelligence
Wojciech Ciezobka; Joan Falcó-Roget; Cemal Koba; Alessandro Crimi, IEEE Access 2025)


## Contributing

Contributions welcome! Please read our contributing guidelines before submitting pull requests.
Currently disabled, just open issues and I will follow up

## License

MIT License

[![Star History Chart](https://api.star-history.com/svg?repos=alecrimi/effconnpy&type=Date)](https://star-history.com/#alecrimi/effconnpy)



