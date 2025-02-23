# effconnpy

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yourusername/effconnpy/main)

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

## Installation

```bash
pip install effconnpy
```

## Quick Example

```python
from effconnpy import CausalityAnalyzer  , create_connectivity_matrix_GC
import numpy as np
# Generate sample time series
data = np.random.rand(100, 3)
analyzer = CausalityAnalyzer(data)
results = analyzer.causality_test(method='granger')
print(results)
binary_matrix =  create_connectivity_matrix_GC(results, threshold=0.1)
print(binary_matrix)
```

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

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/alecrimi">
        <img src="https://avatars.githubusercontent.com/u/16406658?v=4" width="100px;" alt="username1"/>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/KuznetsovIllya">
        <img src="https://avatars.githubusercontent.com/u/10212879?v=4&format=png" width="100px;" alt="username2"/>
  
      </a>
    </td>
  </tr>
</table>

