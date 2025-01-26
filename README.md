# effconnpy

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yourusername/effconnpy/main)

## Overview

<img src="logo.png" alt="logo" width="200"/>

`Effconnpy` is a Python library for advanced causal inference and connectivity analysis in time series data, offering both bivariate and multivariate approaches.

## Bivariate Causality Analysis

Two core classes provide bivariate causal inference methods:

### 1. CausalityAnalyzer
- **Approach**: Granger Causality
- Methods:
  - Standard Granger Causality test
  - Supports multiple time series
  - Computes directional causality

### 2. ExtendedCausalityAnalyzer
Extended methods include:
- Dynamic Bayesian Network
- Structural Equation Modeling
- DoWhy Causal Discovery
- Dynamic Causal Modeling

## Multivariate Causality Analysis

Three specialized multivariate approaches:

### 1. Multivariate Granger Causality
![Multivariate GC](multivariate_gc.png)
- Based on methodology by Barnett & Seth
- VAR model-based causality inference
- Log-likelihood ratio testing

### 2. Multivariate Convergent Cross-Mapping (CCM)
![Multivariate CCM](multivariate_ccm.png)
- Inspired by Nithya & Tangirala
- Nonlinear causality detection
- Network-based causal relationship visualization

### 3. Multivariate Transfer Entropy
![Multivariate TE](multivariate_te.png)
- Methodology from Duan et al.
- Information-theoretic causality measure
- Supports conditional transfer entropy

## Installation

```bash
pip install effconnpy
```

## Quick Example

```python
from effconnpy import CausalityAnalyzer
import numpy as np

# Generate sample time series
data = np.random.rand(100, 3)
analyzer = CausalityAnalyzer(data)
results = analyzer.causality_test(method='granger')
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
