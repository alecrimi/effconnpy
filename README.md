# Effconnpy

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yourusername/effconnpy/main)
Current stable version = 0.1.12

## Overview

<img src="logo.png" alt="logo" width="200"/>

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

## Quick Examples

```python
from effconnpy import CausalityAnalyzer
import numpy as np
# Generate sample time series
data = np.random.rand(100, 3)
analyzer = CausalityAnalyzer(data)
results = analyzer.causality_test(method='granger')
print(results)
binary_matrix = create_connectivity_matrix(results, threshold=0.05, metric='p_value')
```
and for the multivariate case:

```python
import numpy as np
from effconnpy import MultivariateGrangerCausality
# Generate example data
np.random.seed(42)
data = np.random.randn(100, 3)
# Run analysis
mgc = MultivariateGrangerCausality(data)
results = mgc.multivariate_granger_causality(max_lag=1)
print(results)
#Create a connectivity matrix with 1s if the p-val < 0.05
binary_matrix = create_connectivity_matrix(results, threshold=0.05, metric='p_value')
```

## To be done
1. Automatic selection of lags
2. Generate API documentation with Sphinx
3. Add extensions with own works as [Structurally constrained Granger causality A. Crimi Neuroimage 2021}(https://www.sciencedirect.com/science/article/pii/S1053811921005644)
and Reservoir Computing Causality ([End-to-End Stroke Imaging Analysis Using Effective Connectivity and Interpretable Artificial Intelligence
Wojciech Ciezobka; Joan Falcó-Roget; Cemal Koba; Alessandro Crimi, IEEE Access 2025](https://ieeexplore.ieee.org/document/10839398)) Please refer to the individual github's repos for those for now.
4. Add High-order autoregressor Granger Causality

## Citation
If you use or expand this package, please cite the related papers
For the bivariate analysis ([End-to-End Stroke Imaging Analysis Using Effective Connectivity and Interpretable Artificial Intelligence
Wojciech Ciezobka; Joan Falcó-Roget; Cemal Koba; Alessandro Crimi, IEEE Access 2025](https://ieeexplore.ieee.org/document/10839398))
For the multivariate analysis  [Structurally constrained Granger causality A. Crimi Neuroimage 2021}(https://www.sciencedirect.com/science/article/pii/S1053811921005644)

## Inspirations
Code is based on those papers

###1. Convergent Cross-Mapping
Sugihara, George; May, Robert; Ye, Hao; Hsieh, Chih-hao; Deyle, Ethan; Fogarty, Michael; Munch, Stephan (2012). "Detecting Causality in Complex Ecosystems". Science. 338 (6106): 496–500. 

###2. Multivariate Granger Causality
Seth, A. K., Barrett, A. B., & Barnett, L. (2015). Granger causality analysis in neuroscience and neuroimaging. Journal of Neuroscience Methods, 261, 120–137.
Porting of Simplified code from Matlab

###3. Dynamic Bayesian Network
Murphy, K. P. (2002). Dynamic Bayesian networks: Representation, inference and learning. Ph.D. Thesis, UC Berkeley.
Simplification of pomegranate Python library.

###4. Structural Equation Modeling
Bollen, K. A. (1989). Structural Equations with Latent Variables. Wiley.
Porting of Lavaan R code

###5. DoWhy Causal Discovery
Shanmugam, K., Kalainathan, D., Goudet, O., & Lopez-Paz, D. (2020). DoWhy: An End-to-End Library for Causal Inference. ArXiv Paper
Refactoring of DoWhy Library in Python without Pymc3.
 
###6. Dynamic Causal Modeling
Friston, K. (2007). Statistical Parametric Mapping: The Analysis of Functional Brain Images. Elsevier.
Refactoring Matlab code without SPM 

###7. Multivariate Convergent Cross Mapping
S Nithya and Arun K Tangirala. (2019) “Multivariable Causal Analysis of Nonlinear Dynamical Systems using. Convergent Cross Mapping”

###8. Multivariate Transfer Entropy
Ziheng Duan; Haoyan Xu; Yida Huang; Jie Feng; Yueyang Wang (2022) Multivariate Time Series Forecasting with Transfer Entropy Graph


## Contributing

Contributions welcome! Please read the contributing guidelines before submitting pull requests.
Or just open issues and I will follow up

## License

MIT License
