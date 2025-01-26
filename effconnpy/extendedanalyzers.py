import numpy as np
import pandas as pd
from typing import Union, List, Optional
import networkx as nx
import dowhy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from .analyzers import CausalityAnalyzer

class ExtendedCausalityAnalyzer(CausalityAnalyzer):
    def __init__(self, data: Union[np.ndarray, pd.DataFrame, List[np.ndarray]]):
        """
        Initialize ExtendedCausalityAnalyzer with extended causal inference techniques
        
        Args:
            data: Input time series data
        """
        super().__init__(data)
        
        # Convert to DataFrame for some methods
        self.df = pd.DataFrame(self.data)
    
    def dynamic_bayesian_network(self, max_lag: int = 1, correlation_threshold: float = 0.3):
        """
        Construct Dynamic Bayesian Network using NetworkX
        
        Args:
            max_lag: Maximum number of time lags to consider
            correlation_threshold: Correlation threshold for edge creation
        
        Returns:
            Directed graph representing causal relationships
        """
        # Create directed graph
        dbn = nx.DiGraph()
        
        # Create lagged features
        lagged_df = pd.DataFrame()
        for col in self.df.columns:
            for lag in range(1, max_lag + 1):
                lagged_df[f"{col}_lag_{lag}"] = self.df[col].shift(lag)
        
        # Combine original and lagged data
        combined_df = pd.concat([self.df, lagged_df], axis=1).dropna()
        
        # Compute correlations and create edges
        for col1 in self.df.columns:
            for col2 in self.df.columns:
                if col1 != col2:
                    # Check current and lagged correlations
                    for lag in range(1, max_lag + 1):
                        corr, p_value = pearsonr(
                            combined_df[col1], 
                            combined_df[f"{col2}_lag_{lag}"]
                        )
                        
                        # Add edge if correlation exceeds threshold
                        if abs(corr) > correlation_threshold:
                            dbn.add_edge(col2, col1, weight=abs(corr), lag=lag)
        
        return dbn
    
    def structural_equation_modeling(self):
        """
        Perform Structural Equation Modeling using Factor Analysis and Linear Regression
        
        Returns:
            Dictionary containing model parameters and fit statistics
        """
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        
        # Factor Analysis to estimate latent variables
        fa = FactorAnalysis(n_components=min(self.num_series, 3), random_state=42)
        latent_factors = fa.fit_transform(scaled_data)
        
        # Fit linear relationships between latent factors and observed variables
        models = []
        r2_scores = []
        coefficients = []
        
        for i in range(self.num_series):
            model = LinearRegression()
            model.fit(latent_factors, scaled_data[:, i])
            
            models.append(model)
            r2_scores.append(model.score(latent_factors, scaled_data[:, i]))
            coefficients.append(model.coef_)
        
        return {
            'latent_factors': latent_factors,
            'factor_loadings': fa.components_,
            'models': models,
            'r2_scores': r2_scores,
            'coefficients': coefficients,
            'explained_variance_ratio': fa.explained_variance_ratio_
        }
    
    def causal_discovery_dowhy(self, treatment_var: int = 0, outcome_var: int = 1):
        """
        Causal Discovery using DoWhy
        
        Args:
            treatment_var: Index of treatment variable
            outcome_var: Index of outcome variable
        
        Returns:
            Causal inference results
        """
        # Prepare data
        data = self.df.copy()
        
        # Create causal graph
        graph = nx.DiGraph()
        graph.add_edges_from([(f'X{treatment_var}', f'X{outcome_var}')])
        
        # DoWhy causal model
        causal_model = dowhy.CausalModel(
            data=data,
            treatment=f'X{treatment_var}',
            outcome=f'X{outcome_var}',
            graph=graph
        )
        
        # Identify causal effect
        identified_estimand = causal_model.identify_effect()
        
        # Estimate causal effect
        estimate = causal_model.estimate_effect(identified_estimand)
        
        return {
            'causal_model': causal_model,
            'identified_estimand': identified_estimand,
            'estimate': estimate
        }
    
    def dynamic_causal_modeling(self):
        """
        Dynamic Causal Modeling using state space representation and Kalman filtering
        
        Returns:
            Dictionary containing model parameters and state estimates
        """
        n_timesteps = len(self.data)
        
        # Initialize state space matrices
        state_dim = self.num_series
        obs_dim = self.num_series
        
        # Simple state transition matrix (identity + small coupling)
        A = np.eye(state_dim) + 0.1 * np.random.randn(state_dim, state_dim)
        
        # Observation matrix (identity)
        C = np.eye(obs_dim)
        
        # Initialize state estimates
        x_est = np.zeros((n_timesteps, state_dim))
        P_est = np.zeros((n_timesteps, state_dim, state_dim))
        
        # Process and observation noise covariances
        Q = np.eye(state_dim) * 0.1
        R = np.eye(obs_dim) * 0.1
        
        # Forward pass (Kalman filter)
        x_est[0] = np.zeros(state_dim)
        P_est[0] = np.eye(state_dim)
        
        for t in range(1, n_timesteps):
            # Predict
            x_pred = A @ x_est[t-1]
            P_pred = A @ P_est[t-1] @ A.T + Q
            
            # Update
            K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + R)
            x_est[t] = x_pred + K @ (self.data[t] - C @ x_pred)
            P_est[t] = (np.eye(state_dim) - K @ C) @ P_pred
        
        return {
            'state_estimates': x_est,
            'state_covariances': P_est,
            'transition_matrix': A,
            'observation_matrix': C,
            'process_noise': Q,
            'observation_noise': R
        }
    
    def causality_test(self, 
                      method: str = 'granger', 
                      lag: Optional[int] = None, 
                      verbose: bool = False, 
                      **kwargs):
        """
        Extended causality test method
        
        Args:
            method: Causality test method
            lag: Number of lags
            verbose: Verbose output
            **kwargs: Additional method-specific parameters
        
        Returns:
            Causality test results
        """
        # Existing methods plus new ones
        methods = {
            'granger': self.granger_causality,
            'dbn': self.dynamic_bayesian_network,
            'sem': self.structural_equation_modeling,
            'dowhy': self.causal_discovery_dowhy,
            'dcm': self.dynamic_causal_modeling
        }
        
        if method.lower() not in methods:
            raise ValueError(f"Method {method} not supported. Choose from {list(methods.keys())}")
        
        return methods[method.lower()](**kwargs)
