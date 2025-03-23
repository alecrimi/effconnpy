import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from .causality_analyzer import CausalityAnalyzer  # Import the parent class

class Reservoir:
    """Echo State Network reservoir for temporal processing"""
    def __init__(self, input_size, reservoir_size=100, leaking_rate=0.3, 
                 spectral_radius=0.9, activation=np.tanh, seed=None):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.activation = activation
        
        # Initialize weights
        np.random.seed(seed)
        self.W_in = np.random.randn(input_size, reservoir_size) * 0.1
        self.W_res = np.random.randn(reservoir_size, reservoir_size)
        rho = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= spectral_radius / rho
        self.bias = np.random.randn(reservoir_size) * 0.1

    def process(self, inputs):
        """Process input sequence through reservoir"""
        states = np.zeros((inputs.shape[0], self.reservoir_size))
        state = np.zeros(self.reservoir_size)
        
        for t in range(inputs.shape[0]):
            inp = inputs[t]
            state = (1 - self.leaking_rate) * state + \
                    self.leaking_rate * self.activation(
                        inp @ self.W_in + state @ self.W_res + self.bias
                    )
            states[t] = state
        return states

class NonlinearGrangerCausality(CausalityAnalyzer):
    """Nonlinear Granger Causality using Reservoir Computing"""
    def __init__(self, data, reservoir_size=100, leaking_rate=0.3, 
                 spectral_radius=0.9, activation=np.tanh, seed=None):
        super().__init__(data)  # Initialize the parent class
        self.reservoir_size = reservoir_size
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.activation = activation
        self.seed = seed

    def granger_causality(self, lag=1, verbose=False):
        results = {}
        if self.num_series == 1:
            return {"error": "Requires multiple time series"}
        
        # Precompute all possible input sizes
        max_input_size = 2 * lag
        
        for target in range(self.num_series):
            for source in range(self.num_series):
                if target == source:
                    continue
                
                y = self.data[:, target]
                x = self.data[:, source]
                n_samples = len(y) - lag
                
                if n_samples < 1:
                    raise ValueError("Insufficient data for given lag")
                
                # Create feature matrices
                restricted = np.zeros((n_samples, lag))
                unrestricted = np.zeros((n_samples, 2*lag))
                
                for t in range(lag, len(y)):
                    idx = t - lag
                    restricted[idx] = y[t-lag:t][::-1]  # [y_{t-1},..., y_{t-lag}]
                    unrestricted[idx] = np.concatenate([
                        y[t-lag:t][::-1], 
                        x[t-lag:t][::-1]
                    ])
                
                # Process through reservoirs
                res_reservoir = Reservoir(
                    lag, self.reservoir_size, self.leaking_rate,
                    self.spectral_radius, self.activation, self.seed
                )
                res_states = sm.add_constant(res_reservoir.process(restricted))
                
                unres_reservoir = Reservoir(
                    2*lag, self.reservoir_size, self.leaking_rate,
                    self.spectral_radius, self.activation, self.seed
                )
                unres_states = sm.add_constant(unres_reservoir.process(unrestricted))
                
                # Fit models
                model_r = sm.OLS(y[lag:], res_states).fit()
                model_ur = sm.OLS(y[lag:], unres_states).fit()
                
                # Calculate F-statistic
                ssr_r, ssr_ur = model_r.ssr, model_ur.ssr
                df_num = lag  # Approximate additional parameters
                df_den = n_samples - self.reservoir_size - 1
                
                f_stat = ((ssr_r - ssr_ur)/df_num) / (ssr_ur/df_den)
                p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)
                
                key = f"{source} → {target}"
                results[key] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'ssr_restricted': ssr_r,
                    'ssr_unrestricted': ssr_ur
                }
                
                if verbose:
                    print(f"{key}: F={f_stat:.3f}, p={p_value:.4f}")
        
        return results
