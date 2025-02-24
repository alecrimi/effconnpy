import numpy as np
from scipy.integrate import solve_ivp

class TimeSeriesGenerator:
    """A class containing various methods to generate different types of time series data."""
    
    @staticmethod
    def lorenz_system(n_points, sigma=10, beta=8/3, rho=28, initial_conditions=[1, 1, 1]):
        """
        Generate time series from the Lorenz system.
        
        Args:
            n_points (int): Number of time points to generate
            sigma (float): Parameter sigma for the Lorenz system
            beta (float): Parameter beta for the Lorenz system
            rho (float): Parameter rho for the Lorenz system
            initial_conditions (list): Initial values for x, y, z
            
        Returns:
            tuple: (time_points, X, Y, Z) where X, Y, Z are the components of the solution
        """
        def lorenz_equations(t, state, sigma=sigma, beta=beta, rho=rho):
            x, y, z = state
            dxdt = sigma * (y - x)
            dydt = x * (rho - z) - y
            dzdt = x * y - beta * z
            return [dxdt, dydt, dzdt]
        
        t_eval = np.linspace(0, 25, n_points)
        sol = solve_ivp(lorenz_equations, [0, 25], initial_conditions, t_eval=t_eval)
        
        return t_eval, sol.y[0], sol.y[1], sol.y[2]
    
    @staticmethod
    def coupled_logistic_map(n_points, r=3.9, alpha=0.1, seed=None):
        """
        Generate coupled logistic map time series.
        
        Args:
            n_points (int): Number of time points to generate
            r (float): Growth parameter
            alpha (float): Coupling strength
            seed (int): Random seed for reproducibility
            
        Returns:
            tuple: (X, Y) coupled time series
        """
        if seed is not None:
            np.random.seed(seed)
            
        X = np.zeros(n_points)
        Y = np.zeros(n_points)
        
        X[0], Y[0] = np.random.rand(), np.random.rand()
        
        for i in range(n_points-1):
            X[i+1] = r * X[i] * (1 - X[i])
            Y[i+1] = (1 - alpha) * r * Y[i] * (1 - Y[i]) + alpha * X[i]
        
        return X, Y
    
    @staticmethod
    def kuramoto_oscillators(n_points, omega=[1.0, 1.2], K=0.5, seed=None):
        """
        Generate Kuramoto oscillator time series.
        
        Args:
            n_points (int): Number of time points to generate
            omega (list): Natural frequencies of oscillators
            K (float): Coupling strength
            seed (int): Random seed for reproducibility
            
        Returns:
            tuple: (oscillator1, oscillator2) time series
        """
        if seed is not None:
            np.random.seed(seed)
            
        theta = np.zeros((n_points, 2))
        theta[0] = np.random.rand(2) * 2 * np.pi
        
        for t in range(1, n_points):
            theta[t, 0] = theta[t-1, 0] + omega[0]
            theta[t, 1] = theta[t-1, 1] + omega[1] + K * np.sin(theta[t-1, 0] - theta[t-1, 1])
        
        return np.sin(theta[:, 0]), np.sin(theta[:, 1])
    
    @staticmethod
    def ar_process(n_points, a=0.7, b=0.2, noise=0.1, seed=None):
        """
        Generate autoregressive process time series.
        
        Args:
            n_points (int): Number of time points to generate
            a (float): AR coefficient for X
            b (float): Coupling coefficient from X to Y
            noise (float): Noise amplitude
            seed (int): Random seed for reproducibility
            
        Returns:
            tuple: (X, Y) where Y is caused by X
        """
        if seed is not None:
            np.random.seed(seed)
            
        X = np.random.randn(n_points)
        Y = np.zeros(n_points)
        
        for t in range(1, n_points):
            X[t] = a * X[t-1] + np.random.randn() * noise
            Y[t] = b * X[t-1] + np.random.randn() * noise
        
        return X, Y
    
    @staticmethod
    def nonlinear_coupled_system(n_points, alpha=0.2, beta=0.1, noise=0.01, seed=None):
        """
        Generate nonlinear coupled system time series.
        
        Args:
            n_points (int): Number of time points to generate
            alpha (float): Coupling strength from X to Y
            beta (float): Coupling strength from Y to X
            noise (float): Noise amplitude
            seed (int): Random seed for reproducibility
            
        Returns:
            tuple: (X, Y) mutually coupled time series
        """
        if seed is not None:
            np.random.seed(seed)
            
        X = np.zeros(n_points)
        Y = np.zeros(n_points)
        
        X[0], Y[0] = np.random.rand(), np.random.rand()
        
        for t in range(1, n_points):
            X[t] = np.tanh(X[t-1]) + beta * Y[t-1] + np.random.randn() * noise
            Y[t] = np.tanh(Y[t-1]) + alpha * X[t-1] + np.random.randn() * noise
        
        return X, Y

# Example usage:
if __name__ == "__main__":
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
