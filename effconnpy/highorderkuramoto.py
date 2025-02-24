import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class HighOrderKuramotoModel:
    def __init__(self, n_oscillators, natural_frequencies=None, 
                 coupling_matrices=None, max_order=2, seed=None):
        """
        Initialize a high-order Kuramoto model.
        
        Parameters:
        -----------
        n_oscillators : int
            Number of oscillators in the network
        natural_frequencies : array-like, optional
            Natural frequencies of oscillators. If None, random frequencies are generated.
        coupling_matrices : dict, optional
            Dictionary with keys representing order and values as coupling matrices.
            If None, random coupling matrices are generated.
        max_order : int, optional
            Maximum order of interactions to consider (default: 2)
        seed : int, optional
            Random seed for reproducibility
        """
        self.n = n_oscillators
        self.max_order = max_order
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize natural frequencies
        if natural_frequencies is None:
            self.natural_frequencies = np.random.normal(0, 1, n_oscillators)
        else:
            self.natural_frequencies = np.array(natural_frequencies)
        
        # Initialize coupling matrices for different orders
        self.coupling_matrices = {}
        if coupling_matrices is None:
            # First order (traditional Kuramoto)
            self.coupling_matrices[1] = np.random.normal(0, 1.0, (n_oscillators, n_oscillators))
            
            # Higher orders
            for order in range(2, max_order + 1):
                shape = tuple([n_oscillators] * (order + 1))
                self.coupling_matrices[order] = np.random.normal(0, 1.0/order, shape)
        else:
            self.coupling_matrices = coupling_matrices
    
    def _phase_derivative(self, t, phases, external_input=None, time_points=None):
        """
        Calculate the derivative of phases for the differential equation solver.
        
        Parameters:
        -----------
        t : float
            Current time point
        phases : array
            Current phases of oscillators
        external_input : array, optional
            External input time series data with shape (n_oscillators, n_timepoints)
        time_points : array, optional
            Time points corresponding to external_input
        """
        d_phases = np.zeros(self.n)
        
        # Add natural frequencies
        d_phases += self.natural_frequencies
        
        # Add external input if provided
        if external_input is not None and time_points is not None:
            # Find the closest time point in the external input data
            idx = np.abs(time_points - t).argmin()
            d_phases += external_input[:, idx]
        
        # First order interactions (traditional Kuramoto)
        if 1 in self.coupling_matrices:
            K1 = self.coupling_matrices[1]
            for i in range(self.n):
                for j in range(self.n):
                    d_phases[i] += K1[i, j] * np.sin(phases[j] - phases[i])
        
        # Second order interactions (triplet)
        if 2 in self.coupling_matrices and self.max_order >= 2:
            K2 = self.coupling_matrices[2]
            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        d_phases[i] += K2[i, j, k] * np.sin(phases[j] + phases[k] - 2 * phases[i])
        
        # Third order interactions (quadruplet)
        if 3 in self.coupling_matrices and self.max_order >= 3:
            K3 = self.coupling_matrices[3]
            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        for l in range(self.n):
                            d_phases[i] += K3[i, j, k, l] * np.sin(phases[j] + phases[k] + phases[l] - 3 * phases[i])
        
        return d_phases
    
    def simulate(self, t_span, initial_phases=None, t_points=1000, external_input=None, time_points=None):
        """
        Simulate the dynamics of the high-order Kuramoto model.
        
        Parameters:
        -----------
        t_span : tuple
            Time span for simulation (t_start, t_end)
        initial_phases : array-like, optional
            Initial phases of oscillators. If None, random phases are generated.
        t_points : int, optional
            Number of time points to evaluate
        external_input : array, optional
            External input time series with shape (n_oscillators, n_timepoints)
        time_points : array, optional
            Time points corresponding to external_input
            
        Returns:
        --------
        t : array
            Time points
        phases : array
            Phases of oscillators at each time point
        """
        if initial_phases is None:
            initial_phases = 2 * np.pi * np.random.rand(self.n)
        
        t_eval = np.linspace(t_span[0], t_span[1], t_points)
        
        # Create a wrapper for the phase derivative function to include external input
        if external_input is not None and time_points is not None:
            phase_derivative_with_input = lambda t, y: self._phase_derivative(t, y, external_input, time_points)
        else:
            phase_derivative_with_input = self._phase_derivative
        
        solution = solve_ivp(
            phase_derivative_with_input, 
            t_span, 
            initial_phases,
            t_eval=t_eval,
            method='RK45'
        )
        
        return solution.t, solution.y
    
    def fit_model(self, time_series_data, time_points, max_iterations=1000, 
                 learning_rate=0.01, tol=1e-4, verbose=True):
        """
        Fit the model parameters to observed time series data.
        
        Parameters:
        -----------
        time_series_data : array
            Observed time series data with shape (n_oscillators, n_timepoints)
        time_points : array
            Time points corresponding to time_series_data
        max_iterations : int, optional
            Maximum number of iterations for optimization
        learning_rate : float, optional
            Learning rate for gradient descent
        tol : float, optional
            Tolerance for convergence
        verbose : bool, optional
            Whether to print progress
            
        Returns:
        --------
        loss_history : array
            History of loss values during optimization
        """
        # Extract phases from time series data (assuming data is phase or can be converted to phase)
        observed_phases = self._preprocess_time_series(time_series_data)
        
        # Initialize parameters to optimize
        opt_frequencies = self.natural_frequencies.copy()
        opt_coupling = {order: self.coupling_matrices[order].copy() 
                        for order in self.coupling_matrices}
        
        # Initialize loss history
        loss_history = []
        
        # Optimization loop
        for iteration in range(max_iterations):
            # Forward pass: simulate with current parameters
            t_span = (time_points[0], time_points[-1])
            initial_phases = observed_phases[:, 0]
            
            # Update model parameters
            self.natural_frequencies = opt_frequencies
            self.coupling_matrices = opt_coupling
            
            # Simulate
            _, simulated_phases = self.simulate(t_span, initial_phases, 
                                              len(time_points))
            
            # Calculate loss (mean squared error)
            loss = np.mean((simulated_phases - observed_phases) ** 2)
            loss_history.append(loss)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.6f}")
            
            # Check convergence
            if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Calculate gradients and update parameters (simplified gradient descent)
            # This is a simplified approach - in practice, automatic differentiation 
            # or more sophisticated optimization would be used
            
            # Update natural frequencies
            freq_grad = self._calculate_frequency_gradient(observed_phases, simulated_phases)
            opt_frequencies -= learning_rate * freq_grad
            
            # Update coupling matrices (simplified)
            for order in opt_coupling:
                coupling_grad = self._calculate_coupling_gradient(
                    observed_phases, simulated_phases, order)
                opt_coupling[order] -= learning_rate * coupling_grad
        
        return loss_history
    
    def _preprocess_time_series(self, time_series_data):
        """
        Preprocess time series data to extract phases.
        
        This method can be customized based on the nature of input data.
        Current implementation assumes data is already in phase form or
        can be directly interpreted as phases.
        
        Parameters:
        -----------
        time_series_data : array
            Time series data with shape (n_oscillators, n_timepoints)
            
        Returns:
        --------
        phases : array
            Phases extracted from time series data
        """
        # If data is complex, extract phase
        if np.iscomplexobj(time_series_data):
            return np.angle(time_series_data)
        
        # If data is real, convert to phase in [-π, π]
        # This is a simple approach - more sophisticated methods can be used
        # depending on the nature of the data
        normalized_data = (time_series_data - np.mean(time_series_data, axis=1, keepdims=True)) / \
                         (np.std(time_series_data, axis=1, keepdims=True) + 1e-10)
        return np.arctan2(normalized_data, np.abs(normalized_data))
    
    def _calculate_frequency_gradient(self, observed, simulated):
        """Calculate gradient for natural frequencies."""
        # Simplified gradient calculation
        return 2 * np.mean(simulated - observed, axis=1)
    
    def _calculate_coupling_gradient(self, observed, simulated, order):
        """Calculate gradient for coupling matrices."""
        # Simplified gradient calculation - in practice, this would be more complex
        # and would depend on the specific interaction terms
        if order == 1:
            grad = np.zeros_like(self.coupling_matrices[order])
            for i in range(self.n):
                for j in range(self.n):
                    # Simplified gradient - assumes sin(phase_j - phase_i) term
                    phase_diff = simulated[j, :] - simulated[i, :]
                    grad[i, j] = np.mean(2 * (simulated[i, :] - observed[i, :]) * np.sin(phase_diff))
            return grad
        
        # For higher orders, return a small random gradient (placeholder)
        # In a real implementation, this would be computed properly
        return np.random.normal(0, 0.01, self.coupling_matrices[order].shape)
    
    def calculate_order_parameter(self, phases):
        """Calculate the Kuramoto order parameter r(t)."""
        complex_phases = np.exp(1j * phases)
        r = np.abs(np.mean(complex_phases, axis=0))
        psi = np.angle(np.mean(complex_phases, axis=0))
        return r, psi
    
    def analyze_time_series(self, time_series_data, time_points, title="Time Series Analysis"):
        """
        Analyze time series data using the Kuramoto framework.
        
        Parameters:
        -----------
        time_series_data : array
            Time series data with shape (n_oscillators, n_timepoints)
        time_points : array
            Time points corresponding to time_series_data
        title : str, optional
            Title for the visualization
            
        Returns:
        --------
        results : dict
            Dictionary with analysis results
        """
        # Extract phases
        phases = self._preprocess_time_series(time_series_data)
        
        # Calculate order parameter
        r, psi = self.calculate_order_parameter(phases)
        
        # Calculate phase coherence matrix
        n_oscillators = phases.shape[0]
        coherence = np.zeros((n_oscillators, n_oscillators))
        for i in range(n_oscillators):
            for j in range(n_oscillators):
                # Mean phase coherence
                phase_diff = phases[i, :] - phases[j, :]
                coherence[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        # Calculate mean frequency
        mean_freq = np.zeros(n_oscillators)
        for i in range(n_oscillators):
            # Estimate frequency from phase time derivative
            phase_unwrapped = np.unwrap(phases[i, :])
            mean_freq[i] = np.mean(np.diff(phase_unwrapped) / np.diff(time_points))
        
        # Visualize
        self.visualize_analysis(time_points, phases, r, coherence, mean_freq, title)
        
        # Return results
        results = {
            'phases': phases,
            'order_parameter': r,
            'global_phase': psi,
            'coherence_matrix': coherence,
            'mean_frequency': mean_freq
        }
        
        return results
    
    def visualize(self, t, phases, title="High-Order Kuramoto Model Dynamics"):
        """Visualize the simulation results."""
        plt.figure(figsize=(12, 8))
        
        # Plot phase evolution
        plt.subplot(2, 1, 1)
        for i in range(self.n):
            plt.plot(t, phases[i, :], label=f"Oscillator {i+1}")
        plt.xlabel("Time")
        plt.ylabel("Phase")
        plt.title(title)
        if self.n <= 10:  # Only show legend if there aren't too many oscillators
            plt.legend()
        
        # Plot order parameter
        plt.subplot(2, 1, 2)
        r, psi = self.calculate_order_parameter(phases)
        plt.plot(t, r)
        plt.xlabel("Time")
        plt.ylabel("Order Parameter (r)")
        plt.title("Synchronization Level")
        
        plt.tight_layout()
        plt.show()
    
    def visualize_analysis(self, t, phases, order_param, coherence, mean_freq, title):
        """Visualize the analysis results."""
        plt.figure(figsize=(15, 10))
        
        # Plot phase evolution
        plt.subplot(2, 2, 1)
        for i in range(self.n):
            plt.plot(t, phases[i, :], label=f"Oscillator {i+1}")
        plt.xlabel("Time")
        plt.ylabel("Phase")
        plt.title("Phase Evolution")
        if self.n <= 10:
            plt.legend()
        
        # Plot order parameter
        plt.subplot(2, 2, 2)
        plt.plot(t, order_param)
        plt.xlabel("Time")
        plt.ylabel("Order Parameter (r)")
        plt.title("Synchronization Level")
        
        # Plot coherence matrix
        plt.subplot(2, 2, 3)
        plt.imshow(coherence, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label="Phase Coherence")
        plt.title("Phase Coherence Matrix")
        plt.xlabel("Oscillator")
        plt.ylabel("Oscillator")
        
        # Plot frequency distribution
        plt.subplot(2, 2, 4)
        plt.bar(range(1, self.n + 1), mean_freq / (2 * np.pi))
        plt.xlabel("Oscillator")
        plt.ylabel("Mean Frequency (Hz)")
        plt.title("Frequency Distribution")
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

# Example usage with time series input
if __name__ == "__main__":
    # Set parameters
    n_oscillators = 5
    n_timepoints = 1000
    
    # Generate sample time series (synthetic data)
    t = np.linspace(0, 100, n_timepoints)
    
    # Create synthetic data: each oscillator has a different frequency
    # and there's some noise to make it more realistic
    synthetic_data = np.zeros((n_oscillators, n_timepoints))
    frequencies = np.linspace(0.05, 0.15, n_oscillators)
    
    for i in range(n_oscillators):
        # Base oscillation
        synthetic_data[i, :] = np.sin(2 * np.pi * frequencies[i] * t)
        
        # Add noise
        synthetic_data[i, :] += 0.1 * np.random.randn(n_timepoints)
        
        # Add some coupling effect (simplified)
        if i > 0:
            synthetic_data[i, :] += 0.05 * synthetic_data[i-1, :]
    
    # Create model
    model = HighOrderKuramotoModel(
        n_oscillators=n_oscillators,
        max_order=2,
        seed=42
    )
    
    # Analyze the time series
    results = model.analyze_time_series(synthetic_data, t, "Analysis of Synthetic Time Series")
    
    # Simulate model with external input
    t_span = (0, 100)
    initial_phases = np.random.uniform(0, 2*np.pi, n_oscillators)
    
    # Use the synthetic data as external input
    t_sim, phases_sim = model.simulate(
        t_span, 
        initial_phases, 
        n_timepoints,
        external_input=synthetic_data,
        time_points=t
    )
    
    # Visualize simulation results
    model.visualize(t_sim, phases_sim, "Kuramoto Model with External Input")
    
    # Optional: fit the model to the time series data
    # This can be computationally intensive
    print("Fitting model to data (simplified demonstration)...")
    loss_history = model.fit_model(
        synthetic_data, t, 
        max_iterations=50,  # Small number for demonstration
        learning_rate=0.005,
        verbose=True
    )
    
    # Plot loss history
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Model Fitting Loss History")
    plt.yscale('log')
    plt.grid(True)
    plt.show()
