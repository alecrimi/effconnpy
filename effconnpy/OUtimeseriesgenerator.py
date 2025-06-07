import numpy

# Simulation parameters
T = 100.0              # total time
dt = 1              # time step
N = int(T / dt)        # number of steps
time = np.linspace(0, T, N)

# Initialize arrays
X = np.zeros((3, N))   # 3 nodes, N time points

# OU process parameters
mu = np.array([0.0, 0.0, 0.0])             # long-term means
theta = np.array([0.3 , 0.3, 0.3])          # reversion rates
sigma = np.array([0.01, 0.01, 0.01])          # noise strengths

# Causal coefficients (off-diagonal elements)
alpha_21 = 1.5  # X1 → X2
alpha_32 = 1.5  # X2 → X3

# Simulation loop (Euler-Maruyama method)
for i in range(1, N):
    x = X[:, i-1]
    dW = np.random.normal(0, np.sqrt(dt), size=3)  # Brownian increments

    dx1 = -theta[0] * (x[0] - mu[0]) * dt + sigma[0] * dW[0]
    dx2 = (-theta[1] * (x[1] - mu[1]) + alpha_21 * (x[0] - mu[0])) * dt + sigma[1] * dW[1]
    dx3 = (-theta[2] * (x[2] - mu[2]) + alpha_32 * (x[1] - mu[1])) * dt + sigma[2] * dW[2]

    X[0, i] = x[0] + dx1
    X[1, i] = x[1] + dx2
    X[2, i] = np.abs(x[2] + dx3)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(time, X[0], label='X1 (OU): Temperature')
plt.plot(time, X[1], label='X2 (OU + X1 → X2): Pressure')
plt.plot(time, X[2], label='X3 (OU + X2 → X3): Failed component')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('3-Node Causal OU Network Simulation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
