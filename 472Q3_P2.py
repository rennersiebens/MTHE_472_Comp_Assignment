import numpy as np

# Define system parameters
A = np.array([[2, 1, 0, 0],
              [0, 2, 1, 0],
              [0, 0, 2, 1],
              [0, 0, 0, 4]])

C = np.array([[2, 0, 0, 0]])

Noise_Covariance = np.eye(4)

v_t_variance = 1

# Define Riccati recursion function
def riccati_recursion(A, C, W, V, P0):
    P = P0
    max_iter = 1000  # Maximum number of iterations
    tol = 1e-6  # Convergence tolerance
    for _ in range(max_iter):
        P_new = A @ P @ A.T + W - A @ P @ C.T @ np.linalg.inv(C @ P @ C.T + V) @ C @ P @ A.T
        if np.linalg.norm(P_new - P) < tol:
            return P_new
        P = P_new
    return P

# Set initial conditions
P0_1 = np.eye(4)  # Initial covariance matrix 1
P0_2 = np.eye(4) * 3  # Initial covariance matrix 2

# Compute Riccati recursions for both initial conditions
P_final_1 = riccati_recursion(A, C, Noise_Covariance, v_t_variance, P0_1)
P_final_2 = riccati_recursion(A, C, Noise_Covariance, v_t_variance, P0_2)

print("Final covariance matrix (for initial condition 1):\n", P_final_1)
print("Final covariance matrix (for initial condition 2):\n", P_final_2)

