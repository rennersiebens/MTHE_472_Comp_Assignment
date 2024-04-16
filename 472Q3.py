import numpy as np
import matplotlib.pyplot as plt

# Define system parameters
A = np.array([[2, 1, 0, 0],
              [0, 2, 1, 0],
              [0, 0, 2, 1],
              [0, 0, 0, 4]])

C = np.array([[2, 0, 0, 0]])

Noise_Covariance = np.eye(4)

v_t_variance = 1

# initial state
covariance_matrix = np.eye(4)

# Initial State
x0 = np.zeros((4, 1))


def Kalman_equation(A, C, W, yt, v, covariance_matrix_previous, mt_previous):
    inverse_caclulation = np.linalg.inv(C @ covariance_matrix_previous @ C.T + 1)
    mt_new = A @ mt_previous + A @ covariance_matrix_previous @ C.T @ inverse_caclulation @ (yt - (C @ mt_previous))

    new_covariance = A @ covariance_matrix_previous @ A.T + W - A @ covariance_matrix_previous @ C.T @ inverse_caclulation @ C @ covariance_matrix_previous @ A.T

    return mt_new, new_covariance


def update_x(A, x_previous, wt):
    x_new = A @ x_previous + wt
    return x_new


T = 1000
xt = np.zeros((4, T + 1))
yt = np.zeros((1, T + 1))
m_tilde = np.zeros((4, T + 1))
x_minus_m_tilde = np.zeros((4, T + 1))
x = x0
m_new = np.zeros((4, 1))

for t in range(T + 1):  # Adjusted range to include 1000
    # Generate process and measurement noise
    wt = np.random.multivariate_normal(mean=np.zeros(4), cov=Noise_Covariance).reshape(-1, 1)
    vt = np.random.normal(0, 1)

    y = C @ x + vt

    # Kalman filter update
    m_new, covariance_matrix = Kalman_equation(A, C, Noise_Covariance, y, v_t_variance, covariance_matrix, m_new)
    x = update_x(A, x, wt)

    # Store values
    xt[:, t] = x.flatten()
    m_tilde[:, t] = m_new.flatten()
    x_minus_m_tilde[:, t] = (x - m_new).flatten()

# Plot results
plt.figure(figsize=(12, 8))

# Plot x(t)
plt.subplot(3, 1, 1)
plt.plot(range(T + 1), xt[3], label='x(t)', color='blue')  # Adjusted range
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('True State (x(t))')
plt.legend()

# Plot m_tilda(t)
plt.subplot(3, 1, 2)
plt.plot(range(T + 1), m_tilde[3], label='m~(t)', color='red')  # Adjusted range
plt.xlabel('Time')
plt.ylabel('m~(t)')
plt.title('Estimated State (m~(t))')
plt.legend()

# Plot (m_tilda(t) - x(t))
plt.subplot(3, 1, 3)
plt.plot(range(T + 1), x_minus_m_tilde[3], label='m~(t) - x(t)', color='green')  # Adjusted range
plt.xlabel('Time')
plt.ylabel('m~(t) - x(t)')
plt.title('Difference (x(t) - m~(t))')
plt.legend()

plt.tight_layout()
plt.show()
