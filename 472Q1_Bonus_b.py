import numpy as np

# Transition probabilities
def transition_probs(x, u):
    if u == 1:
        return lambda A: 2 * np.trapz((1 - x) * np.ones_like(A), A)
    else:
        return lambda A: 2 * np.trapz(x * np.ones_like(A), A)

# Per-stage cost function
def cost(x, u, eta):
    return -x * u + eta * u

# Quantization function
def quantize(x, num_levels):
    return np.digitize(x, np.linspace(0, 1, num_levels))

# Q-learning function
def quantized_q_learning(num_levels, eta, alpha, beta, num_episodes=1000):
    actions = [0, 1]  # 0: Not using channel, 1: Using channel
    
    # Initialize Q-values arbitrarily
    Q = np.zeros((num_levels, len(actions)))
    
    for episode in range(num_episodes):
        x = np.random.rand()  # Initial state
        state = quantize(x, num_levels)
        for t in range(100):  # Maximum time steps per episode
            action = np.random.choice(actions)  # Choose next action randomly.
            prob_func = transition_probs(x, action)
            next_x = np.random.rand()  # Sample next state
            next_state = quantize(next_x, num_levels) #quantize
            reward = cost(x, action, eta)
            td_target = reward + beta * np.min(Q[next_state])
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            
            x = next_x
            state = next_state
            
    return Q

# Parameters
eta = 0.7
alpha = 0.1
beta = 0.9
num_episodes = 1000

# Number of quantization levels (increase for finer quantization)
num_levels_list = [2, 4, 8, 16, 32]

# Run quantized Q-learning for different number of levels
for num_levels in num_levels_list:
    Q_values = quantized_q_learning(num_levels, eta, alpha, beta, num_episodes)
    print(f"Number of levels: {num_levels}")
    print("Learned Q-values:")
    print(Q_values)
    print("Average Q-value:", np.mean(Q_values))
    print("\n")
