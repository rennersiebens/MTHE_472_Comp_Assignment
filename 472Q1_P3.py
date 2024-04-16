import numpy as np

# Transition probabilities
transition_probs = {
    ('G', 1): {'G': 0.1, 'B': 0.9},
    ('G', 0): {'G': 0.9, 'B': 0.1},
    ('B', 1): {'G': 0.8, 'B': 0.2},
    ('B', 0): {'G': 0.5, 'B': 0.5}
}

# Per-stage cost function
def cost(x, u, eta):
    cost = 0
    if x == 'G' and u == 1:
        cost = -1
    cost += eta*u
    return cost

# Q-learning function
def q_learning(transition_probs, cost, eta, alpha, beta, num_iterations=1000):
    states = ['G', 'B']
    actions = [0, 1]  # 0: Not using channel, 1: Using channel
    
    # Initialize Q-values arbitrarily
    Q = np.zeros((len(states), len(actions)))
    
    for iterations in range(num_iterations):
        state = np.random.choice(states)  # Initial state
        for t in range(100):  # Maximum time steps per episode
            action = np.random.choice(actions)  # Choose action using epsilon-greedy policy
            next_state = np.random.choice(states, p=[transition_probs[(state, action)][s] for s in states])
            costing = cost(state, action, eta)
            #Calculate the value that will be multiplied by alpha
            multiply_by_alpha = costing + beta * np.min(Q[states.index(next_state)]) - Q[states.index(state), action]
            Q[states.index(state), action] += alpha(t, state, action) * multiply_by_alpha
            
            state = next_state
            
    return Q

# Parameters
eta = 0.01
#Calculating the alpha value by the time step we are in and the state and action space.
alpha = lambda t, x, u: 1 / (1 + np.sum([1 for k in range(t + 1) if x == 'G' and u == 1]))
beta = 0.7

# Run Q-learning
Q_values = q_learning(transition_probs, cost, eta, alpha, beta)

# Print the learned Q-values
print("Learned Q-values:")
for i, state in enumerate(['G', 'B']):
    for j, action in enumerate(['Not using channel', 'Using channel']):
        print(f"Q({state}, {action}): {Q_values[i, j]}")
