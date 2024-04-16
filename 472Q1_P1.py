import numpy as np

#Define State Space, Action Space and Parameters

X = ['B', 'G']
U = [0, 1]
beta = 0.75 # Arbitrarily fixed beta value between 0 and 1
eta_parameters = [0.9, 0.7, 0.01] # Inituialize the different parameters associated with cost function

#Transition Probabilities defined in transition matrix form

#start with transition matrix for u = 0:
Transition_Dont_Use = np.array([[0.5, 0.5], [0.1, 0.9]])
# Transition matrix for u = 1:
Transition_Use = np.array([[0.2, 0.8], [0.9, 0.1]])

# cost function
def cost_function (current_step, action, eta_paramater):
    cost = 0
    if current_step == 1 and action == 1:
        cost = -1
    cost += action*eta_paramater
    return cost
         


# Value iteration function
def value_iteration (eta_parameter):
    max_iterations = 1000
    convergence_threshold = 0.000001
    itera = 0
    v_value = np.zeros(2) #v(0) and v(1) initialized
    u_chosen = 0

    while itera < max_iterations:
        itera += 1
        v_new = np.zeros(2)
        chosen_u = 2
        u_new = np.zeros(2)
        for state_index, state in enumerate(X):
            for action_index, action in enumerate(U):
                if action == 0:
                    transition_matrix = Transition_Dont_Use
                else:
                    transition_matrix = Transition_Use
                test = transition_matrix[state_index, 0]
                test2 = transition_matrix[state_index, 1]
                test3 = v_value[0]
                test4 = v_value[1]
                calculation = cost_function(state_index, action_index, eta_parameter) + beta * (v_value[0]*transition_matrix[state_index, 0] + v_value[1]*transition_matrix[state_index, 1])
                v_new [state_index] = min(v_new[state_index], calculation)
                if v_new[state_index] > calculation:
                    chosen_u = action
                
        if np.max(np.abs(v_new - v_value)) < convergence_threshold:
            break
        
        v_value = v_new
    
    print (itera)
    return (v_value)



for eta in eta_parameters:
    V = value_iteration(eta)
    print("For the eta value of ", eta, "the optimal solution is: \n")
    print (V)







