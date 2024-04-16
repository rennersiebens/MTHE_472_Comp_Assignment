import numpy as np

#Define State Space, Action Space and Parameters

X = [0, 1] # 0 is Bad, 1 is Good
U = [0, 1]
beta = 0.7 # Arbitrarily fixed beta value between 0 and 1
eta_parameters = 0.01 # Inituialize the different parameters associated with cost function, changed each iteration
N_states = len(X)
N_actions = len(U)

T = np.zeros((len(X), len(X), len(U)))
#Transition Probabilities defined in transition matrix form. T[next state, previous_state, action]
T[0, 0, 0] = 0.5
T[1, 0, 0] = 0.5
T[0, 1, 0] = 0.1
T[0, 0, 1] = 0.2
T[0, 1, 1] = 0.9
T[1, 1, 0] = 0.9
T[1, 0, 1] = 0.8
T[1, 1, 1] = 0.1

# cost function
def cost_function (current_step, action, eta_paramater):
    cost = 0
    if current_step == 1 and action == 1:
        cost = -1
    cost += action*eta_paramater
    return cost


def policy_iteration (eta):
        is_value_changed = True
        iterations = 0
        while is_value_changed and iterations < 100:
            iterations += 1
            print("After ", iterations, " iterations:")
            is_value_changed = False
            
            for s in range(N_states):
                # print "State", s, "q_best", q_best
                for a in range (N_actions):
                    v_new  = cost_function(s, a, eta) + beta * sum (V_min[s1]*T[s, s1, a] for s1 in range (N_states))
                    if v_new < V_min [s]:
                        print ("State", s, ": q_sa", v_new, "q_best", V_min[s])
                        policy[s] = a
                        V_min [s] = v_new
                        is_value_changed = True

        print ("Iterations:", iterations)
        # print "Policy now", policy

        print ("Final policy")
        print (policy)
        print (V_min)



# initialize policy and value arbitrarily
policy = [0 for s in range(N_states)]
V_min = np.zeros(N_states)
policy_iteration(eta_parameters)