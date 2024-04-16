from scipy.optimize import linprog
eta = 0.01 #This value was changed for each eta testing value. 

# Define Cost vector
c = [0, eta, 0, eta - 1] # [c(B, 0), c(B, 1), c(G,0), c(G,1)] with eta = 0.9

# Define transition probabilities
A = [[1, 1, 1, 1],
     [-0.5, -0.8, 0.1, 0.9],
     [0.5, 0.8, -0.1, -0.9]]

b = [1, 0, 0]

# Solve the linear program
res = linprog(c, A_eq=A, b_eq=b)

# Print the result
print("Optimal probabilities:")
print("p_B0 =", res.x[0])
print("p_B1 =", res.x[1])
print("p_G0 =", res.x[2])
print("p_G1 =", res.x[3])

print("Gamma (0|B) = ", res.x[0]/(res.x[0] + res.x[1]))
print("Gamma (1|B) = ", res.x[1]/(res.x[0] + res.x[1]))
print("Gamma (0|G) = ", res.x[2]/(res.x[2] + res.x[3]))
print("Gamma (1|G) = ", res.x[3]/(res.x[2] + res.x[3]))

