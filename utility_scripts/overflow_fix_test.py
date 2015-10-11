
V = utilities_matrix
e_V = exp(V - V.mean(axis=1))
P = e_V / e_V.sum(axis=1)

