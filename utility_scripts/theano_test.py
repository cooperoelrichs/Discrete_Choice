import theano
import theano.tensor as T
import numpy as np

X = T.matrix('X', dtype='float64')
W = T.matrix('W', dtype='float64')

# a[[b[:,0],b[:,1]]] = c[b[:,2]]
# W_mod = T.set_subtensor(W[[[0, 1], [0, 1]]], np.array([1, 2]))
W_mod = W

V = T.dot(X, W_mod)

f = theano.function(inputs=[X, W], outputs=V, name='dot_prod')

X_input = np.ones((5, 4))
W_input = np.zeros((4, 2))
W_input[[[0, 1], [0, 1]]] = np.array([1, 2])

V_output = f(X_input, W_input)
print(V_output.shape)
print(V_output)
