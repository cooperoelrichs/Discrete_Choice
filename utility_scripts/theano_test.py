import theano
import theano.tensor as T
import numpy as np

X = np.zeros((2, 2))
X[[[0, 1], [0, 1]]] = np.array([1, 2])
print(X)


X = theano.shared(value=np.zeros((2, 2)))
X = T.set_subtensor(X[[[0, 1], [0, 1]]], np.array([1, 2]))
X.eval()
