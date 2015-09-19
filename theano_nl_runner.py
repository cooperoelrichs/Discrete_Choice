import numpy as np
from sklearn.datasets import load_digits
from nested_logit_estimator import NestedLogitEstimator
import theano.tensor as T
import theano

# Setup
alternatives = 10
nests = np.array([0, 1, 2], dtype='int32')
nest_indices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
# alt_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
lambdas = np.array([1, 2, 3])
# num_nests = lambdas.shape[0]
# lambdas_expanded = lambdas[nest_indices]

digits = load_digits()
data_shape = digits.data.shape

W_input = np.random.rand(data_shape[1], alternatives)
b_input = np.zeros(alternatives)
# P = np.zeros((data_shape[0], alternatives))

X = digits.data
y = digits.target
nle = NestedLogitEstimator(X, y, W_input, b_input, lambdas, nests, nest_indices, alternatives)

x = theano.shared(value=np.ones((5, 10)), name='W', borrow=True)

cost, error, _ = nle.results(nle.initial_W, nle.initial_b, nle.initial_lambdas)

print(error)
print(cost)

W_grad, b_grad, l_grad = nle.gradient(nle.initial_W, nle.initial_b, nle.initial_lambdas)

print(W_grad.shape)
print(b_grad.shape)
print(l_grad.shape)

cost, error, _, W, b, lambdas = nle.estimate()

print(error)
print(cost)
print(lambdas)

print('Error is: %.2f' % (error * 100))

# mnl_probs = T.nnet.softmax(V)
# mnl_pred = T.argmax(mnl_probs, axis=1)
# mnl_error = T.mean(T.neq(mnl_pred, digits.target))

# params = [W, b]
# input = input
