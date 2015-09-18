import theano
import theano.typed_list
import theano.tensor as T
import numpy as np
from sklearn.datasets import load_digits

# from sklearn.preprocessing import LabelBinarizer
# from scipy import optimize
# from collections import namedtuple
# import matplotlib.pyplot as plt
# import os


np.seterr(all='raise')
# theano.config.optimizer = 'fast_compile'

# Define Theano Functions
# 1. Calc Utilities matrix, dims=(alternatives, experiments)
X = T.matrix('X', dtype='float64')
W = T.matrix('W', dtype='float64')
b = T.vector('b', dtype='float64')
calculate_utilities = theano.function([X, W, b], T.dot(X, W) + b, name='calculate_utilities')

# 2. Exp of scaled utilities
V = T.matrix('V', dtype='float64')
l_expanded = T.vector('l_expanded', dtype='float64')
calculate_exp_V = theano.function([V, l_expanded], T.exp(V / l_expanded), name='calculate_exp_V')


# 3. Sum by nest
exp_V = T.matrix('exp_V', dtype='float64')
nests = T.vector('nests', dtype='int64')
alt_indices = theano.typed_list.TypedListType(T.lvector)()  # T.matrix('alt_indices', dtype='int32')
nest_sums, _ = theano.scan(lambda i_, alt_indices_, exp_V_: exp_V_[:, alt_indices_[i_]].sum(axis=1),
                           sequences=[nests],
                           non_sequences=[alt_indices, exp_V])
calculate_nest_sums = theano.function([exp_V, nests, alt_indices], nest_sums.T, name='calculate_nest_sums')


# 4. Calc probs
lambdas = T.vector('lambdas', dtype='float64')
alternatives = T.vector('alternatives', dtype='int64')
nest_indices = T.vector('nest_indices', dtype='int64')
exp_V = T.matrix('exp_V', dtype='float64')
nest_sums = T.matrix('nest_sums', dtype='float64')
denominator = np.power(nest_sums, lambdas).sum(axis=1)
# for i in range(alternatives):
#     nest_sum = nest_sums[:, nest_indices[i]]
#     numerator = exp_V[:, i] * np.power(nest_sum, lambdas[nest_indices][i] - 1)
#     P[:, i] = numerator / denominator


def calculate_prob(alt_, lambdas_, nest_indices_, exp_V_, nest_sums_, denominator_):
    numerator = exp_V_[:, alt_] * np.power(nest_sums_[:, nest_indices_[alt_]], lambdas_[nest_indices_][alt_] - 1)
    return numerator / denominator_

P, _ = theano.scan(calculate_prob,
                   sequences=[alternatives],
                   non_sequences=[lambdas, nest_indices, exp_V, nest_sums, denominator])
calculate_probs = theano.function([alternatives, lambdas, nest_indices, exp_V, nest_sums],
                                  P.T, name='calculate_probs')

# Calculate results
P = T.matrix('P', dtype='float64')
calculate_predictions = theano.function([P], T.argmax(P, axis=1), name='calculate_predictions')

predictions = T.vector('predictions', dtype='int64')
y = T.vector('y', dtype='int64')
calculate_error = theano.function([predictions, y], T.mean(T.neq(predictions, y)), name='calculate_error')


# Calc cost
P = T.matrix('P', dtype='float64')
y = T.vector('y', dtype='int64')
calculate_cost = theano.function([P, y], -T.mean(T.log(P)[T.arange(y.shape[0]), y]), name='calculate_cost')


# Setup
alternatives = 10
nests = np.array([0, 1, 2], dtype='int32')
nest_indices = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
alt_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
lambdas = np.array([1, 2, 3])
num_nests = lambdas.shape[0]
# lambdas_expanded = lambdas[nest_indices]

digits = load_digits()
data_shape = digits.data.shape

W_input = np.random.rand(data_shape[1], alternatives)
P = np.zeros((data_shape[0], alternatives))
b_input = np.zeros(alternatives)

V = calculate_utilities(digits.data, W_input, b_input)
exp_V = calculate_exp_V(V, lambdas[nest_indices])
nest_sums = calculate_nest_sums(exp_V, nests, alt_indices)
P = calculate_probs(np.arange(alternatives), lambdas, nest_indices, exp_V, nest_sums)

predictions = calculate_predictions(P)
error = calculate_error(predictions, digits.target)
cost = calculate_cost(P, digits.target)

print(error)
print(cost)

# Do This!!!
# I think we need to make cost a function => f(X, ...) = cost
# T.grad(cost, wrt=X)


# mnl_probs = T.nnet.softmax(V)
# mnl_pred = T.argmax(mnl_probs, axis=1)
# mnl_error = T.mean(T.neq(mnl_pred, digits.target))

# params = [W, b]
# input = input
