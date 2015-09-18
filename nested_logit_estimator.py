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


class TheanoNestedLogitEstimator(object):
    def __init__(self):
        np.seterr(all='raise')
        # theano.config.optimizer = 'fast_compile'  # More traceable errors

        self.X = T.matrix('X', dtype='float64')
        self.W = T.matrix('W', dtype='float64')
        self.b = T.vector('b', dtype='float64')

        self.l_expanded = T.vector('l_expanded', dtype='float64')
        self.nests = T.vector('nests', dtype='int64')
        self.alt_indices = theano.typed_list.TypedListType(T.lvector)()

    def calculate_utilities(self, X, W, b):
        V = T.dot(X, W) + b
        return V

    def calculate_exp_V(self, V, l_expanded):
        exp_V = T.exp(V / l_expanded)
        return exp_V

    def calculate_nest_sums(self, exp_V, nests, alt_indices):
        nest_sums_T, _ = theano.scan(lambda i, alt_indices, exp_V: exp_V[:, alt_indices[i]].sum(axis=1),
                                     sequences=[nests],
                                     non_sequences=[alt_indices, exp_V])
        return nest_sums_T.T

    def calculate_probabilities_alternative(self, alt, lambdas, nest_indices, exp_V, nest_sums, denominator):
        numerator = exp_V[:, alt] * np.power(nest_sums[:, nest_indices[alt]], lambdas[nest_indices][alt] - 1)
        return numerator / denominator

    def calculate_probabilities(self, exp_V, nest_sums, lambdas, alternatives, nest_indices):
        denominator = np.power(nest_sums, lambdas).sum(axis=1)

        P_T, _ = theano.scan(self.calculate_probabilities_alternative,
                             sequences=[alternatives],
                             non_sequences=[lambdas, nest_indices, exp_V, nest_sums, denominator])
        return P_T.T

    def calculate_predictions(self, P):
        predictions = T.argmax(P, axis=1)
        return predictions

    def calculate_error(self, predictions, y):
        return T.mean(T.neq(predictions, y))

    def calculate_cost(self, P, y):
        cost = -T.mean(T.log(P)[T.arange(y.shape[0]), y])
        return cost


    def cost_function(self):
        self.X = digits.data
        V = self.calculate_utilities(self.X, self.W, self.b)
        exp_V = self.calculate_exp_V(V, lambdas[nest_indices])
        nest_sums = self.calculate_nest_sums(exp_V, nests, alt_indices)
        P = self.calculate_probs(np.arange(alternatives), lambdas, nest_indices, exp_V, nest_sums)

        predictions = self.calculate_predictions(P)
        error = self.calculate_error(predictions, digits.target)
        cost = self.calculate_cost(P, digits.target)
        return error, cost

    def gradient_function(self):
        W_grad = T.grad(self.cost, wrt=self.W)
        b_grad = T.grad(self.cost, wrt=self.b)


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
