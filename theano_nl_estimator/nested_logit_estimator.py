import theano
import theano.typed_list
import theano.tensor as T
import numpy as np
from scipy import optimize

# from sklearn.preprocessing import LabelBinarizer
# from collections import namedtuple
# import matplotlib.pyplot as plt
# import os


class TheanoNestedLogit(object):
    def __init__(self):
        np.seterr(all='raise')
        theano.config.optimizer = 'fast_compile'  # More traceable errors
        theano.config.exception_verbosity = 'high'  # More traceable errors

        self.X = T.matrix('X', dtype='float64')
        self.y = T.vector('y', dtype='int64')
        self.initial_W = T.matrix('W', dtype='float64')
        self.initial_b = T.vector('b', dtype='float64')
        self.initial_l = T.vector('lambdas', dtype='float64')
        self.parameters = T.vector('parameters', dtype='float64')

        self.utility_functions = T.matrix('utility_functions', dtype='int64')
        self.biases = T.matrix('biases', dtype='int64')
        self.lambdas = T.matrix('lambdas', dtype='int64')

        self.nests = T.vector('nests', dtype='int64')
        self.nest_indices = T.vector('nest_indices', dtype='int64')
        # self.alt_indices = theano.typed_list.TypedListType(T.lvector)()
        self.alternatives = T.vector('alternatives', dtype='int64')

        self.cost, self.error, self.predictions = self.nested_logit_cost()

        self.cost_function = self.compile_cost_function()
        # self.gradient_function = self.compile_gradient_function()

    @staticmethod
    def calculate_utilities(X, W, b):
        V = T.dot(X, W) + b
        return V

    @staticmethod
    def calculate_exp_V(V, l, nest_indices):
        exp_V = T.exp(V / l[nest_indices])
        return exp_V

    @staticmethod
    def calculate_nest_sums(exp_V, nests, nest_indices):
        # nest_sums_T, _ = theano.scan(lambda i, alt_indices, exp_V: exp_V[:, alt_indices[i]].sum(axis=1),
        nest_sums_T, _ = theano.scan(
            lambda i, nest_indices, exp_V: exp_V[:, T.eq(nest_indices, i).nonzero()[0]].sum(axis=1),
            sequences=[nests],
            non_sequences=[nest_indices, exp_V]
        )
        return nest_sums_T.T

    @staticmethod
    def calculate_probability_for_alternative(alt, lambdas, nest_indices, exp_V, nest_sums, denominator):
        numerator = exp_V[:, alt] * np.power(nest_sums[:, nest_indices[alt]], lambdas[nest_indices][alt] - 1)
        return numerator / denominator

    def calculate_probabilities(self, exp_V, nest_sums, lambdas, alternatives, nest_indices):
        denominator = np.power(nest_sums, lambdas).sum(axis=1)
        P_T, _ = theano.scan(self.calculate_probability_for_alternative,
                             sequences=[alternatives],
                             non_sequences=[lambdas, nest_indices, exp_V, nest_sums, denominator])
        return P_T.T

    @staticmethod
    def calculate_predictions(P):
        predictions = T.argmax(P, axis=1)
        return predictions

    @staticmethod
    def calculate_error(predictions, y):
        return T.mean(T.neq(predictions, y))

    @staticmethod
    def calculate_cost(P, y):
        cost = -T.mean(T.log(P)[T.arange(y.shape[0]), y])
        # cost = -T.sum(T.log(P)[T.arange(y.shape[0]), y])
        return cost

    def nested_logit_cost(self):
        W = self.initial_W
        b = self.initial_b
        l = self.initial_l
        W = T.set_subtensor(W[[self.utility_functions[:, 0], self.utility_functions[:, 1]]],
                            self.parameters[self.utility_functions[:, 2]])
        b = T.set_subtensor(b[self.biases[:, 0]], self.parameters[self.biases[:, 1]])
        l = T.set_subtensor(l[self.lambdas[:, 0]], self.parameters[self.lambdas[:, 1]])

        V = self.calculate_utilities(self.X, W, b)
        exp_V = self.calculate_exp_V(V, l, self.nest_indices)
        nest_sums = self.calculate_nest_sums(exp_V, self.nests, self.nest_indices)
        P = self.calculate_probabilities(exp_V, nest_sums, l,
                                         self.alternatives, self.nest_indices)

        predictions = self.calculate_predictions(P)
        error = self.calculate_error(predictions, self.y)
        cost = self.calculate_cost(P, self.y)
        return cost, error, predictions

    def compile_cost_function(self):
        cost_function = theano.function([self.X, self.y,
                                         self.initial_W, self.initial_b, self.initial_l,
                                         self.utility_functions, self.biases, self.lambdas,
                                         self.parameters, self.alternatives,
                                         self.nest_indices, self.nests],
                                        [self.cost, self.error, self.predictions],
                                        name='cost_function')
        return cost_function

    def compile_gradient_function(self):
        grad = theano.function([self.X, self.y,
                                self.initial_W, self.initial_b, self.initial_l,
                                self.utility_functions, self.biases, self.lambdas,
                                self.parameters, self.alternatives, self.alternatives,
                                self.nest_indices, self.nests],
                               T.grad(self.cost, wrt=self.parameters))
        return grad


class NestedLogitEstimator(object):
    def __init__(self, X, y, initial_W, initial_b, initial_l, nests, nest_indices, alternatives,
                 parameters, utility_functions, biases, lambdas):
        self.X = X
        self.y = y

        self.initial_W = initial_W
        self.initial_b = initial_b

        self.initial_l = initial_l
        self.nests = nests
        self.nest_indices = nest_indices
        # self.alt_indices = alt_indices
        self.alternatives = np.arange(alternatives)
        self.parameters = parameters
        self.utility_functions = utility_functions
        self.biases = biases
        self.lambdas = lambdas

        self.W_shape = initial_W.shape
        self.num_nests = len(nests)

        tnl = TheanoNestedLogit()
        self.cost_function = tnl.cost_function
        # self.gradient_function = tnl.gradient_function

    def cost(self, parameters):
        cost, _, _ = self.cost_function(self.X, self.y,
                                        self.initial_W, self.initial_b, self.initial_l,
                                        self.utility_functions, self.biases, self.lambdas,
                                        parameters, self.alternatives,
                                        self.nest_indices, self.nests)
        return cost  # , error, predictions

    def results(self, parameters):
        cost, error, predictions = self.cost_function(self.X, self.y,
                                                      self.initial_W, self.initial_b, self.initial_l,
                                                      self.utility_functions, self.biases, self.lambdas,
                                                      parameters, self.alternatives,
                                                      self.nest_indices, self.nests)
        return cost, error, predictions

    def gradient(self, parameters):
        grad = self.gradient_function(self.X, self.y,
                                      self.initial_W, self.initial_b, self.initial_l,
                                      self.utility_functions, self.biases, self.lambdas,
                                      parameters, self.alternatives,
                                      self.nest_indices, self.nests)
        return grad

    def estimate(self):
        # input_params = self.ravel_parameters(self.initial_W, self.initial_b, self.initial_lambdas)
        # self.gradient_check(self.cost_f, self.gradient_f, input_params)
        self.parameters = optimize.fmin_bfgs(self.cost_f,
                                             self.parameters,
                                             fprime=self.gradient_f,
                                             gtol=0.0001, disp=False)
        # W, b, lambdas = self.unravel_parameters(parameters)
        # cost, error, predictions = self.results(W, b, lambdas)
        cost, error, predictions = self.results(self.parameters)
        # return cost, error, predictions, W, b, lambdas
        return cost, error, predictions

    @staticmethod
    def gradient_check(cost_function, gradient_function, parameters):
        grad_check = optimize.check_grad(cost_function, gradient_function, parameters)
        if abs(grad_check) > 1 * 10**-4:
            error = 'Gradient failed check with an error of ' + str(grad_check)
            raise ValueError(error)
