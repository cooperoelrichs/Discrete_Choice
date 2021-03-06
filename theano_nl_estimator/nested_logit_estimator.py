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
        # theano.config.optimizer = 'None'  # 'fast_compile'  # More traceable errors
        # theano.config.exception_verbosity = 'high'  # More traceable errors
        # theano.config.compute_test_value = 'raise'

        float_type = 'floatX'
        int_type = 'int64'

        self.X = T.matrix('X', dtype=float_type)
        self.y = T.vector('y', dtype=int_type)
        self.W_input = T.matrix('W_input', dtype=float_type)
        self.b_input = T.vector('b_input', dtype=float_type)
        self.l_input = T.vector('l_input', dtype=float_type)
        self.parameters = T.vector('parameters', dtype=float_type)

        self.utility_functions = T.matrix('utility_functions', dtype=int_type)
        self.biases = T.matrix('biases', dtype=int_type)
        self.lambdas = T.matrix('lambdas', dtype=int_type)

        self.nests = T.vector('nests', dtype=int_type)
        self.nest_indices = T.vector('nest_indices', dtype=int_type)
        # self.alt_indices = theano.typed_list.TypedListType(T.lvector)()
        self.alternatives = T.vector('alternatives', dtype=int_type)

        self.weights = T.vector('weights', dtype=float_type)

        self.cost, self.error, self.predictions = self.nested_logit_cost()

        self.cost_function = self.compile_cost_function()
        self.gradient_function = self.compile_gradient_function()

    def calculate_nest_sums(self, exp_V, nests, nest_indices):
        nest_sums_T, _ = theano.scan(
            self.sum_nest_for_nest,
            sequences=[nests],
            non_sequences=[nest_indices, exp_V],
            name='sum_exp_utilities_by_nests'
        )
        return nest_sums_T.T

    @staticmethod
    def sum_nest_for_nest(i, nest_indices, exp_V):
        return exp_V[:, T.eq(nest_indices, i).nonzero()[0]].sum(axis=1)

    def calculate_probabilities(self, exp_V, lambdas, alternatives, nest_indices):
        nest_sums = self.calculate_nest_sums(exp_V, self.nests, self.nest_indices)
        denominator = np.power(nest_sums, lambdas).sum(axis=1)

        P_T, _ = theano.scan(
            self.calculate_probability_for_alternative,
            sequences=[alternatives],
            non_sequences=[lambdas, nest_indices, exp_V, nest_sums, denominator],
            name='calculate_probabilities_by_alternatives'
        )
        return P_T.T

    @staticmethod
    def calculate_probability_for_alternative(alt, lambdas, nest_indices, exp_V, nest_sums, denominator):
        numerator = exp_V[:, alt] * np.power(nest_sums[:, nest_indices[alt]], lambdas[nest_indices][alt] - 1)
        return numerator / denominator

    def nested_logit_cost(self):
        W = T.set_subtensor(self.W_input[(self.utility_functions[:, 0], self.utility_functions[:, 1])],
                            self.parameters[self.utility_functions[:, 2]])
        b = T.set_subtensor(self.b_input[self.biases[:, 0]], self.parameters[self.biases[:, 1]])
        l = T.set_subtensor(self.l_input[self.lambdas[:, 0]], self.parameters[self.lambdas[:, 1]])

        V = T.dot(self.X, W) + b  # calculate utilities
        V = V - V.mean(axis=1, keepdims=True)  # numerical stability
        # V = T.clip(V, -80, 80)  # numerical stability
        exp_V = T.exp(V / l[self.nest_indices])  # exp of the scaled utilities
        P = self.calculate_probabilities(exp_V, l, self.alternatives, self.nest_indices)

        predictions = T.argmax(P, axis=1)
        error = T.mean(T.neq(predictions, self.y))
        cost = -T.mean(T.log(P)[T.arange(self.y.shape[0]), self.y] * self.weights)
        return cost, error, predictions

    def compile_cost_function(self):
        cost_function = theano.function([self.X, self.y,
                                         self.W_input, self.b_input, self.l_input,
                                         self.utility_functions, self.biases, self.lambdas,
                                         self.parameters,
                                         self.alternatives,
                                         self.nest_indices, self.nests,
                                         self.weights],
                                        [self.cost, self.error, self.predictions],
                                        name='cost_function')  # , mode='DebugMode')
        return cost_function

    def compile_gradient_function(self):
        grad = theano.function([self.X, self.y,
                                self.W_input, self.b_input, self.l_input,
                                self.utility_functions, self.biases, self.lambdas,
                                self.parameters,
                                self.alternatives,
                                self.nest_indices, self.nests,
                                self.weights],
                               T.grad(self.cost, wrt=self.parameters),
                               name='cost_function')
        return grad


class NestedLogitEstimator(object):
    def __init__(self, X, y, W_input, b_input, l_input, nests, nest_indices, alternatives,
                 parameters, utility_functions, biases, lambdas, weights):
        self.X = X
        self.y = y

        self.W_input = W_input
        self.b_input = b_input

        self.l_input = l_input
        self.nests = nests
        self.nest_indices = nest_indices
        # self.alt_indices = alt_indices
        self.alternatives = alternatives
        self.parameters = parameters
        self.utility_functions = utility_functions
        self.biases = biases
        self.lambdas = lambdas

        self.weights = weights

        self.W_shape = W_input.shape
        self.num_nests = len(nests)

        tnl = TheanoNestedLogit()
        self.cost_function = tnl.cost_function
        self.gradient_function = tnl.gradient_function

    def cost(self, parameters):
        cost, _, _ = self.results(parameters)
        return cost

    def maybe_fix_dtype(self, parameters):
        if theano.config.floatX == 'float32':
            # optimize.fmin_bfgs changes the dtype of parameters to float64
            parameters = parameters.astype('float32')
        return parameters

    def results(self, parameters):
        parameters = self.maybe_fix_dtype(parameters)
        cost, error, predictions = self.cost_function(self.X, self.y,
                                                      self.W_input, self.b_input, self.l_input,
                                                      self.utility_functions, self.biases, self.lambdas,
                                                      parameters,
                                                      self.alternatives,
                                                      self.nest_indices, self.nests,
                                                      self.weights)
        return cost, error, predictions

    def gradient(self, parameters):
        parameters = self.maybe_fix_dtype(parameters)
        grad = self.gradient_function(self.X, self.y,
                                      self.W_input, self.b_input, self.l_input,
                                      self.utility_functions, self.biases, self.lambdas,
                                      parameters,
                                      self.alternatives,
                                      self.nest_indices, self.nests,
                                      self.weights)
        return grad

    def estimate(self):
        # self.gradient_check(self.cost, self.gradient, self.parameters)
        self.parameters = optimize.fmin_bfgs(self.cost,
                                             self.parameters,
                                             fprime=self.gradient,
                                             gtol=0.0000001, disp=True)

        # self.gradient_check(self.cost, self.gradient, self.parameters)
        cost, error, predictions = self.results(self.parameters)
        return cost, error, predictions, self.parameters

    def extract_parameters(self, parameters):
        W = self.W_input
        b = self.b_input
        l = self.l_input

        W[(self.utility_functions[:, 0], self.utility_functions[:, 1])] = self.parameters[self.utility_functions[:, 2]]
        b[self.biases[:, 0]] = parameters[self.biases[:, 1]]
        l[self.lambdas[:, 0]] = parameters[self.lambdas[:, 1]]
        return W, b, l

    @staticmethod
    def gradient_check(cost_function, gradient_function, parameters):
        grad_check = optimize.check_grad(cost_function, gradient_function, parameters)
        if abs(grad_check) > 1 * 10**-4:
            error = 'Gradient failed check with an error of ' + str(grad_check)
            raise ValueError(error)
