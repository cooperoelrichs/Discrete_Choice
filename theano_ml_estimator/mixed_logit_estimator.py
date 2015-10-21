import theano
import theano.typed_list
import theano.tensor as T
import numpy as np
from scipy import optimize

# from sklearn.preprocessing import LabelBinarizer
# from collections import namedtuple
# import matplotlib.pyplot as plt
# import os


class TheanoMixedLogit(object):
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
        self.parameters = T.vector('parameters', dtype=float_type)

        self.utility_functions = T.matrix('utility_functions', dtype=int_type)
        self.biases = T.matrix('biases', dtype=int_type)
        self.weights = T.vector('weights', dtype=float_type)

        self.cost, self.error, self.predictions = self.multinomial_logit_cost()
        self.cost_function = self.compile_cost_function()
        self.gradient_function = self.compile_gradient_function()

    def multinomial_logit_cost(self):
        # I HATE UTILITY FUNCTIONS. ARG!!!
        W = T.set_subtensor(self.W_input[(self.utility_functions[:, 0], self.utility_functions[:, 1])],
                            self.parameters[self.utility_functions[:, 2]])
        b = T.set_subtensor(self.b_input[self.biases[:, 0]], self.parameters[self.biases[:, 1]])

        # V = b + w_cost * cost + w_random_cost * cost_draw * cost + w_random_error * error_draw
        V = T.dot(self.X, W) + b  # calculate utilities
        P = T.nnet.softmax(V)

        predictions = T.argmax(P, axis=1)
        error = T.mean(T.neq(predictions, self.y))
        cost = -T.mean(T.log(P)[T.arange(self.y.shape[0]), self.y] * self.weights)
        return cost, error, predictions

    def compile_cost_function(self):
        cost_function = theano.function([self.X, self.y,
                                         self.W_input, self.b_input,
                                         self.utility_functions, self.biases,
                                         self.parameters,
                                         self.weights],
                                        [self.cost, self.error, self.predictions],
                                        name='cost_function')  # , mode='DebugMode')
        return cost_function

    def compile_gradient_function(self):
        grad = theano.function([self.X, self.y,
                                self.W_input, self.b_input,
                                self.utility_functions, self.biases,
                                self.parameters,
                                self.weights],
                               T.grad(self.cost, wrt=self.parameters),
                               name='cost_function')
        return grad


class MixedLogitEstimator(object):
    def __init__(self, X, y, W_input, b_input, W_rand_input, error_input,
                 parameters, utility_functions, biases, random_cost_params, random_error_params, weights):
        self.X = X
        self.y = y

        self.W_input = W_input
        self.b_input = b_input
        self.W_rand_input = W_rand_input
        self.error_input = error_input

        self.parameters = parameters
        self.utility_functions = utility_functions
        self.biases = biases
        self.random_cost_params = random_cost_params
        self.random_error_params = random_error_params

        self.weights = weights

        self.W_shape = W_input.shape

        tml = TheanoMixedLogit()
        self.cost_function = tml.cost_function
        self.gradient_function = tml.gradient_function

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
                                                      self.W_input, self.b_input,
                                                      self.utility_functions, self.biases,
                                                      parameters,
                                                      self.weights)
        return cost, error, predictions

    def gradient(self, parameters):
        parameters = self.maybe_fix_dtype(parameters)
        grad = self.gradient_function(self.X, self.y,
                                      self.W_input, self.b_input,
                                      self.utility_functions, self.biases,
                                      parameters,
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

        W[(self.utility_functions[:, 0], self.utility_functions[:, 1])] = self.parameters[self.utility_functions[:, 2]]
        b[self.biases[:, 0]] = parameters[self.biases[:, 1]]
        return W, b

    @staticmethod
    def gradient_check(cost_function, gradient_function, parameters):
        grad_check = optimize.check_grad(cost_function, gradient_function, parameters)
        if abs(grad_check) > 1 * 10**-4:
            error = 'Gradient failed check with an error of ' + str(grad_check)
            raise ValueError(error)
