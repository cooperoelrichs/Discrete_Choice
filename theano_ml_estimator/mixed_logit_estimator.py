import numpy as np
from scipy import optimize
import theano
import theano.typed_list
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

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
        # theano.config.profile = True
        # theano.config.profile_memory = True

        float_type = 'floatX'
        int_type = 'int64'

        self.X = T.matrix('X', dtype=float_type)
        self.y = T.vector('y', dtype=int_type)
        self.W_input = T.matrix('W_input', dtype=float_type)
        self.b_input = T.vector('b_input', dtype=float_type)
        self.W_rp_input = T.matrix('W_rp_input', dtype=float_type)
        self.b_rp_input = T.vector('b_rp_input', dtype=float_type)
        
        self.parameters = T.vector('parameters', dtype=float_type)

        self.costs = T.matrix('costs', dtype=int_type)
        self.biases = T.matrix('biases', dtype=int_type)
        self.rp_costs = T.matrix('rp_costs', dtype=int_type)
        self.rp_biases = T.matrix('rp_biases', dtype=int_type)
        
        self.weights = T.vector('weights', dtype=float_type)

        self.cost, self.error, self.predictions = self.multinomial_logit_cost()
        self.cost_function = self.compile_cost_function()
        self.gradient_function = self.compile_gradient_function()

    def multinomial_logit_cost(self):
        srng = RandomStreams(seed=2341)

        # I HATE UTILITY FUNCTIONS
        W = T.set_subtensor(self.W_input[(self.costs[:, 0], self.costs[:, 1])],
                            self.parameters[self.costs[:, 2]])
        b = T.set_subtensor(self.b_input[self.biases[:, 0]], self.parameters[self.biases[:, 1]])

        W_rp = T.set_subtensor(self.W_rp_input[(self.rp_costs[:, 0], self.rp_costs[:, 1])],
                               self.parameters[self.rp_costs[:, 2]])
        b_rp = T.set_subtensor(self.b_rp_input[self.rp_biases[:, 0]], self.parameters[self.rp_biases[:, 1]])

        # Random draws
        num_draws = 1000
        num_alternatives = T.max(self.y)
        num_observations = self.X.shape[0]
        dims = (num_observations, num_alternatives, num_draws)
        cost_draws = -T.log(-T.log(srng.uniform(dims)))
        bias_draws = -T.log(-T.log(srng.uniform(dims)))

        # V = b + w_cost * cost + w_random_cost * cost_draw * cost + w_random_error * error_draw
        V = T.dot(self.X, W) + b
        V_rp = (T.dot(self.X, W_rp)[:, :, np.newaxis] * cost_draws +
                (b_rp[:, np.newaxis, np.newaxis] * bias_draws))

        V = V[:, :, np.newaxis] + V_rp
        # P = T.nnet.softmax(V)
        e_V = T.exp(V - V.max(axis=1, keepdims=True))
        P = e_V / e_V.sum(axis=1, keepdims=True)

        predictions = T.argmax(P, axis=1)
        error = T.mean(T.neq(predictions, self.y[:, np.newaxis]))
        cost = -T.mean(T.log(P)[T.arange(self.y.shape[0]), self.y] * self.weights[:, np.newaxis])
        return cost, error, predictions

    def compile_cost_function(self):
        cost_function = theano.function([self.X, self.y,
                                         self.W_input, self.b_input,
                                         self.W_rp_input, self.b_rp_input,
                                         self.costs, self.biases,
                                         self.rp_costs, self.rp_biases,
                                         self.parameters,
                                         self.weights],
                                        [self.cost, self.error, self.predictions],
                                        name='cost_function')  #, on_unused_input='ignore')  # , mode='DebugMode')
        return cost_function

    def compile_gradient_function(self):
        grad_function = theano.function([self.X, self.y,
                                         self.W_input, self.b_input,
                                         self.W_rp_input, self.b_rp_input,
                                         self.costs, self.biases,
                                         self.rp_costs, self.rp_biases,
                                         self.parameters,
                                         self.weights],
                                        T.grad(self.cost, wrt=self.parameters),
                                        name='gradient_function', on_unused_input='ignore')
        return grad_function


class MixedLogitEstimator(object):
    def __init__(self, X, y, W_input, b_input, W_rb_input, b_rp_input,
                 parameters, costs_params, biases, rp_costs, rp_biases, weights):
        self.X = X
        self.y = y

        self.W_input = W_input
        self.b_input = b_input
        self.W_rp_input = W_rb_input
        self.b_rp_input = b_rp_input

        self.parameters = parameters
        self.cost_params = costs_params
        self.biases = biases
        self.rp_costs = rp_costs
        self.rp_biases = rp_biases

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
                                                      self.W_rp_input, self.b_rp_input,
                                                      self.cost_params, self.biases,
                                                      self.rp_costs, self.rp_biases,
                                                      parameters,
                                                      self.weights)
        return cost, error, predictions

    def gradient(self, parameters):
        parameters = self.maybe_fix_dtype(parameters)
        grad = self.gradient_function(self.X, self.y,
                                      self.W_input, self.b_input,
                                      self.W_rp_input, self.b_rp_input,
                                      self.cost_params, self.biases,
                                      self.rp_costs, self.rp_biases,
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

        W[(self.cost_params[:, 0], self.cost_params[:, 1])] = self.parameters[self.cost_params[:, 2]]
        b[self.biases[:, 0]] = parameters[self.biases[:, 1]]
        return W, b

    @staticmethod
    def gradient_check(cost_function, gradient_function, parameters):
        grad_check = optimize.check_grad(cost_function, gradient_function, parameters)
        if abs(grad_check) > 1 * 10**-4:
            error = 'Gradient failed check with an error of ' + str(grad_check)
            raise ValueError(error)
