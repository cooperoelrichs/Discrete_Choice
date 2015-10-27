import numpy as np
from scipy import optimize
import theano
import theano.typed_list
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import time

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
        self.cost_draws = T.tensor3('cost_draws', dtype=float_type)
        self.bias_draws = T.tensor3('bias_draws', dtype=float_type)
        self.W_input = T.matrix('W_input', dtype=float_type)
        self.b_input = T.vector('b_input', dtype=float_type)
        self.W_rp_input = T.matrix('W_rp_input', dtype=float_type)
        self.b_rp_input = T.vector('b_rp_input', dtype=float_type)
        
        self.parameters = T.vector('parameters', dtype=float_type)

        self.costs = T.matrix('costs', dtype=int_type)
        self.biases = T.matrix('biases', dtype=int_type)
        self.srng = RandomStreams(seed=2341)
        self.rp_costs = T.matrix('rp_costs', dtype=int_type)
        self.rp_biases = T.matrix('rp_biases', dtype=int_type)
        
        self.weights = T.vector('weights', dtype=float_type)

        self.cost, self.error, self.predictions = self.multinomial_logit_cost()
        self.cost_function = self.compile_cost_function()
        self.gradient_function = self.compile_gradient_function()

    def multinomial_logit_cost(self):
        # I HATE UTILITY FUNCTIONS
        W = T.set_subtensor(self.W_input[(self.costs[:, 0], self.costs[:, 1])],
                            self.parameters[self.costs[:, 2]])
        b = T.set_subtensor(self.b_input[self.biases[:, 0]], self.parameters[self.biases[:, 1]])

        W_rp = T.set_subtensor(self.W_rp_input[(self.rp_costs[:, 0], self.rp_costs[:, 1])],
                               self.parameters[self.rp_costs[:, 2]])
        b_rp = T.set_subtensor(self.b_rp_input[self.rp_biases[:, 0]], self.parameters[self.rp_biases[:, 1]])

        # V = b + w_cost * cost + w_random_cost * cost_draw * cost + w_random_error * error_draw
        V = T.dot(self.X, W) + b
        V_rp_cost = T.dot(self.X, W_rp)[:, :, np.newaxis] * self.cost_draws
        V_rp_bias = b_rp[np.newaxis, :, np.newaxis] * self.bias_draws
        V_rp = (V_rp_cost + V_rp_bias)

        V = V[:, :, np.newaxis] + V_rp
        # P = T.nnet.softmax(V)
        e_V = T.exp(V - V.max(axis=1, keepdims=True))
        P = e_V / e_V.sum(axis=1, keepdims=True)
        P = P.mean(axis=2)

        predictions = T.argmax(P, axis=1)
        error = T.mean(T.neq(predictions, self.y))
        cost = -T.mean(T.log(P)[T.arange(self.y.shape[0]), self.y] * self.weights)
        return cost, error, predictions

    def compile_cost_function(self):
        cost_function = theano.function([self.X, self.y,
                                         self.cost_draws, self.bias_draws,
                                         self.W_input, self.b_input,
                                         self.W_rp_input, self.b_rp_input,
                                         self.costs, self.biases,
                                         self.rp_costs, self.rp_biases,
                                         self.parameters,
                                         self.weights],
                                        [self.cost, self.error, self.predictions],
                                        no_default_updates=True,
                                        name='cost_function')  # , on_unused_input='ignore')  # , mode='DebugMode')
        return cost_function

    def compile_gradient_function(self):
        grad_function = theano.function([self.X, self.y,
                                         self.cost_draws, self.bias_draws,
                                         self.W_input, self.b_input,
                                         self.W_rp_input, self.b_rp_input,
                                         self.costs, self.biases,
                                         self.rp_costs, self.rp_biases,
                                         self.parameters,
                                         self.weights],
                                        T.grad(self.cost, wrt=self.parameters),
                                        no_default_updates=True,
                                        name='gradient_function')  # , on_unused_input='ignore')
        return grad_function


class MixedLogitEstimator(object):
    def __init__(self, X, y, W_input, b_input, W_rb_input, b_rp_input,
                 parameters, costs_params, biases, rp_costs, rp_biases, weights, num_alternatives):
        self.X = X
        self.y = y
        self.iter = 0
        self.last_time = time.clock()

        # Random draws
        self.num_draws = 1000
        self.num_observations = self.X.shape[0]
        self.num_alternatives = num_alternatives
        self.cost_draws, self.bias_draws = self.generate_random_draws()

        # Input params
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

    def generate_random_draws(self):
        cost_draws = -np.log(-np.log(np.random.rand(self.num_observations, self.num_alternatives, self.num_draws)))
        bias_draws = -np.log(-np.log(np.random.rand(self.num_observations, self.num_alternatives, self.num_draws)))
        return cost_draws, bias_draws

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
                                                      self.cost_draws, self.bias_draws,
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
                                      self.cost_draws, self.bias_draws,
                                      self.W_input, self.b_input,
                                      self.W_rp_input, self.b_rp_input,
                                      self.cost_params, self.biases,
                                      self.rp_costs, self.rp_biases,
                                      parameters,
                                      self.weights)
        return grad

    def estimate(self):
        self.iter = 1
        self.gradient_check(self.cost, self.gradient, self.parameters)

        print('Starting estimation. Elapsed time: %.0fs' % self.get_elapsed_time_and_reset_timer())
        self.parameters = optimize.fmin_bfgs(self.cost,
                                             self.parameters,
                                             fprime=self.gradient,
                                             gtol=0.00001,
                                             maxiter=100,
                                             callback=self.update_random_draws,
                                             disp=True)

        self.gradient_check(self.cost, self.gradient, self.parameters)
        cost, error, predictions = self.results(self.parameters)
        return cost, error, predictions, self.parameters

    def update_random_draws(self, params):
        elapsed_time = self.get_elapsed_time_and_reset_timer()
        print('Updating draws. Current iter: %i. Iter time: %.0fs' % (self.iter, elapsed_time))

        self.iter += 1
        self.cost_draws, self.bias_draws = self.generate_random_draws()

    def get_elapsed_time_and_reset_timer(self):
        current_time = time.clock()
        elapsed_time = current_time - self.last_time
        self.last_time = current_time
        return elapsed_time

    def extract_parameters(self, parameters):
        W = self.W_input
        b = self.b_input

        W[(self.cost_params[:, 0], self.cost_params[:, 1])] = self.parameters[self.cost_params[:, 2]]
        b[self.biases[:, 0]] = parameters[self.biases[:, 1]]
        return W, b

    @staticmethod
    def gradient_check(cost_function, gradient_function, parameters):
        grad_check = optimize.check_grad(cost_function, gradient_function, parameters)
        print('Gradient error: %f' % grad_check)
        # if abs(grad_check) > 1 * 10**-4:
        #     error = 'Gradient failed check with an error of ' + str(grad_check)
        #     raise ValueError(error)
