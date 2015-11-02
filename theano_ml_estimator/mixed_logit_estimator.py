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
    def __init__(self, utility_functions):
        np.seterr(all='raise')
        theano.config.optimizer = 'None'  # 'fast_compile'  # More traceable errors
        theano.config.exception_verbosity = 'high'  # More traceable errors
        # theano.config.compute_test_value = 'raise'
        # theano.config.profile = True
        # theano.config.profile_memory = True

        self.float_type = 'floatX'
        int_type = 'int64'

        self.utility_functions = utility_functions

        self.X = T.matrix('X', dtype=self.float_type)
        self.y = T.vector('y', dtype=int_type)
        self.draws = T.tensor3('draws', dtype=self.float_type)

        self.parameters = T.vector('parameters', dtype=self.float_type)
        self.weights = T.vector('weights', dtype=self.float_type)

        self.cost, self.error, self.predictions = self.multinomial_logit_cost()
        self.cost_function = self.compile_cost_function()
        self.gradient_function = self.compile_gradient_function()

    def multinomial_logit_cost(self):
        # V = b + w_cost * cost + w_random_cost * cost_draw * cost + w_random_error * error_draw
        # V_rp_cost[14000, 6, 1000] = f(X[14000, costs], W_rp[costs, 6], draws[14000, random_cost_params, 1000])
        # V_rb_bias[14000, 6, 1000] = f(b_rp[6], draws[14000, random_error_params, 1000])

        V, _ = theano.scan(lambda fn, X_, parameters_, draws_: fn(X_, parameters_, draws_),
                           sequences=self.utility_functions,
                           outputs_info=T.tensor3('V', dtype=self.float_type),
                           non_sequences=[self.X, self.parameters, self.draws],
                           strict=True)

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
                                         self.draws,
                                         self.parameters,
                                         self.weights],
                                        [self.cost, self.error, self.predictions],
                                        no_default_updates=True,
                                        name='cost_function')  # , on_unused_input='ignore')  # , mode='DebugMode')
        return cost_function

    def compile_gradient_function(self):
        grad_function = theano.function([self.X, self.y,
                                         self.draws,
                                         self.parameters,
                                         self.weights],
                                        T.grad(self.cost, wrt=self.parameters),
                                        no_default_updates=True,
                                        name='gradient_function')  # , on_unused_input='ignore')
        return grad_function


class MixedLogitEstimator(object):
    def __init__(self, X, y, parameters, utility_functions, weights, num_alternatives):
        self.X = X
        self.y = y
        self.iter = 0
        self.last_time = time.clock()

        # Random draws
        self.num_draws = 1000
        self.num_observations = self.X.shape[0]
        self.num_alternatives = num_alternatives

        self.parameters = parameters
        self.utility_functions = utility_functions

        self.draws = self.generate_random_draws()
        self.weights = weights

        tml = TheanoMixedLogit(utility_functions)
        self.cost_function = tml.cost_function
        self.gradient_function = tml.gradient_function

    def generate_random_draws(self):
        draws = -np.log(-np.log(np.random.rand(self.num_observations, len(self.parameters), self.num_draws)))
        return draws

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
                                                      self.draws,
                                                      parameters,
                                                      self.weights)
        return cost, error, predictions

    def gradient(self, parameters):
        parameters = self.maybe_fix_dtype(parameters)
        grad = self.gradient_function(self.X, self.y,
                                      self.draws,
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

    def update_random_draws(self):
        elapsed_time = self.get_elapsed_time_and_reset_timer()
        print('Updating draws. Current iter: %i. Iter time: %.0fs' % (self.iter, elapsed_time))

        self.iter += 1
        self.cost_draws, self.bias_draws = self.generate_random_draws()

    def get_elapsed_time_and_reset_timer(self):
        current_time = time.clock()
        elapsed_time = current_time - self.last_time
        self.last_time = current_time
        return elapsed_time

    @staticmethod
    def gradient_check(cost_function, gradient_function, parameters):
        grad_check = optimize.check_grad(cost_function, gradient_function, parameters)
        print('Gradient error: %f' % grad_check)
        # if abs(grad_check) > 1 * 10**-4:
        #     error = 'Gradient failed check with an error of ' + str(grad_check)
        #     raise ValueError(error)
