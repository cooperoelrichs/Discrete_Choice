import numpy as np
from scipy import optimize
import theano
import theano.typed_list
import theano.tensor as T
# from theano.tensor.shared_randomstreams import RandomStreams
import time
import matplotlib.pyplot as plt


# from sklearn.preprocessing import LabelBinarizer
# from collections import namedtuple
# import matplotlib.pyplot as plt
# import os


class TheanoMixedLogit(object):
    def __init__(self, utility_functions, float_dtype, int_dtype):
        np.seterr(all='raise')
        # theano.config.optimizer = 'None'  # 'fast_compile'  # More traceable errors
        # theano.config.exception_verbosity = 'high'  # More traceable errors
        # theano.config.compute_test_value = 'raise'
        # theano.config.profile = True
        # theano.config.profile_memory = True

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

        self.utility_functions = utility_functions

        self.V = T.tensor3('V', dtype=self.float_dtype)
        self.X = T.matrix('X', dtype=self.float_dtype)
        self.y = T.vector('y', dtype=self.int_dtype)
        self.draws = T.tensor3('draws', dtype=self.float_dtype)

        self.parameters = T.vector('parameters', dtype=self.float_dtype)
        self.weights = T.vector('weights', dtype=self.float_dtype)

        self.cost, self.error, self.predictions = self.multinomial_logit_cost()
        self.cost_function = self.compile_cost_function()
        self.gradient_function = self.compile_gradient_function()

    def multinomial_logit_cost(self):
        V = self.utility_functions.calculate_V(self.V, self.X, self.parameters, self.draws)

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
                                         self.V,
                                         self.draws,
                                         self.parameters,
                                         self.weights],
                                        [self.cost, self.error, self.predictions],
                                        no_default_updates=True,
                                        name='cost_function')  # , on_unused_input='ignore')  # , mode='DebugMode')
        return cost_function

    def compile_gradient_function(self):
        grad_function = theano.function([self.X, self.y,
                                         self.V,
                                         self.draws,
                                         self.parameters,
                                         self.weights],
                                        T.grad(self.cost, wrt=self.parameters),
                                        no_default_updates=True,
                                        name='gradient_function')  # , on_unused_input='ignore')
        return grad_function


class MixedLogitEstimator(object):
    def __init__(self, X, y, parameters, utility_functions, weights, num_alternatives, num_draws, float_dtype, int_dtype):
        self.X = X
        self.y = y
        self.iter = 0
        self.results_by_iteration = {}
        self.last_time = time.clock()

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

        # Random draws
        self.num_draws = num_draws
        self.num_observations = self.X.shape[0]
        self.num_alternatives = num_alternatives

        self.V = np.zeros((X.shape[0], num_alternatives, self.num_draws), dtype=self.float_dtype)
        self.parameters = parameters
        self.utility_functions = utility_functions

        self.draws = self.generate_random_draws()
        self.weights = weights

        tml = TheanoMixedLogit(utility_functions, self.float_dtype, self.int_dtype)
        self.cost_function = tml.cost_function
        self.gradient_function = tml.gradient_function

    def generate_random_draws(self):
        np.random.random_sample()
        draws = -np.log(-np.log(
            np.random.random_sample(
                (self.num_observations, len(self.parameters), self.num_draws)
            )
        )).astype(self.float_dtype)
        return draws

    def cost(self, parameters):
        cost, _, _ = self.results(parameters)
        return cost

    def maybe_fix_dtype(self, parameters):
        if self.float_dtype == 'float32':
            # optimize.fmin_bfgs changes the dtype of parameters to float64
            parameters = parameters.astype('float32')
        return parameters

    def results(self, parameters):
        parameters = self.maybe_fix_dtype(parameters)
        cost, error, predictions = self.cost_function(self.X, self.y,
                                                      self.V,
                                                      self.draws,
                                                      parameters,
                                                      self.weights)
        return cost, error, predictions

    def gradient(self, parameters):
        parameters = self.maybe_fix_dtype(parameters)
        grad = self.gradient_function(self.X, self.y,
                                      self.V,
                                      self.draws,
                                      parameters,
                                      self.weights)
        return grad

    def estimate(self):
        self.iter = 0
        self.gradient_check(self.cost, self.gradient, self.parameters)

        cost, error, _ = self.results(self.parameters)
        self.results_by_iteration[self.iter] = (cost, 1 - error)

        print('Starting estimation. Elapsed time: %.0fs' % self.get_elapsed_time_and_reset_timer())
        result = optimize.minimize(
            fun=self.cost,
            x0=self.parameters,
            method='BFGS',
            # method='CG',
            jac=self.gradient,
            tol=0.00001,
            callback=self.update_random_draws,
            # retall=True,
            options={
                'disp': True,
                'maxiter': 100,
            }
        )

        self.parameters = result.x

        self.plot_cost_by_iteration()
        self.gradient_check(self.cost, self.gradient, self.parameters)
        cost, error, predictions = self.results(self.parameters)
        return cost, error, predictions, self.parameters

    def update_random_draws(self, current_parameters):
        elapsed_time = self.get_elapsed_time_and_reset_timer()

        cost, error, _ = self.results(current_parameters)
        self.results_by_iteration[self.iter] = (cost, 1 - error)

        print('Updating draws. Current iter: %i. Iter time: %.0fs' % (self.iter, elapsed_time))
        self.draws = self.generate_random_draws()
        self.iter += 1

    def plot_cost_by_iteration(self):
        fig, ax1 = plt.subplots()
        title = 'cost by iter - %i draws' % self.num_draws
        ax1.set_title(title)

        iters = list(self.results_by_iteration.keys())
        costs = [cost for cost, _ in self.results_by_iteration.values()]
        accuracies = [accuracy for _, accuracy in self.results_by_iteration.values()]

        ax1.plot(iters, costs, color='b')
        ax1.set_ylabel('mean cost', color='b')
        ax1.set_xlabel('iter')

        ax2 = ax1.twinx()
        ax2.plot(iters, accuracies, color='r')
        ax2.set_ylabel('accuracy', color='r')

        dir = 'theano_ml_estimator/results/'
        plt.savefig(dir + title, bbox_inches='tight')
        plt.close()

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
