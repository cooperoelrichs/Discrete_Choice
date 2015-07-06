# Prototype logit estimation class
#
# This is a prototype class for estimating discrete choice logit models
# Python 3.4 64 bit with SciPy

from sklearn.preprocessing import LabelBinarizer
from scipy import optimize
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt


ModelResults = namedtuple('ModelResults', 'cost thetas lambdas iteration parameters')


class ModelEstimator(object):
    """A home made implementation of logistic regression"""

    def __init__(self, x, y, c):
        np.seterr(all='raise')
        self.x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        self.y = LabelBinarizer().fit_transform(y)
        self.y_index = y
        self.n = x.shape[1] + 1
        self.m = x.shape[0]
        self.k = self.y.shape[1]
        self.c = c
        self.cost = None
        self.iteration = 0
        self.sqrt_eps = np.sqrt(np.finfo(np.float64).eps)

    def estimate(self):
        parameters = self.initial_parameters
        self.gradient_check(self.cost_function, self.gradient_function, parameters)
        parameters = optimize.fmin_bfgs(self.cost_function, parameters,
                                        fprime=self.gradient_function,
                                        gtol=0.0001, disp=False)

        self.gradient_check(self.cost_function, self.gradient_function, parameters)
        return self.get_results(parameters)

    @staticmethod
    def gradient_check(cost_function, gradient_function,
                       theta):
        grad_check = optimize.check_grad(cost_function,
                                         gradient_function,
                                         theta)

        if abs(grad_check) > 1 * 10**-5:  # 1 * 10**-6:
            error = 'Gradient failed check with an error of ' + str(grad_check)
            raise ValueError(error)

    def cost_function(self, parameters):
        raise NotImplementedError("Don't instantiate the Base Class")

    def gradient_function(self, parameters):
        raise NotImplementedError("Don't instantiate the Base Class")

    def get_results(self, parameters):
        raise NotImplementedError("Don't instantiate the Base Class")


class NestedLogitEstimator(ModelEstimator):
    """
    Nested Logit!

    Good resources:
        1. http://www.civil.iitb.ac.in/~kvkrao/cE%20780%20
           Behavioural%20Travel%20Modelling/NL.pdf
        2. http://eml.berkeley.edu/books/train1201.pdf
        3. http://en.wikipedia.org/wiki/Maximum_likelihood
    """

    # TODO: Add support for availability conditions
    # TODO: Compare NL to Biogeme
    # TODO: Multi level NL
    # TODO: Regularised NL

    def __init__(self, x, y, c, alts, initial_parameters, fixed_parameters, utility_functions):
        super(NestedLogitEstimator, self).__init__(x, y, c)

        self.alts = np.array(alts)
        self.nest_index = np.zeros_like(self.y[0])  # = [0, 0, 1, 1]
        self.h = len(self.alts)
        self.nest_lens = [len(x) for x in self.alts]
        self.lambda_map = [4, 5]

        for i, x in enumerate(self.alts):
            for j in x:
                self.nest_index[j] = i

        self.initial_parameters = initial_parameters
        self.fixed_parameters = fixed_parameters
        self.utility_functions = utility_functions

        # plt.interactive(False)
        self.plot, = plt.plot([], [])
        plt.show(block=False)

    def cost_function(self, parameters):
        """
        Based on formula (4.2) from:
        K. Train, http://eml.berkeley.edu/books/train1201.pdf

        Indices
        m - number of experiments
        n - number of features
        k - number of classes
        h - number of nests
        j - class, subset some nest l, [1, ..., k]
        i - current experiment, [1, ..., m]
        l - nest, [1, ..., h]
        """

        nest_sums = np.zeros((self.m, self.h))
        v = np.zeros((self.m, self.k))
        for i in range(0, self.m):
            for l in range(0, self.h):
                for j in range(0, self.nest_lens[l]):
                    v_ilj = self.utility_functions[self.alts[l][j]](self.x[i], parameters)
                    v[i, self.alts[l][j]] = v_ilj
                    v_scaled = v_ilj / parameters[self.lambda_map[l]]
                    if abs(v_scaled) > 100:
                        print('%0.2f ~ %0.2f ~ %0.3f' % (v_scaled, v_ilj, parameters[self.lambda_map[l]]))
                    nest_sums[i, l] += np.exp(v_scaled)

        p = np.zeros(self.m)
        for i in range(0, self.m):
            j = self.y_index[i]
            l = self.nest_index[j]
            num = (np.exp(v[i, j] / parameters[self.lambda_map[l]]) *
                   (nest_sums[i, l] ** (parameters[self.lambda_map[l]] - 1.0)))
            dom = 0
            for l_2 in range(0, self.h):
                dom += nest_sums[i, l_2] ** parameters[self.lambda_map[l_2]]
            p[i] = num / dom

        cost = - np.sum(np.log(p)) / self.m
        return cost

    def gradient_function(self, parameters):
        """Serious numerical gradient stuff"""
        self.iteration += 1
        base_cost = self.cost_function(parameters)

        gradient = np.zeros_like(parameters)
        for p in range(0, len(parameters)):
            if p in self.fixed_parameters:
                gradient[p] = 0
            else:
                theta_p = parameters[p]
                step_size = self.sqrt_eps
                theta_p_step = theta_p + step_size
                d_theta_p = theta_p_step - theta_p  # This doesn't work...
                theta_f_step = np.copy(parameters)
                theta_f_step[p] = theta_p_step
                step_cost = self.cost_function(theta_f_step)
                gradient[p] = ((step_cost - base_cost) / d_theta_p)

                # if p == 0:
                #     print('%i ~ %0.4f ~ %0.4f ~ %0.4f ~ %0.4f ~ %0.4f'
                #           % (self.iteration, theta_p, theta_p_step, base_cost, step_cost, gradient[p]))

        print('%i ~ ' % self.iteration + '%0.4f, ' * len(parameters) % tuple(parameters))
        self.update_plot(self.iteration, base_cost)
        return gradient

    def get_results(self, parameters):
        return ModelResults(cost=self.cost_function(parameters),
                            thetas=parameters[:-1 * self.h],
                            lambdas=parameters[-1 * self.h:],
                            iteration=self.iteration,
                            parameters=parameters)

    def update_plot(self, x, y):
        self.plot.set_xdata(np.append(self.plot.get_xdata(), x))
        self.plot.set_ydata(np.append(self.plot.get_ydata(), y))
        self.plot.axes.relim()
        self.plot.axes.autoscale_view()
        plt.draw()
        # plt.show(block=False)

class MultinomialLogitEstimator(ModelEstimator):
    """
    Based on:
    http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
    """

    def __init__(self, x, y, c, initial_parameters, parameter_indices, fixed_parameters, variable_indices):
        super(MultinomialLogitEstimator, self).__init__(x, y, c)
        self.theta = np.random.randn(self.k, self.n)

        self.initial_parameters = initial_parameters
        self.parameter_indices = parameter_indices
        self.fixed_parameters = fixed_parameters
        self.variable_indices = variable_indices

    def cost_function(self, parameters):
        """
        m - number of data points
        n - number of features
        k - number of classes
        X - m * n
        y - m * k
        theta - k * n
        """

        # theta = np.reshape(parameters, (self.k, self.n))
        cost = 0
        for i in range(0, self.m):
            # j = self.y_index[i]
            for j in range(0, self.k):
                # numerator = np.exp(np.dot(self.x[i], theta[j]))
                numerator = np.exp(np.dot(self.x[i, self.variable_indices[j]], parameters[self.parameter_indices[j]]))
                denominator = 0
                for l in range(0, self.k):
                    # denominator += np.exp(np.dot(self.x[i], theta[l]))
                    denominator += np.exp(np.dot(self.x[i, self.variable_indices[l]],
                                                 parameters[self.parameter_indices[l]]))
                cost += self.y[i, j] * np.log(numerator / denominator)

        # regularisation = (0.5 / self.c * np.sum(theta ** 2))
        regularisation = (0.5 / self.c * np.sum(parameters ** 2))
        cost = (-1 * cost + regularisation) / self.m
        return cost

    def gradient_function(self, parameters):
        self.iteration += 1
        # theta = np.reshape(parameters, (self.k, self.n))
        # gradient = np.zeros_like(theta)
        gradient = np.zeros_like(parameters)
        for i in range(0, self.m):
            for j in range(0, self.k):
                # numerator = np.exp(np.dot(self.x[i], theta[j]))
                numerator = np.exp(np.dot(self.x[i, self.variable_indices[j]], parameters[self.parameter_indices[j]]))
                denominator = 0
                for l in range(0, self.k):
                    # denominator += np.exp(np.dot(self.x[i], theta[l]))
                    denominator += np.exp(np.dot(self.x[i, self.variable_indices[l]],
                                                 parameters[self.parameter_indices[l]]))

                # gradient[j] += self.x[i] * (self.y[i, j] - numerator / denominator)
                gradient[self.parameter_indices[j]] += (self.x[i, self.variable_indices[j]] *
                                                        (self.y[i, j] - numerator / denominator))

        # penalty_gradient = (1 / self.c) * theta
        penalty_gradient = (1 / self.c) * parameters
        gradient = (-1 * gradient + penalty_gradient) / self.m
        # return np.ravel(gradient)
        return np.ravel(gradient)

    def get_results(self, parameters):
        return ModelResults(cost=self.cost_function(parameters),
                            thetas=parameters,
                            lambdas=[],
                            iteration=self.iteration,
                            parameters=parameters)


class LogisticRegressionEstimator(ModelEstimator):
    def cost_function(self, theta):
        predicted_probs = self.predict_probs(theta)
        log_likelihood = ((-1 * self.y) * np.log(predicted_probs) -
                          (1 - self.y) * np.log(1 - predicted_probs))

        regularisation = (0.5 / self.c * np.sum(theta[1:] ** 2))

        cost = log_likelihood.mean() + regularisation / self.m
        self.cost = cost
        return cost

    def gradient_function(self, theta):
        predicted_probs = self.predict_probs(theta)
        error = predicted_probs - self.y
        objective_grad = np.dot(error, self.x)

        penalty_gradient = (1 / self.c) * theta
        penalty_gradient[0] = 0

        gradient = (objective_grad + penalty_gradient) / self.m
        return gradient

    @staticmethod
    def sigmoid(vector):
        return 1 / (1 + np.exp(- vector))

    @staticmethod
    def utility(theta, X):
        return np.dot(X, theta)

    def predict_probs(self, theta):
        return self.sigmoid(self.utility(theta, self.x))


class AltLogisticRegressionEstimator(ModelEstimator):
    """Testing some alternative math (based on liblinear)"""
    def cost_function(self, theta):
        y = np.copy(self.y)
        y[self.y == 0] = -1  # y vector must be (1, -1)

        objective_cost = 0
        for i in range(0, self.m):
            objective_cost += np.log(self.inverted_sigmoid(self.x[i], y[i]))

        regularisation = 0
        for j in range(1, self.n):
            regularisation += 0.5 * theta[j] ** 2

        cost = (regularisation / self.c + objective_cost) / self.m
        return cost

    def gradient_function(self, theta):
        y = np.copy(self.y)
        y[self.y == 0] = -1  # y vector must be (1, -1)

        objective_grad = np.zeros_like(theta)
        for i in range(0, self.m):
            grad_mat = (((1 / self.inverted_sigmoid(self.x[i], y[i])) - 1) *
                        y[i] * self.x[i])
            for j in range(0, self.n):
                objective_grad[j] += grad_mat[j]

        regularisation_grad = np.copy(theta)
        regularisation_grad[0] = 0

        gradient = (regularisation_grad / self.c + objective_grad) / self.m
        return gradient

    def inverted_sigmoid(self, x_i, y_i):
        sum = 0
        for j in range(0, self.n):
            sum += self.theta[j] * x_i[j]
        return 1 + np.exp(-1 * y_i * sum)
