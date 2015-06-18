# Prototype logit estimation class
#
# This is a prototype class for estimating discrete choice logit models
# Python 3.4 64 bit with SciPy

from sklearn.preprocessing import LabelBinarizer
from scipy import optimize
from collections import namedtuple
import numpy as np


ModelResults = namedtuple('ModelResults', 'cost thetas lambdas iteration')


class ModelEstimator(object):
    """A home made implementation of logistic regression"""

    def __init__(self, x, y, c, alts):
        np.seterr(all='raise')
        self.x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        # self.x[0, 0] = 0
        self.y = LabelBinarizer().fit_transform(y)
        self.y_index = y
        self.n = x.shape[1] + 1
        self.m = x.shape[0]
        self.k = self.y.shape[1]
        # self.theta = np.random.randn(self.k, self.n)
        # self.theta_f = np.ravel(self.theta)
        self.c = c
        self.cost = None
        self.iteration = 0
        machine_epsilon = np.finfo(np.float64).eps
        self.sqrt_eps = np.sqrt(machine_epsilon)

        # Migrated from the NL prep_work function
        self.alts = np.array(alts)
        self.nest_index = np.zeros_like(self.y[0])  # = [0, 0, 1, 1]
        self.h = len(self.alts)
        self.nest_lens = [len(x) for x in self.alts]
        self.lambda_map = [5, 6]

        for i, x in enumerate(self.alts):
            for j in x:
                self.nest_index[j] = i

        # Experimental utility function specification
        self.fixed_parameters = set([6])  # Set of parameter numbers

        def u1(x_i, params):
            return np.dot(x_i[[0, 1, 2]], params[[0, 3, 4]])

        def u2(x_i, params):
            return np.dot(x_i[[0, 3, 4]], params[[1, 3, 4]])

        def u3(x_i, params):
            return np.dot(x_i[[0, 5, 6]], params[[2, 3, 4]])

        self.utility_functions = [  # Alternative number to utility function
            u1,
            u2,
            u3,
        ]

    def estimate(self):
        parameters = self.get_parameters()
        self.gradient_check(self.cost_function, self.gradient_function, parameters)
        parameters = optimize.fmin_bfgs(self.cost_function, parameters,
                                        fprime=self.gradient_function,
                                        gtol=0.001, disp=False)

        self.gradient_check(self.cost_function, self.gradient_function, parameters)
        results = ModelResults(cost=self.cost_function(parameters),
                               thetas=parameters[:-1 * self.h],
                               lambdas=parameters[-1 * self.h:],
                               iteration=self.iteration)
        return results

    @staticmethod
    def gradient_check(cost_function, gradient_function,
                       theta):
        grad_check = optimize.check_grad(cost_function,
                                         gradient_function,
                                         theta)

        if abs(grad_check) > 6 * 10**-6:
            error = 'Gradient failed check with an error of ' + str(grad_check)
            raise ValueError(error)

    @staticmethod
    def get_parameters():
        raise NotImplementedError("Don't instantiate the Base Class")

    def cost_function(self, parameters):
        raise NotImplementedError("Don't instantiate the Base Class")

    def gradient_function(self, parameters):
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

        Storage containers
        v - 2D matrix of utilities, m * k
        theta - vector of utility parameters, m
        lambdas - vector of nest parameters, h
        nest_lens - vector of nest lengths, h
        nest_sums - 2D matrix of e^(v/lambda), m * h
        alts - vector of vectors, maps nest and nested alternative to
               overall alternative, h * (classes in nest)
        """

        # TODO: Get the MNL estimator running, for comparison purposes!

        nest_sums = np.zeros((self.m, self.h))
        v = np.zeros((self.m, self.k))
        for i in range(0, self.m):
            for l in range(0, self.h):
                for j in range(0, self.nest_lens[l]):
                    v_ilj = self.utility_functions[self.alts[l][j]](self.x[i], parameters)
                    v[i, self.alts[l][j]] = v_ilj
                    v_scaled = v_ilj / parameters[self.lambda_map[l]]
                    if v_scaled > 200.0 or v_scaled < -200.0:
                        # We are getting very small lambda values sometimes.
                        # np.exp(x), where x > 709, causes an overflow
                        print('%0.6f - %s - %0.6f' %
                              (v_ilj, str(parameters[-2:]), v_scaled))
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
        # self.lambdas = theta_f[-1 * self.h:]
        # self.theta = np.reshape(theta_f[:-1 * self.h], (self.k, self.n))

        base_cost = self.cost_function(parameters)

        gradient = np.zeros_like(parameters)
        for p in range(0, len(parameters)):
            if p in self.fixed_parameters:
                gradient[p] = 0.0
            else:
                theta_p = parameters[p]
                step_size = self.sqrt_eps  # * 2.0
                theta_p_step = theta_p + step_size
                d_theta_p = theta_p_step - theta_p  # This doesn't work...
                theta_f_step = np.copy(parameters)
                theta_f_step[p] = theta_p_step
                step_cost = self.cost_function(theta_f_step)
                gradient[p] = ((step_cost - base_cost) / d_theta_p)

                if p == 0:
                    print((str(self.iteration) + ' - ' +
                           str(theta_p) + ' - ' + str(theta_p_step) + ' - ' +
                           str(base_cost) + ' - ' +
                           str(step_cost) + ' - ' +
                           str(gradient[p])))

        return gradient

    @staticmethod
    def get_parameters():
        # ASC_CAR = Beta('ASC_CAR',-0.167,-10,10,0)
        # ASC_TRAIN = Beta('ASC_TRAIN',-0.512,-10,10,0)
        # ASC_SM = Beta('ASC_SM',0,-10,10,1)
        # B_TIME = Beta('B_TIME',-0.899,-10,10,0)
        # B_COST = Beta('B_COST',-0.857,-10,10,0)
        # MU = Beta('MU',2.05,1,10,0)

        return np.array([  # Parameter number to initial parameter
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
            1.0,
            1.0,
        ])


class MultinomialLogitEstimator(ModelEstimator):
    """
    Based on:
    http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
    """
    def cost_function(self, theta_f):
        """
        m - number of data points
        n - number of features
        k - number of classes
        X - m * n
        y - m * k
        theta - k * n
        """

        theta = np.reshape(theta_f, (3, 3))  # (self.k, self.n))
        cost = 0
        for i in range(0, self.m):
            for j in range(0, self.k):
                # numerator = np.exp(np.dot(self.x[i], theta[j]))
                numerator = np.exp(self.utility_functions[j](self.x[i], theta_f))
                denominator = 0
                for l in range(0, self.k):
                    # denominator += np.exp(np.dot(self.x[i], theta[l]))
                    denominator += np.exp(self.utility_functions[l](self.x[i], theta_f))
                cost += self.y[i, j] * np.log(numerator / denominator)

        # regularisation = (0.5 / self.c * np.sum(theta[:, 1:] ** 2))
        regularisation = (0.5 / self.c * np.sum(theta ** 2))
        cost = (-1 * cost + regularisation) / self.m
        return cost

    def gradient_function(self, theta_f):
        theta = np.reshape(theta_f, (self.k, self.n))
        gradient = np.zeros_like(theta)
        for i in range(0, self.m):
            for j in range(0, self.k):
                numerator = np.exp(np.dot(self.x[i], theta[j]))
                denominator = 0
                for l in range(0, self.k):
                    denominator += np.exp(np.dot(self.x[i], theta[l]))
                gradient[j] += self.x[i] * (self.y[i, j] - numerator / denominator)

        penalty_gradient = (1 / self.c) * theta
        gradient = (-1 * gradient + penalty_gradient) / self.m
        return np.ravel(gradient)


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
