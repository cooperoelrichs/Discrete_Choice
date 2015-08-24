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

    # @staticmethod
    # def get_parameters():
    #     raise NotImplementedError("Don't instantiate the Base Class")

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

    def __init__(self, x, y, c, alts, av, weights, initial_parameters, fixed_parameters, utility_functions,
                 lambda_params):
        super(NestedLogitEstimator, self).__init__(x, y, c)

        self.alts = np.array(alts)
        self.av = av
        self.weights = weights
        self.nest_index = np.zeros_like(self.y[0])  # = [0, 0, 1, 1]
        self.h = len(self.alts)  # number of nests
        self.nest_lens = [len(x) for x in self.alts]
        self.lambda_params = lambda_params

        for i, x in enumerate(self.alts):
            for j in x:
                self.nest_index[j] = i

        self.initial_parameters = initial_parameters
        self.fixed_parameters = fixed_parameters
        self.utility_functions = utility_functions

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

        # TODO: Get the MNL estimator running, for comparison purposes!

        # np.seterr(all='ignore')

        nest_sums = np.zeros((self.m, self.h), dtype=np.longdouble)
        v_scaled = np.zeros((self.m, self.k), dtype=np.longdouble)
        for i in range(0, self.m):
            for l in range(0, self.h):
                for j in range(0, self.nest_lens[l]):
                    if self.av[i, self.alts[l][j]] == 1:
                        v_ilj = self.utility_functions[self.alts[l][j]](self.x[i], parameters)
                        v_scaled[i, self.alts[l][j]] = v_ilj / parameters[self.lambda_params[l]]
                        # if v_scaled[i, self.alts[l][j]] > 700.0 or v_scaled[i, self.alts[l][j]] < -700.0:
                        #     print('%0.0f - %0.0f - %0.6f' %
                        #           (v_ilj, v_scaled[i, self.alts[l][j]], np.exp(v_scaled[i, self.alts[l][j]])))
                        nest_sums[i, l] += np.exp(v_scaled[i, self.alts[l][j]])

        p = np.zeros(self.m)
        for i in range(0, self.m):
            j = self.y_index[i]
            l = self.nest_index[j]
            if self.av[i, j] == 0:
                raise RuntimeError('Chosen is unavailable')
            num = (np.exp(v_scaled[i, j]) *
                   (nest_sums[i, l] ** (parameters[self.lambda_params[l]] - 1.0)))
            dom = 0
            for l_2 in range(0, self.h):
                dom += nest_sums[i, l_2] ** parameters[self.lambda_params[l_2]]
            p[i] = num / dom

            # print([i, p[i], num, dom])  # , self.x[i]])
            # if i == 100: exit()

        cost = - np.dot(np.log(p), np.transpose(self.weights))  # / self.m
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


class MultinomialLogitEstimator(ModelEstimator):
    """
    Based on:
    http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
    """

    def __init__(self, x, y, c):
        super(MultinomialLogitEstimator, self).__init__(x, y, c)
        self.theta = np.random.randn(self.k, self.n)
        self.initial_parameters = np.ravel(self.theta)

    def cost_function(self, parameters):
        """
        m - number of data points
        n - number of features
        k - number of classes
        X - m * n
        y - m * k
        theta - k * n
        """

        theta = np.reshape(parameters, (self.k, self.n))
        cost = 0
        for i in range(0, self.m):
            for j in range(0, self.k):
                numerator = np.exp(np.dot(self.x[i], theta[j]))
                # numerator = np.exp(self.utility_functions[j](self.x[i], parameters))
                denominator = 0
                for l in range(0, self.k):
                    denominator += np.exp(np.dot(self.x[i], theta[l]))
                    # denominator += np.exp(self.utility_functions[l](self.x[i], parameters))
                cost += self.y[i, j] * np.log(numerator / denominator)

        # regularisation = (0.5 / self.c * np.sum(theta[:, 1:] ** 2))
        regularisation = (0.5 / self.c * np.sum(theta ** 2))
        cost = (-1 * cost + regularisation) / self.m
        return cost

    def gradient_function(self, parameters):
        self.iteration += 1
        theta = np.reshape(parameters, (self.k, self.n))
        gradient = np.zeros_like(theta)
        for i in range(0, self.m):
            for j in range(0, self.k):
                numerator = np.exp(np.dot(self.x[i], theta[j]))
                denominator = 0
                for l in range(0, self.k):
                    denominator += np.exp(np.dot(self.x[i], theta[l]))
                gradient[j] += self.x[i] * (self.y[i, j] - numerator / denominator)

        print((str(self.iteration) + ' - ' +
               str(parameters[0]) + ' - ' +
               str(gradient[0][0])))

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
