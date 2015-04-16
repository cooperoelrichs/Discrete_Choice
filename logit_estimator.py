# Prototype logit estimation class
#
# This is a prototype class for estimating discrete choice logit models
# Python 3.4 64 bit with SciPy

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from scipy import optimize
import numpy as numpy


class LogitEstimator:
    '''A prototype class for logit estimation'''
    def scaler(X):
        return preprocessing.StandardScaler().fit(X)

    def estimate_scikit_learn_model(X, y, C):
        '''Estimate a scikit-learn multinomial logit model'''
        X_mod = numpy.append(numpy.ones((X.shape[0], 1)), X, axis=1)

        # Add a ones column to X rather than fitting the intercept
        lr_r = LogisticRegression(penalty='l1', dual=False, tol=0.0000001,
                                  C=C, fit_intercept=False,
                                  class_weight='auto')

        lr_r.fit(X_mod, y)
        return lr_r

    def estimate_home_made_model(X, y, C):
        lr = LogisticRegressionEstimator(X, y, C)
        lr.estimate()
        return lr

    def estimate_home_made_model_alt(X, y, C):
        y_edit = numpy.copy(y)
        y_edit[y_edit == 0] = -1

        y_edit = numpy.array([-1, 1])
        X = numpy.array([[-1], [1]])
        lr = LogisticRegressionEstimator(X, y_edit, C)
        lr.estimate_alt()
        return lr


class LogisticRegressionEstimator:
    '''A home made implimentation of logist.C regression'''
    def __init__(self, X, y, C):
        self.X = X
        self.y = y
        self.n = X.shape[1] + 1
        self.m = X.shape[0]
        self.theta = numpy.random.randn(self.n)
        self.C = C
        self.cost = None

    def estimate(self):
        X_mod = numpy.append(numpy.ones((self.X.shape[0], 1)),
                             self.X, axis=1)

        grad_check = optimize.check_grad(self.cost_function,
                                         self.gradient_function,
                                         self.theta, X_mod, self.y)

        if grad_check > 10**-6:
            exit('Gradient failed check with an error of ' + str(grad_check))

        self.theta = optimize.fmin_bfgs(self.cost_function, self.theta,
                                        fprime=self.gradient_function,
                                        args=(X_mod, self.y),
                                        gtol=0.0000001, disp=False)

    def sigmoid(self, matrix):
        return 1 / (1 + numpy.exp(- matrix))

    def utility(self, X, theta):
        return numpy.dot(X, theta)

    def cost_function(self, theta, X, y):
        predicted_probs = self.sigmoid(self.utility(X, theta))

        log_likelihood = ((-1 * y) * numpy.log(predicted_probs) -
                          (1 - y) * numpy.log(1 - predicted_probs))

        penalty = ((1 / (2 * self.C * self.m)) * numpy.sum(theta[1:] ** 2))

        cost = log_likelihood.mean() + penalty
        self.cost = cost
        return cost

    def gradient_function(self, theta, X, y):
        predicted_probs = self.sigmoid(self.utility(X, theta))
        error = predicted_probs - y

        penalty_gradient = (1 / self.C) * theta
        penalty_gradient[0] = 0

        gradient = (numpy.dot(error, X) + penalty_gradient) / self.m
        return gradient

    # ===== Testing some alternative math (based on liblinear) =====
    def estimate_alt(self):
        X_mod = numpy.append(numpy.ones((self.X.shape[0], 1)),
                             self.X, axis=1)

        grad_check = optimize.check_grad(self.cost_function_alt,
                                         self.gradient_function_alt,
                                         self.theta, X_mod, self.y)

        if grad_check > 5 * 10**-7:
            exit('Gradient failed check with an error of ' + str(grad_check))

        self.theta = optimize.fmin_bfgs(self.cost_function_alt, self.theta,
                                        fprime=self.gradient_function_alt,
                                        args=(X_mod, self.y),
                                        gtol=0.0000001, disp=False)

    def sigmoid_log(self, X_i, y_i):
        return numpy.log(self.inverted_sigmoid(X_i, y_i))

    def inverted_sigmoid(self, X_i, y_i):
        theta_T = numpy.transpose(self.theta)
        return 1 + numpy.exp(-1 * y_i * numpy.dot(theta_T, X_i))

    def grad_math(self, X_i, y_i):
        return numpy.dot(((1 / self.inverted_sigmoid(X_i, y_i)) - 1) * y_i, X_i)

    def cost_function_alt(self, theta, X, y):
        '''Alternative math test'''
        partial_cost = numpy.sum(list(map(self.sigmoid_log, X, y)))
        penalty = 0.5 * numpy.dot(numpy.transpose(theta[1:]), theta[1:])
        cost = penalty + self.C * partial_cost

        self.cost = cost
        return cost

    def gradient_function_alt(self, theta, X, y):
        '''Alternative math test'''

        theta = numpy.array([0.2358176, 0.77858431])
        self.theta = theta

        penalty_gradient = numpy.copy(theta)
        penalty_gradient[0] = 0
        cost_gradient = numpy.sum(list(map(self.grad_math, X, y)))

        gradient = (penalty_gradient + self.C * cost_gradient)
        return gradient
