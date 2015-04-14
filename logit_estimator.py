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

        if grad_check > 5 * 10**-7:
            exit('Gradient failed check with an error of ' + str(grad_check))

        self.theta = optimize.fmin_bfgs(self.cost_function, self.theta,
                                        fprime=self.gradient_function,
                                        args=(X_mod, self.y),
                                        gtol=0.0000001, disp=False)

    def sigmoid(self, matrix):
        return 1 / (1 + numpy.exp(- matrix))

    def utility(self, X, theta):
        return numpy.dot(X, theta)

    def log_func(self, X_i, y_i):
        theta_T = numpy.transpose(self.theta)
        return numpy.log(1 + numpy.exp(-1 * y_i * numpy.dot(theta_T, X_i)))

    def cost_function(self, theta, X, y):
        predicted_probs = self.sigmoid(self.utility(X, theta))

        log_likelihood = ((-1 * y) * numpy.log(predicted_probs) -
                          (1 - y) * numpy.log(1 - predicted_probs))

        # l1 - this is not working yet, the math appears to be incorrect
        # penalty = ((1 / self.C) * numpy.sum(numpy.absolute(theta)))

        # l2
        penalty = ((1 / (2 * self.C * self.m)) * numpy.sum(theta[1:] ** 2))

        # Alternative math test
        # partial_cost = numpy.sum(list(map(self.log_func, X, y)))
        # reg_penalty = 0.5 * numpy.dot(numpy.transpose(theta[1:]), theta[1:])
        # cost = self.C * partial_cost + reg_penalty

        cost = log_likelihood.mean() + penalty
        self.cost = cost
        return cost

    def gradient_function(self, theta, X, y):
        predicted_probs = self.sigmoid(self.utility(X, theta))
        error = predicted_probs - y

        # penalty_gradient = (1 / self.C) * numpy.ones(theta.shape)  # l1
        penalty_gradient = (1 / self.C) * theta  # l2
        penalty_gradient[0] = 0

        gradient = (numpy.dot(error, X) + penalty_gradient) / self.m

        # Alternative math test
        # penalty_gradient = numpy.sum(theta[1:])
        # gradient = (self.C * numpy.dot(error, X) + penalty_gradient)

        return gradient
