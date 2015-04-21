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
        lr = LogisticRegressionEstimator(X, y, C)
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
            print('Gradient failed check with an error of ' + str(grad_check))

        self.theta = optimize.fmin_bfgs(self.cost_function, self.theta,
                                        fprime=self.gradient_function,
                                        args=(X_mod, self.y),
                                        gtol=0.0000001, disp=False)

    def cost_function(self, theta, X, y):
        predicted_probs = self.sigmoid(self.utility(X, theta))

        log_likelihood = ((-1 * y) * numpy.log(predicted_probs) -
                          (1 - y) * numpy.log(1 - predicted_probs))

        penalty = ((1 / (2 * self.C * self.m)) * numpy.sum(theta[1:] ** 2))

        cost = log_likelihood.mean() + penalty
        self.cost = cost
        return cost

    def sigmoid(self, vector):
        return 1 / (1 + numpy.exp(- vector))

    def utility(self, X, theta):
        return numpy.dot(X, theta)

    def gradient_function(self, theta, X, y):
        predicted_probs = self.sigmoid(self.utility(X, theta))
        error = predicted_probs - y

        penalty_gradient = (1 / self.C) * theta
        penalty_gradient[0] = 0

        gradient = (numpy.dot(error, X) + penalty_gradient) / self.m
        return gradient

    # ===== Testing some alternative math (based on liblinear) =====
    def estimate_alt(self):
        self.y_orig = numpy.copy(self.y)
        self.y[self.y == 0] = -1

        X_mod = numpy.append(numpy.ones((self.X.shape[0], 1)),
                             self.X, axis=1)

        grad_check = optimize.check_grad(self.cost_function_alt,
                                         self.gradient_function_alt,
                                         self.theta, X_mod, self.y)

        if grad_check > 5 * 10**-7:
            print('Gradient failed check with an error of ' + str(grad_check))

        self.theta = optimize.fmin_bfgs(self.cost_function_alt, self.theta,
                                        fprime=self.gradient_function_alt,
                                        args=(X_mod, self.y),
                                        gtol=0.0000001, disp=False)

        print(list(map(numpy.sum, map(self.grad_math, X_mod, self.y)))[:3])

        cost_gradient = numpy.sum(list(map(self.grad_math, X_mod, self.y)))
        predicted_probs = self.sigmoid(self.utility(X_mod, self.theta))
        error = predicted_probs - self.y
        alt_cost_gradient = numpy.dot(error, X_mod)
        print('Alt partial grad ' + str(cost_gradient))
        print('Alt partial grad w standard func ' + str(alt_cost_gradient))

    def sigmoid_log(self, X_i, y_i):
        return numpy.log(self.inverted_sigmoid(X_i, y_i))

    def inverted_sigmoid(self, X_i, y_i):
        theta_T = numpy.transpose(self.theta)
        return 1 + numpy.exp(-1 * y_i * numpy.dot(theta_T, X_i))

    def grad_math(self, X_i, y_i):
        return numpy.dot(((1 / self.inverted_sigmoid(X_i, y_i)) - 1) * y_i, X_i)

    def cost_function_alt(self, theta, X, y):
        '''Alternative math test'''
        self.theta = theta
        partial_cost = numpy.sum(list(map(self.sigmoid_log, X, y)))

        penalty = 0.5 * numpy.dot(numpy.transpose(theta[1:]), theta[1:])

        cost = (penalty + self.C * partial_cost) / self.m
        self.cost = cost
        return cost

    def gradient_function_alt(self, theta, X, y):
        '''Alternative math test'''
        self.theta = theta
        # cost_gradient = numpy.sum(list(map(self.grad_math, X, y)))
        cost_gradient = numpy.array(list(map(numpy.sum, numpy.transpose(map(self.grad_math, X, y)))))

        penalty_gradient = numpy.copy(theta)
        penalty_gradient[0] = 0

        gradient = (penalty_gradient + self.C * cost_gradient) / self.m
        return gradient
