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

    def estimate_scikit_learn_model(X, y):
        '''Estimate a scikit-learn multinomial logit model'''
        X_mod = numpy.append(numpy.ones((X.shape[0], 1)), X, axis=1)

        # Add a ones column to X rather than fitting the intercept
        lr_r = LogisticRegression(penalty='l1', dual=False, tol=0.0000001,
                                  C=10, fit_intercept=False,
                                  class_weight='auto')

        lr_r.fit(X_mod, y)
        return lr_r

    def estimate_home_made_model(X, y):
        lr = LogisticRegressionEstimator(X, y, 10)
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

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(- x))

    def utility(self, x, theta):
        return numpy.dot(x, theta)

    def cost_function(self, theta, x, y):
        predicted_probs = self.sigmoid(self.utility(x, theta))

        if 1 in predicted_probs:
            exit('1 found')

        log_likelihood = ((-1 * y) * numpy.log(predicted_probs) -
                          (1 - y) * numpy.log(1 - predicted_probs))

        # l1 - this is not working yet, the math appears to be incorrect
        # penalty = ((1 / self.C) * numpy.sum(numpy.absolute(theta)))

        # l2
        penalty = ((1 / (2 * self.C * self.m)) * numpy.sum(theta[1:] ** 2))

        cost = log_likelihood.mean() + penalty
        self.cost = cost
        return cost

    def gradient_function(self, theta, x, y):
        predicted_probs = self.sigmoid(self.utility(x, theta))
        error = predicted_probs - y

        # penalty_gradient = (1 / self.C) * numpy.ones(theta.shape)  # l1
        penalty_gradient = (1 / self.C) * theta  # l2

        penalty_gradient[0] = 0
        gradient = (numpy.dot(error, x) + penalty_gradient) / self.m

        return gradient
