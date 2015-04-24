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
        X = numpy.append(numpy.ones((X.shape[0], 1)), X, axis=1)

        # Add a ones column to X rather than fitting the intercept
        lr_r = LogisticRegression(penalty='l2', dual=False, tol=0.0000001,
                                  C=C, fit_intercept=False,
                                  class_weight='auto')

        lr_r.fit(X, y)
        return lr_r

    def estimate_home_made_model(X, y, C):
        lr = LogisticRegressionEstimator(X, y, C)
        lr.estimate()
        return lr

    def estimate_home_made_model_alt(X, y, C):
        lr = LogisticRegressionEstimator(X, y, C)
        lr.estimate_alt()
        return lr

# TODO:
# 1. Split LRE into LRE and LRE_alt, both of which will inherit
#    from a general class
# 2. Multi-class!!!
#    http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression

class LogisticRegressionEstimator:
    '''A home made implimentation of logist.C regression'''
    def __init__(self, X, y, C):
        self.X = numpy.append(numpy.ones((X.shape[0], 1)), X, axis=1)
        self.y = y
        self.n = X.shape[1] + 1
        self.m = X.shape[0]
        self.theta = numpy.random.randn(self.n)
        self.C = C
        self.cost = None

    def estimate(self):
        self.gradient_check(self.cost_function,
                            self.gradient_function,
                            self.theta, self.X, self.y)

        self.theta = optimize.fmin_bfgs(self.cost_function, self.theta,
                                        fprime=self.gradient_function,
                                        args=(self.X, self.y),
                                        gtol=0.0000001, disp=False)

    def cost_function(self, theta, X, y):
        predicted_probs = self.predict_probs(theta, X)
        log_likelihood = ((-1 * y) * numpy.log(predicted_probs) -
                          (1 - y) * numpy.log(1 - predicted_probs))

        regularisation = (0.5 / self.C * numpy.sum(theta[1:] ** 2))

        cost = log_likelihood.mean() + regularisation / self.m
        self.cost = cost
        return cost

    def gradient_function(self, theta, X, y):
        predicted_probs = self.predict_probs(theta, X)
        error = predicted_probs - y
        objective_grad = numpy.dot(error, X)

        penalty_gradient = (1 / self.C) * theta
        penalty_gradient[0] = 0

        gradient = (objective_grad + penalty_gradient) / self.m
        return gradient

    def sigmoid(self, vector):
        return 1 / (1 + numpy.exp(- vector))

    def utility(self, theta, X):
        return numpy.dot(X, theta)

    def predict_probs(self, theta, X):
        return self.sigmoid(self.utility(theta, X))

    def gradient_check(self, cost_function_alt, gradient_function_alt,
                       theta, X, y):
        grad_check = optimize.check_grad(cost_function_alt,
                                         gradient_function_alt,
                                         theta, X, y)

        if grad_check > 5 * 10**-7:
            print('Gradient failed check with an error of ' + str(grad_check))

    # ===== Testing some alternative math (based on liblinear) =====
    def estimate_alt(self):
        self.y[self.y == 0] = -1  # Assumes y vector is (1, 0)

        self.gradient_check(self.cost_function_alt,
                            self.gradient_function_alt,
                            self.theta, self.X, self.y)

        self.theta = optimize.fmin_bfgs(self.cost_function_alt, self.theta,
                                        fprime=self.gradient_function_alt,
                                        args=(self.X, self.y),
                                        gtol=0.0000001, disp=False)

    def cost_function_alt(self, theta, X, y):
        '''Alternative math test'''
        self.theta = theta

        objective_cost = 0
        for i in range(0, self.m):
            objective_cost += numpy.log(self.inverted_sigmoid(X[i], y[i]))

        regularisation = 0
        for j in range(1, self.n):
            regularisation += 0.5 * theta[j] ** 2

        cost = (regularisation / self.C + objective_cost) / self.m
        self.cost = cost
        return cost

    def gradient_function_alt(self, theta, X, y):
        '''Alternative math test'''
        self.theta = theta

        objective_grad = numpy.zeros_like(theta)
        for i in range(0, self.m):
            grad_mat = ((1 / self.inverted_sigmoid(X[i], y[i])) - 1) * y[i] * X[i]
            for j in range(0, self.n):
                objective_grad[j] += grad_mat[j]

        regularisation_grad = numpy.copy(theta)
        regularisation_grad[0] = 0

        gradient = (regularisation_grad / self.C + objective_grad) / self.m
        return gradient

    def inverted_sigmoid(self, X_i, y_i):
        sum = 0
        for j in range(0, self.n):
            sum += self.theta[j] * X_i[j]
        return 1 + numpy.exp(-1 * y_i * sum)
