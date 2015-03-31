# Prototype logit estimation class
#
# This is a prototype class for estimating discrete choice logit models
# Python 3.4 64 bit with SciPy

from sklearn.linear_model import LogisticRegression
from scipy import optimize
import numpy as numpy


class LogitEstimator:
    '''A prototype class for logit estimation'''
    def estimate_scikit_learn_model(data_x, data_y):
        '''Estimate a scikit-learn multinomial logit model'''
        data_x_mod = numpy.append(numpy.ones((data_x.shape[0], 1)),
                                  data_x, axis=1)

        lr_n = LogisticRegression(dual=False, tol=0.0001,
                                  fit_intercept=True,
                                  class_weight='auto')
        # lr_r = LogisticRegression(penalty='l1', dual=False, tol=0.0001,
        #                           C=20, fit_intercept=True,
        #                           class_weight='auto')

        lr_n.fit(data_x_mod, data_y)
        # lr_r.fit(data_x, data_y)
        print('Coefficients for MNL')
        print(lr_n.coef_)
        # print('Coefficients for regularised MNL')
        # print(lr_r.coef_)

    def estimate_logit_model(data_x, data_y):
        lr = LogisticRegressionEstimator(data_x, data_y)
        lr.estimate()


class LogisticRegressionEstimator:
    '''A home made implimentation of logistic regression'''
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.theta = numpy.random.randn(self.data_x.shape[1] + 1)

    def estimate(self):
        x_mod = numpy.append(numpy.ones((self.data_x.shape[0], 1)),
                             self.data_x, axis=1)
        self.theta = optimize.fmin_bfgs(self.cost_function, self.theta,
                                        fprime=self.gradient_function,
                                        args=(x_mod, self.data_y))

        print('And the paramaters are: ')
        print(self.theta)

    def cost_function(self, theta, x, y):
        # Predict results
        predictions = numpy.dot(x, theta)
        # Use the sigmoid function to calculate predicted probabilities
        predicted_probs = 1 / (1 + numpy.exp(- predictions))

        log_likelihood = ((-1 * y) * numpy.log(predicted_probs) -
                          (1 - y) * numpy.log(1 - predicted_probs))
        cost = log_likelihood.mean()
        return cost

    def gradient_function(self, theta, x, y):
        # Predict results
        predictions = numpy.dot(x, theta)
        # Use the sigmoid function to calculate predicted probabilities
        predicted_probs = 1 / (1 + numpy.exp(- predictions))

        error = predicted_probs - y
        gradient = numpy.dot(error, x) / y.size
        return gradient
