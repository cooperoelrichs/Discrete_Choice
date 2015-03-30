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
        lr_n = LogisticRegression(dual=False, tol=0.0001,
                                  fit_intercept=True,
                                  class_weight='auto')
        lr_r = LogisticRegression(penalty='l1', dual=False, tol=0.0001,
                                  C=20, fit_intercept=True,
                                  class_weight='auto')

        lr_n.fit(data_x, data_y)
        lr_r.fit(data_x, data_y)
        print('Coefficients for MNL')
        print(lr_n.coef_)
        print('Coefficients for regularised MNL')
        print(lr_r.coef_)

    def estimate_logit_model(data_x, data_y):
        LogisticRegressionEstimator.estimate(data_x, data_y)


class LogisticRegressionEstimator:
    '''A home made implimentation of logistic regression'''
    def estimate(data_x, data_y):
        theta_initial = 1  # numpy.random.randn()
        x_mod = numpy.append(numpy.ones((data_x.shape[0], 1)), data_x, axis=1)
        theta = optimize.fmin_bfgs(cost_function, theta_initial,
                                   fprime=gradient_function,
                                   args=(x_mod, data_y))

        print('And the paramater is ... ' + theta)

    def sigmoid(self, data_x):
        return 1 / (1 + numpy.exp(- data_x))

    def cost_function(self, theta, data_x, data_y):
        predicted_probabilities = self.predict_probabilities(data_x, theta)
        log_likelihood = ((-1 * data_y) * numpy.log(predicted_probabilities) -
                          (1 - data_y) * numpy.log(1 - predicted_probabilities))
        cost = log_likelihood.mean()
        return cost

    def predict_probabilities(self, x, theta):
        return self.sigmoid(numpy.dot(x, theta))

    def gradient_function(self, theta, data_x, data_y):
        predicted_probability = self.sigmoid(numpy.dot(data_x, theta))
        error = predicted_probability - data_y
        gradient = numpy.dot(error, data_x) / data_y.size
        return gradient
