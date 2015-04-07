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
    def estimate_scikit_learn_model(data_x, data_y):
        '''Estimate a scikit-learn multinomial logit model'''
        scaler = preprocessing.StandardScaler().fit(data_x)
        data_x_trans = scaler.transform(data_x)

        data_x_mod = numpy.append(numpy.ones((data_x_trans.shape[0], 1)),
                                  data_x, axis=1)

        lr_r = LogisticRegression(penalty='l1', dual=False, tol=0.0001,
                                  C=10, fit_intercept=True,
                                  class_weight='auto')

        lr_r.fit(data_x_mod, data_y)
        print('Coefficients for MNL')
        print(lr_r.coef_)

    def estimate_home_made_model(data_x, data_y):
        scaler = preprocessing.StandardScaler().fit(data_x)
        data_x_trans = scaler.transform(data_x)

        lr = LogisticRegressionEstimator(data_x_trans, data_y, 10)
        lr.estimate()

        print('And the paramaters are: ')
        print(lr.theta)


class LogisticRegressionEstimator:
    '''A home made implimentation of logistic regression'''
    def __init__(self, data_x, data_y, c):
        self.data_x = data_x
        self.data_y = data_y
        self.theta = numpy.random.randn(self.data_x.shape[1] + 1)
        self.c = c

    def estimate(self):
        x_mod = numpy.append(numpy.ones((self.data_x.shape[0], 1)),
                             self.data_x, axis=1)
        self.theta = optimize.fmin_bfgs(self.cost_function, self.theta,
                                        fprime=self.gradient_function,
                                        args=(x_mod, self.data_y),
                                        gtol=0.0001)

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(- x))

    def utility(self, x, theta):
        return numpy.dot(x, theta)

    def cost_function(self, theta, x, y):
        # Use the sigmoid function to calculate predicted probabilities
        predicted_probs = self.sigmoid(self.utility(x, theta))

        if 1 in predicted_probs:
            exit('1 found')

        log_likelihood = ((-1 * y) * numpy.log(predicted_probs) -
                          (1 - y) * numpy.log(1 - predicted_probs))

        penalty = (1 / self.c) * numpy.sum(numpy.absolute(theta))
        cost = log_likelihood.mean() + penalty

        return cost

    def gradient_function(self, theta, x, y):
        # Use the sigmoid function to calculate predicted probabilities
        predicted_probs = self.sigmoid(self.utility(x, theta))

        error = predicted_probs - y
        penalty_gradient = (1 / self.c)  # Quick guess at the penalty - this needs to be fixed
        gradient = (numpy.dot(error, x) + penalty_gradient) / y.size

        return gradient
