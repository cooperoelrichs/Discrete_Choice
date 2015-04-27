# Prototype logit estimation class
#
# This is a prototype class for estimating discrete choice logit models
# Python 3.4 64 bit with SciPy

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from scipy import optimize
import numpy as numpy


class LogitEstimator:
    '''This class just prepares and runs actual model estimation classes'''
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
        lr = AltLogisticRegressionEstimator(X, y, C)
        lr.estimate()
        return lr

    def estimate_multinomial_model(X, y, C):
        lr = MultiNomialLogitEstimator(X, y, C)
        lr.cost_function(lr.theta, lr.X, lr.y)
        return lr

# TODO:
# 2. Multi-class!!!
#    http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression


class ModelEstimator(object):
    '''A home made implimentation of logist.C regression'''
    def __init__(self, X, y, C):
        self.X = numpy.append(numpy.ones((X.shape[0], 1)), X, axis=1)
        self.y = LabelBinarizer().fit_transform(y)
        self.n = X.shape[1] + 1
        self.m = X.shape[0]
        self.k = self.y.shape[1]
        self.theta = numpy.random.randn(self.k, self.n)
        self.C = C
        self.cost = None

    def estimate(self):
        self.prep_work()
        self.gradient_check(self.cost_function,
                            self.gradient_function,
                            self.theta, self.X, self.y)

        self.theta = optimize.fmin_bfgs(self.cost_function, self.theta,
                                        fprime=self.gradient_function,
                                        args=(self.X, self.y),
                                        gtol=0.0000001, disp=False)

    def gradient_check(self, cost_function, gradient_function,
                       theta, X, y):
        grad_check = optimize.check_grad(cost_function,
                                         gradient_function,
                                         theta, X, y)

        if grad_check > 5 * 10**-7:
            print('Gradient failed check with an error of ' + str(grad_check))


class MultiNomialLogitEstimator(ModelEstimator):
    def prep_work(self):
        '''No prep work required'''

    def cost_function(self, theta, X, y):
        '''
        Calculate some MNL costs here
        y(i) is {1, 2, ..., k}
        '''
        cost_sum = 0
        for i in range(0, self.m):
            for j in range(0, self.k):
                numerator = numpy.exp(numpy.dot(X[i], theta[j]))
                denominator = 0
                for l in range(0, self.k):
                    denominator += numpy.exp(numpy.dot(X[i], theta[l]))
                cost_sum += y[i, j] * numpy.log(numerator / denominator)

        self.cost = cost_sum / self.m
        return self.cost

    def gradient_function(self, theta, X, y):
        '''Calc some graidents here'''


class LogisticRegressionEstimator(ModelEstimator):
    def prep_work(self):
        '''No prep work required'''

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


class AltLogisticRegressionEstimator(ModelEstimator):
    '''Testing some alternative math (based on liblinear)'''
    def prep_work(self):
        self.y[self.y == 0] = -1  # y vector must be (1, -1)

    def cost_function(self, theta, X, y):
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

    def gradient_function(self, theta, X, y):
        self.theta = theta

        objective_grad = numpy.zeros_like(theta)
        for i in range(0, self.m):
            grad_mat = (((1 / self.inverted_sigmoid(X[i], y[i])) - 1) *
                        y[i] * X[i])
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
