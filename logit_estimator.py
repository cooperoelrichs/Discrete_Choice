# Prototype logit estimation class
#
# This is a prototype class for estimating discrete choice logit models
# Python 3.4 64 bit with SciPy

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from scipy import optimize
import numpy as np


class LogitEstimator:
    '''This class just prepares and runs actual model estimation classes'''
    def scaler(X):
        return preprocessing.StandardScaler().fit(X)

    def estimate_scikit_learn_model(X, y, C):
        '''Estimate a scikit-learn multinomial logit model'''
        X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        # X[0, 0] = 0

        # Add a ones column to X rather than fitting the intercept
        lr_r = LogisticRegression(penalty='l2', dual=False, tol=0.0000001,
                                  C=C, fit_intercept=False,
                                  class_weight='auto',
                                  multi_class='multinomial',
                                  solver='lbfgs')

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
        lr.estimate()
        # lr.cost_function(lr.theta, lr.X, lr.y)
        # lr.gradient_function(lr.theta, lr.X, lr.y)
        return lr

# TODO:
# 2. Multi-class!!!
#    http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
# 3. Constants on my MNL are weird and really small


class ModelEstimator(object):
    '''A home made implimentation of logist.C regression'''
    def __init__(self, X, y, C):
        self.X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        # self.X[0, 0] = 0
        self.y = LabelBinarizer().fit_transform(y)
        self.n = X.shape[1] + 1
        self.m = X.shape[0]
        self.k = self.y.shape[1]
        self.theta = np.random.randn(self.k, self.n)
        # self.theta = np.zeros((self.k, self.n))
        self.theta_f = np.ravel(self.theta)
        self.C = C
        self.cost = None
        self.iteration = 0

    def estimate(self):
        self.prep_work()
        self.gradient_check(self.cost_function,
                            self.gradient_function,
                            self.theta_f, self.X, self.y)

        self.theta_f = optimize.fmin_bfgs(self.cost_function, self.theta_f,
                                          fprime=self.gradient_function,
                                          args=(self.X, self.y),
                                          gtol=0.0000001, disp=False)

    def gradient_check(self, cost_function, gradient_function,
                       theta, X, y):
        grad_check = optimize.check_grad(cost_function,
                                         gradient_function,
                                         theta, X, y)

        if grad_check > 1 * 10**-6:
            print('Gradient failed check with an error of ' + str(grad_check))


class NestedLogitEstimator(ModelEstimator):
    '''Nested Logit!

    Good resources:
        1. http://www.civil.iitb.ac.in/~kvkrao/CE%20780%20
           Behavioural%20Travel%20Modelling/NL.pdf
        2. http://eml.berkeley.edu/books/train1201.pdf
        3. http://en.wikipedia.org/wiki/Maximum_likelihood

    Plan:
        1. Start with only one nest level and use formula (4.2) from
           K. Train, http://eml.berkeley.edu/books/train1201.pdf
        2. Estimate the gradient for this function numerically!
    '''
    def prep_work(self):
        '''Nothing to see here'''

    def cost_function(self, theta_f, X, y):
        '''Costs calc'''
        theta = np.reshape(theta_f[:self.k * self.nests], (self.k, self.n))
        lda = np.reshape(theta_f[self.k * self.nests:], (self.nests, self.n))
        self.theta = theta

        # Two level NL
        # Need to logsum the probabilities!
        for i in range(0, self.m):
            # lda is the list of the indipendence paramaters
            j = y[i]  # Get the index of the chosen, assumed y is [1, ..., k]
            l = self.nests[j]
            V = np.dot(X[i], theta[j])
            num = (np.exp(V / lda[l]) *
                   nest_sum[l] ** (lda[l] - 1)
                   # nest_sum is an array of the sum of each exp(V/lambda)
                   # for all options in a nest
            den = np.sum(nest_sum[l_2] ** lda[l_2])

            P = num / den
            # P = estimated probability for the option j
            # Where j is the chosen option in experiment i, and j
            # is in nest l

    def grat_function(self, theta_f, X, y):
        '''Gradient calc'''


class MultiNomialLogitEstimator(ModelEstimator):
    '''
    Based on:
    http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
    '''
    def prep_work(self):
        '''No prep work required'''

    def cost_function(self, theta_f, X, y):
        '''
        m - number of data points
        n - number of features
        k - number of classes
        X - m * n
        y - m * k
        theta - k * n
        '''

        theta = np.reshape(theta_f, (self.k, self.n))
        self.theta = theta
        cost = 0
        for i in range(0, self.m):
            for j in range(0, self.k):
                numerator = np.exp(np.dot(X[i], theta[j]))
                denominator = 0
                for l in range(0, self.k):
                    denominator += np.exp(np.dot(X[i], theta[l]))
                cost += y[i, j] * np.log(numerator / denominator)

        # regularisation = (0.5 / self.C * np.sum(theta[:, 1:] ** 2))
        regularisation = (0.5 / self.C * np.sum(theta ** 2))
        cost = (-1 * cost + regularisation) / self.m
        self.cost = cost
        return cost

    def gradient_function(self, theta_f, X, y):
        theta = np.reshape(theta_f, (self.k, self.n))
        self.theta = theta
        gradient = np.zeros_like(theta)
        for i in range(0, self.m):
            for j in range(0, self.k):
                numerator = np.exp(np.dot(X[i], theta[j]))
                denominator = 0
                for l in range(0, self.k):
                    denominator += np.exp(np.dot(X[i], theta[l]))
                gradient[j] += X[i] * (y[i, j] - numerator / denominator)

        penalty_gradient = (1 / self.C) * theta
        # penalty_gradient[:, 0] = 0
        # print(penalty_gradient)
        gradient = (-1 * gradient + penalty_gradient) / self.m

        self.grad = gradient
        return np.ravel(self.grad)


class LogisticRegressionEstimator(ModelEstimator):
    def prep_work(self):
        '''No prep work required'''

    def cost_function(self, theta, X, y):
        self.theta = theta
        predicted_probs = self.predict_probs(theta, X)
        log_likelihood = ((-1 * y) * np.log(predicted_probs) -
                          (1 - y) * np.log(1 - predicted_probs))

        regularisation = (0.5 / self.C * np.sum(theta[1:] ** 2))

        cost = log_likelihood.mean() + regularisation / self.m
        self.cost = cost
        return cost

    def gradient_function(self, theta, X, y):
        self.theta = theta
        predicted_probs = self.predict_probs(theta, X)
        error = predicted_probs - y
        objective_grad = np.dot(error, X)

        penalty_gradient = (1 / self.C) * theta
        penalty_gradient[0] = 0

        gradient = (objective_grad + penalty_gradient) / self.m
        return gradient

    def sigmoid(self, vector):
        return 1 / (1 + np.exp(- vector))

    def utility(self, theta, X):
        return np.dot(X, theta)

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
            objective_cost += np.log(self.inverted_sigmoid(X[i], y[i]))

        regularisation = 0
        for j in range(1, self.n):
            regularisation += 0.5 * theta[j] ** 2

        cost = (regularisation / self.C + objective_cost) / self.m
        self.cost = cost
        return cost

    def gradient_function(self, theta, X, y):
        self.theta = theta

        objective_grad = np.zeros_like(theta)
        for i in range(0, self.m):
            grad_mat = (((1 / self.inverted_sigmoid(X[i], y[i])) - 1) *
                        y[i] * X[i])
            for j in range(0, self.n):
                objective_grad[j] += grad_mat[j]

        regularisation_grad = np.copy(theta)
        regularisation_grad[0] = 0

        gradient = (regularisation_grad / self.C + objective_grad) / self.m
        return gradient

    def inverted_sigmoid(self, X_i, y_i):
        sum = 0
        for j in range(0, self.n):
            sum += self.theta[j] * X_i[j]
        return 1 + np.exp(-1 * y_i * sum)
