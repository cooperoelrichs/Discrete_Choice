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

    def estimate_nested_model(X, y, C, alts):
        lr_nl = NestedLogitEstimator(X, y, C, alts=alts)
        init_nl_cost = lr_nl.cost_function(lr_nl.theta_f, lr_nl.X, lr_nl.y)
        lr_mnl = MultiNomialLogitEstimator(X, y, 999999999)
        init_mnl_cost = lr_mnl.cost_function(lr_nl.theta, lr_nl.X, lr_nl.y)

        lr_nl.estimate()
        print('initial MNL results - cost: %.6f' % init_mnl_cost)
        print('initial NL results  - cost: %.6f' % init_nl_cost)
        return lr_nl

    def _estimate_nested_model(X, y, C):
        X = np.array([[1, 2], [0, 0]])
        theta = np.array([[2, 0, 2], [-1, 2, 1], [0, 1, 0], [0, 0, 0]])
        lambdas = np.array([0.5, 1])
        y = np.array([0, 1, 2, 3])

        theta_f = np.ravel(theta)
        theta_f = np.append(theta_f, lambdas)

        lr_nl = NestedLogitEstimator(X, y, C, alts=[[0, 1], [2, 3]])
        X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        lr_nl.cost_function(theta_f, X, y)
        print('NL results  - cost: %.6f' % lr_nl.cost)


class ModelEstimator(object):
    '''A home made implimentation of logist.C regression'''
    def __init__(self, X, y, C, alts=None, fixed_params=set()):
        np.seterr(all='raise')
        self.X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        # self.X[0, 0] = 0
        self.y = LabelBinarizer().fit_transform(y)
        self.y_index = y
        self.n = X.shape[1] + 1
        self.m = X.shape[0]
        self.k = self.y.shape[1]
        self.theta = np.random.randn(self.k, self.n)
        # self.theta = np.ones((self.k, self.n))
        self.theta_f = np.ravel(self.theta)
        self.C = C
        self.cost = None
        self.iteration = 0
        machine_epsilon = np.finfo(np.float64).eps
        self.sqrt_eps = np.sqrt(machine_epsilon)
        self.alts = alts
        # self.fixed_params = fixed_params
        self.prep_work()

    def estimate(self):
        self.gradient_check(self.cost_function,
                            self.gradient_function,
                            self.theta_f, self.X, self.y)

        self.theta_f = optimize.fmin_bfgs(self.cost_function, self.theta_f,
                                          fprime=self.gradient_function, args=(self.X, self.y),
                                          gtol=0.001, disp=False)

    def gradient_check(self, cost_function, gradient_function,
                       theta, X, y):
        grad_check = optimize.check_grad(cost_function,
                                         gradient_function,
                                         theta, X, y)

        if abs(grad_check) > 6 * 10**-6:
            error = 'Gradient failed check with an error of ' + str(grad_check)
            raise ValueError(error)


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
        self.alts = np.array(self.alts)
        self.nest_index = np.zeros_like(self.y[0])  # = [0, 0, 1, 1]
        self.h = len(self.alts)
        self.lambdas = np.ones(self.h)  # np.random.randn(self.h)
        self.nest_lens = [len(x) for x in self.alts]
        # self.V = np.zeros((self.m, self.k))
        self.theta_f = np.append(self.theta_f, self.lambdas)

        for i, x in enumerate(self.alts):
            for j in x:
                self.nest_index[j] = i

        # Experimental utility function specification
        self.fixed_parameters = set([0, 4])
        self.parameters = {
            0: np.random.rand(),
            1: np.random.rand(),
            2: np.random.rand(),
            3: 1,
            4: 1,
        }

        def u1(X_i, params): np.dot(X_i, params.get(1, 2,  3))
        self.utility_functions = {
            0: u1,
        }

    def cost_function(self, theta_f, X, y):
        '''
        Based on formula (4.2) from:
        K. Train, http://eml.berkeley.edu/books/train1201.pdf

        Indices
        m - number of experiments
        n - number of features
        k - number of classes
        h - number of nests
        j - class, subset some nest l, [1, ..., k]
        i - current experiment, [1, ..., m]
        l - nest, [1, ..., h]

        Storage containers
        V - 2D matrix of utilities, m * k
        theta - vector of utility parameters, m
        lambdas - vector of nest parameters, h
        nest_lens - vector of nest lengths, h
        nest_sums - 2D matrix of e^(V/lambda), m * h
        alts - vector of vectors, maps nest and nested alternative to
               overall alternative, h * (classes in nest)
        '''

        # TODO: Alternative specific utility functions
        # TODO: Fixed parameters
        #
        # 1. Have a dict of parameters (starting with an initial value)
        #    and have a set specifying parameters that are fixed
        #       { 0 : 0.5, ... }
        #       set([2, 5, ...]) # skip parameters that are in this set
        # 2. Have a dict of utility functions, each of which takes the
        #    parameter dict as an input
        #       def u(X_i, parameters): np.dot(X_i, parameters[...])
        #       { 0 : u(X_i, parameters), ... }

        self.lambdas = theta_f[-1 * self.h:]
        self.theta = np.reshape(theta_f[:-1 * self.h], (self.k, self.n))

        nest_sums = np.zeros((self.m, self.h))
        V = np.zeros((self.m, self.k))
        for i in range(0, self.m):
            for l in range(0, self.h):
                for j in range(0, self.nest_lens[l]):
                    V_ilj = np.dot(X[i], self.theta[self.alts[l][j]])
                    V[i, self.alts[l][j]] = V_ilj
                    V_scaled = V_ilj / self.lambdas[l]
                    # if V_scaled > 200.0 or V_scaled < -200.0:
                    #     # We are getting very small lambda values sometimes.
                    #     # np.exp(x), where x > 709, causes an overflow
                    #     print('%0.6f - %s - %0.6f' %
                    #           (V_ilj, str(self.lambdas), V_scaled))
                    nest_sums[i, l] += np.exp(V_scaled)

        P = np.zeros((self.m))
        for i in range(0, self.m):
            j = self.y_index[i]
            l = self.nest_index[j]
            num = (np.exp(V[i, j] / self.lambdas[l]) *
                   (nest_sums[i, l] ** (self.lambdas[l] - 1)))
            dom = 0
            for l_2 in range(0, self.h):
                dom += nest_sums[i, l_2] ** self.lambdas[l_2]
            P[i] = num / dom

        self.cost = - np.sum(np.log(P)) / self.m
        return self.cost

    def gradient_function(self, theta_f, X, y):
        '''Serious numerical gradient stuff'''
        self.iteration += 1
        self.lambdas = theta_f[-1 * self.h:]
        self.theta = np.reshape(theta_f[:-1 * self.h], (self.k, self.n))

        base_cost = self.cost_function(theta_f, self.X, self.y)

        gradient = np.zeros_like(theta_f)
        for p in range(0, len(theta_f)):
            theta_p = theta_f[p]
            step_size = self.sqrt_eps  # * 2.0
            theta_p_step = theta_p + step_size
            d_theta_p = theta_p_step - theta_p  # This doesn't work...
            theta_f_step = np.copy(theta_f)
            theta_f_step[p] = theta_p_step
            step_cost = self.cost_function(theta_f_step, self.X, self.y)
            gradient[p] = ((step_cost - base_cost) / d_theta_p)

            if p == 0:
                print((str(self.iteration) + ' - ' +
                       str(theta_p) + ' - ' + str(theta_p_step) + ' - ' +
                       str(base_cost) + ' - ' +
                       str(step_cost) + ' - ' +
                       str(gradient[p])))

        self.grad = gradient
        return self.grad

    def calc_d_theta(self, step, initial):
        return step - initial


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
