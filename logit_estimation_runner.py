from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
from logit_estimator import NestedLogitEstimator, LogisticRegressionEstimator,\
    AltLogisticRegressionEstimator, MultinomialLogitEstimator


class LogitEstimationRunner:
    """This class just prepares and runs actual model estimation classes"""

    @staticmethod
    def scaler(x):
        return preprocessing.StandardScaler().fit(x)

    @staticmethod
    def estimate_scikit_learn_model(x, y, c):
        """Estimate a scikit-learn multinomial logit model"""
        x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        # x[0, 0] = 0

        # Add a ones column to x rather than fitting the intercept
        lr_r = LogisticRegression(penalty='l2', dual=False, tol=0.0000001,
                                  C=c, fit_intercept=False,
                                  class_weight='auto',
                                  multi_class='multinomial',
                                  solver='lbfgs')

        lr_r.fit(x, y)
        return lr_r

    @staticmethod
    def estimate_home_made_model(x, y, c):
        lr = LogisticRegressionEstimator(x, y, c, [])
        lr.estimate()
        return lr

    @staticmethod
    def estimate_home_made_model_alt(x, y, c):
        lr = AltLogisticRegressionEstimator(x, y, c, [])
        lr.estimate()
        return lr

    @staticmethod
    def estimate_multinomial_model(x, y, c):
        lr = MultinomialLogitEstimator(x, y, c, [])
        lr.estimate()
        # lr.cost_function(lr.theta, lr.X, lr.y)
        # lr.gradient_function(lr.theta, lr.X, lr.y)
        return lr

    @staticmethod
    def estimate_nested_model(x, y, c, alts):
        lr_nl = NestedLogitEstimator(x, y, c, alts=alts)
        # init_nl_cost = lr_nl.cost_function(lr_nl.theta_f, lr_nl.X, lr_nl.y)
        nl_parameters = lr_nl.get_parameters()
        init_nl_cost = lr_nl.cost_function(nl_parameters)
        # lr_mnl = MultinomialLogitEstimator(x, y, 999999999, [])
        # init_mnl_cost = lr_mnl.cost_function(nl_parameters[:-2])

        # print('initial MNL results - cost: %.6f' % init_mnl_cost)
        print('initial NL results  - cost: %.6f' % init_nl_cost)

        nl_results = lr_nl.estimate()
        return nl_results

    # @staticmethod
    # def _estimate_nested_model(x, y, c):
    #     x = np.array([[1, 2], [0, 0]])
    #     theta = np.array([[2, 0, 2], [-1, 2, 1], [0, 1, 0], [0, 0, 0]])
    #     lambdas = np.array([0.5, 1])
    #     y = np.array([0, 1, 2, 3])
    #
    #     theta_f = np.ravel(theta)
    #     theta_f = np.append(theta_f, lambdas)
    #
    #     lr_nl = NestedLogitEstimator(x, y, c, alts=[[0, 1], [2, 3]])
    #     # x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
    #     lr_nl.cost_function(theta_f)
    #     print('NL results  - cost: %.6f' % lr_nl.cost)
