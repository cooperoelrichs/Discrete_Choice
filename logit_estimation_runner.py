from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from logit_estimator import ModelResults
import numpy as np
from logit_estimator import NestedLogitEstimator, LogisticRegressionEstimator,\
    AltLogisticRegressionEstimator, MultinomialLogitEstimator


class LogitEstimationRunner:
    """This class just prepares and runs actual model estimation classes"""

    @staticmethod
    def scaler(x):
        return preprocessing.StandardScaler().fit(x)

    @staticmethod
    def print_run_results(model_name, coefficients, cost, run_time, lambdas='not_given'):
        print('%s results' % model_name)
        print(' - coefficients:')
        print(coefficients)
        print(' - lambdas: ' + str(lambdas))
        print(' - cost: %.6f' % cost)
        print(' - run time: %.6f' % run_time)

    def print_data_statistics(X, y):
        print('Sample size: %i' % len(X))
        LogitEstimationRunner.print_alternative_availability(X, y, 0)
        LogitEstimationRunner.print_alternative_availability(X, y, 1)
        LogitEstimationRunner.print_alternative_availability(X, y, 2)

    @staticmethod
    def print_alternative_availability(X, y, i):
        # print('Alt. %i available %i' % (i, X[...]))
        print('Alt. %i chosen %i' % (i, len(y[y == i])))

    @staticmethod
    def estimate_scikit_learn_model(x, y, c):
        """Estimate a scikit-learn multinomial logit model"""
        lr_r = LogisticRegression(penalty='l2', dual=False, tol=0.0000001,
                                  C=c, fit_intercept=False,
                                  class_weight='auto',
                                  multi_class='multinomial',
                                  solver='lbfgs')

        # Add a ones column to x rather than fitting the intercept
        x_extended = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        lr_r.fit(x_extended, y)

        my_lr_temp = MultinomialLogitEstimator(x, y, c)
        return ModelResults(cost=my_lr_temp.cost_function(np.ravel(lr_r.coef_)),
                            thetas=lr_r.coef_,
                            lambdas=[],
                            iteration=None)

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
    def estimate_multinomial_model(x, y, c, initial_parameters, parameter_indices, fixed_parameters, variable_indices):
        lr = MultinomialLogitEstimator(x, y, c,
                                       initial_parameters=initial_parameters,
                                       parameter_indices=parameter_indices,
                                       fixed_parameters=fixed_parameters,
                                       variable_indices=variable_indices)
        results = lr.estimate()
        # lr.cost_function(lr.theta, lr.X, lr.y)
        # lr.gradient_function(lr.theta, lr.X, lr.y)
        return results

    @staticmethod
    def estimate_nested_model(x, y, c, alts, av, weights, initial_parameters, fixed_parameters, utility_functions,
                              lambda_params):
        lr_nl = NestedLogitEstimator(x, y, c, alts=alts,
                                     av=av,
                                     weights=weights,
                                     initial_parameters=initial_parameters,
                                     fixed_parameters=fixed_parameters,
<<<<<<< HEAD
                                     utility_functions=utility_functions,
                                     lambda_params=lambda_params)

        nl_parameters = lr_nl.initial_parameters
        init_nl_cost = lr_nl.cost_function(nl_parameters)
        # lr_mnl = MultinomialLogitEstimator(x, y, c,
        #                                    initial_parameters=initial_parameters,
        #                                    parameter_indices=parameter_indices,
        #                                    fixed_parameters=fixed_parameters,
        #                                    variable_indices=variable_indices)

        nl_results = lr_nl.estimate()

        # initial_nl_parameters = lr_nl.initial_parameters
        # initial_nl_cost = lr_nl.cost_function(initial_nl_parameters)
        # initial_mnl_cost = lr_mnl.cost_function(initial_nl_parameters[:-2])
        #
        # print('initial MNL results - cost: %.6f' % initial_mnl_cost)
        # print('initial NL results - cost: %.6f' % initial_nl_cost)
        #
        # final_nl_parameters = nl_results.parameters
        # final_nl_cost = lr_nl.cost_function(final_nl_parameters)
        # final_mnl_cost = lr_mnl.cost_function(final_nl_parameters[:-2])
        #
        # print('final MNL results - cost: %.6f' % final_nl_cost)
        # print('final NL results - cost: %.6f' % final_mnl_cost)
        #
        # exit()

        return nl_results

    def calculate_nl_cost(x, y, c, alts, av, weights, initial_parameters, fixed_parameters, utility_functions,
                          lambda_params):
        lr_nl = NestedLogitEstimator(x, y, c, alts=alts,
                                     av=av,
                                     weights=weights,
                                     initial_parameters=initial_parameters,
                                     fixed_parameters=fixed_parameters,
                                     utility_functions=utility_functions,
                                     lambda_params=lambda_params)

        nl_parameters = lr_nl.initial_parameters
        init_nl_cost = lr_nl.cost_function(nl_parameters)

        # lr_mnl = MultinomialLogitEstimator(x, y, 999999999)
        # init_mnl_cost = lr_mnl.cost_function(nl_parameters[:-2])

        # print('initial MNL results - cost: %.6f' % init_mnl_cost)
        return init_nl_cost

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
