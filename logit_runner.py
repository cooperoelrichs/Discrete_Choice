# Script for running the logit_estimator
#
# Python 3.4 64 bit with SciPy

from sklearn import datasets
from logit_estimator import LogitEstimator
from logit_estimator import LogisticRegressionEstimator
import numpy
import time
# import pandas


def print_run_results(model_name, coefs, cost, run_time):
    print('%s results' % model_name)
    print(' - coefficients: %s' % coefs)
    print(' - cost: %.10f' % cost)
    print(' - run time: %.10f' % run_time)

project_location = 'D:\Cooper\Python\Discrete_Choice'
data_csv_file = project_location + '\\' + 'data.csv'

X, y = datasets.make_classification(1000, 2, 2, 0, 0, 2)
C = 0.01
# print(X)
# print(y)

scaler = LogitEstimator.scaler(X)
X_scaled = scaler.transform(X)

start = time.clock()
lr_scikit = LogitEstimator.estimate_scikit_learn_model(X_scaled, y, C)
scikit_time = time.clock() - start

X_mod = numpy.append(numpy.ones((X_scaled.shape[0], 1)), X_scaled, axis=1)
lr_temp = LogisticRegressionEstimator(X_mod, y, C)
lr_scikit_cost = lr_temp.cost_function(lr_scikit.coef_[0], X_mod, y)

start = time.clock()
lr_new = LogitEstimator.estimate_home_made_model(X_scaled, y, C)
new_time = time.clock() - start

start = time.clock()
lr_new_alt = LogitEstimator.estimate_home_made_model_alt(X_scaled, y, C)
alt_time = time.clock() - start


print_run_results('scikit-learn', lr_scikit.coef_[0],
                  lr_scikit_cost, scikit_time)
print_run_results('my', lr_new.theta,
                  lr_new.cost, new_time)
print_run_results('my alt', lr_new_alt.theta,
                  lr_new_alt.cost, alt_time)
