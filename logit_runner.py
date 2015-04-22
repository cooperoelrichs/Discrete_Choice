# Script for running the logit_estimator
#
# Python 3.4 64 bit with SciPy

from sklearn import datasets
from logit_estimator import LogitEstimator
from logit_estimator import LogisticRegressionEstimator
import numpy
import time
# import pandas

project_location = 'D:\Cooper\Python\Discrete_Choice'
data_csv_file = project_location + '\\' + 'data.csv'

X, y = datasets.make_classification(10000, 2, 2, 0, 0, 2)
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

print('scikit-learn results')
print(' - coefficients: ' + str(lr_scikit.coef_[0]))
print(' - cost: ' + str(lr_scikit_cost))
print(' - run time: ' + str(scikit_time))
print('my results')
print(' - coefficients: ' + str(lr_new.theta))
print(' - cost: ' + str(lr_new.cost))
print(' - run time: ' + str(new_time))
print('my alt results')
print(' - coefficients: ' + str(lr_new_alt.theta))
print(' - cost: ' + str(lr_new_alt.cost))
print(' - run time: ' + str(alt_time))
