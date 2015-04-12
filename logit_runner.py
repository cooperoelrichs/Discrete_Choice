# Script for running the logit_estimator
#
# Python 3.4 64 bit with SciPy

from sklearn import datasets
from logit_estimator import LogitEstimator
from logit_estimator import LogisticRegressionEstimator
import numpy
# import pandas

project_location = 'D:\Cooper\Python\Discrete_Choice'
data_csv_file = project_location + '\\' + 'data.csv'

X, y = datasets.make_classification(10000,  2, 2, 0, 0, 2)
# print(X)
# print(y)

scaler = LogitEstimator.scaler(X)
X_scaled = scaler.transform(X)

lr_scikit = LogitEstimator.estimate_scikit_learn_model(X_scaled, y)
lr_new = LogitEstimator.estimate_home_made_model(X_scaled, y)

X_mod = numpy.append(numpy.ones((X_scaled.shape[0], 1)), X_scaled, axis=1)
lr_temp = LogisticRegressionEstimator(X_mod, y, 10)
lr_scikit_cost = lr_temp.cost_function(lr_scikit.coef_[0], X_mod, y)

print('scikit-learn results')
print(' - coefficients: ' + str(lr_scikit.coef_[0]))
print(' - cost: ' + str(lr_scikit_cost))
print('my results')
print(' - coefficients: ' + str(lr_new.theta))
print(' - cost: ' + str(lr_new.cost))
