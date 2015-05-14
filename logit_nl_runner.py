from sklearn import datasets
from logit_estimator import LogitEstimator
import time


def print_run_results(model_name, coefs, cost, run_time):
    print('%s results' % model_name)
    print(' - coefficients:')
    print(coefs)
    print(' - cost: %.6f' % cost)
    print(' - run time: %.6f' % run_time)

X, y = datasets.make_classification(1000, 2, 2, 0, 0, 4, 1)
C = 0.01

scaler = LogitEstimator.scaler(X)
X_scaled = scaler.transform(X)

start = time.clock()
my_mnl = LogitEstimator.estimate_nested_model(X_scaled, y, C)
my_mnl_time = time.clock() - start

print_run_results('my MNL', my_mnl.theta, my_mnl.cost, my_mnl_time)
