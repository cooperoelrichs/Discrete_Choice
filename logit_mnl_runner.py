from sklearn import datasets
from logit_estimator import LogitEstimator
import time


def print_run_results(model_name, coefs, cost, run_time):
    print('%s results' % model_name)
    print(' - coefficients: %s' % coefs)
    print(' - cost: %.10f' % cost)
    print(' - run time: %.10f' % run_time)

X, y = datasets.make_classification(1000, 2, 2, 0, 0, 4, 1)
C = 0.01

scaler = LogitEstimator.scaler(X)
X_scaled = scaler.transform(X)

start = time.clock()
mnl = LogitEstimator.estimate_multinomial_model(X_scaled, y, C)
alt_time = time.clock() - start

print_run_results('MNL', mnl.theta, mnl.cost, alt_time)
