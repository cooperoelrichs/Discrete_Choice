from sklearn import datasets
from logit_estimator import LogitEstimator
from logit_estimator import MultiNomialLogitEstimator
import time
import numpy


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
sk_mnl = LogitEstimator.estimate_scikit_learn_model(X_scaled, y, C)
sk_mnl_time = time.clock() - start

mnl_temp = MultiNomialLogitEstimator(X, y, C)
sk_mnl_cost = mnl_temp.cost_function(numpy.ravel(sk_mnl.coef_),
                                     mnl_temp.X, mnl_temp.y)

start = time.clock()
my_mnl = LogitEstimator.estimate_multinomial_model(X_scaled, y, C)
my_mnl_time = time.clock() - start

print_run_results('scikit-learn MNL', sk_mnl.coef_, sk_mnl_cost, sk_mnl_time)
print_run_results('my MNL', my_mnl.theta, my_mnl.cost, my_mnl_time)
