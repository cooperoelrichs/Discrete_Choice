from sklearn import datasets

from logit_estimator.logit_estimation_runner import LogitEstimationRunner

# from logit_estimator import MultinomialLogitEstimator
import time

def print_run_results(model_name, coefs, cost, run_time):
    print('%s results' % model_name)
    print(' - coefficients:')
    print(coefs)
    print(' - cost: %.6f' % cost)
    print(' - run time: %.6f' % run_time)

X, y = datasets.make_classification(1000, 2, 2, 0, 0, 4, 1)
C = 0.01

scaler = LogitEstimationRunner.scaler(X)
X_scaled = scaler.transform(X)

start = time.clock()
sk_mnl = LogitEstimationRunner.estimate_scikit_learn_model(X_scaled, y, C)
sk_mnl_time = time.clock() - start

start = time.clock()
my_mnl = LogitEstimationRunner.estimate_multinomial_model(X_scaled, y, C)
my_mnl_time = time.clock() - start

LogitEstimationRunner.print_run_results('SKL', sk_mnl.thetas, sk_mnl.cost, sk_mnl_time)
LogitEstimationRunner.print_run_results('MNL', my_mnl.thetas, my_mnl.cost, my_mnl_time)

