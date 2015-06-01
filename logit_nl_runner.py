from sklearn import datasets
from logit_estimator import LogitEstimator
import time


def print_run_results(model_name, coefs, cost, run_time):
    print('%s results' % model_name)
    print(' - coefficients:')
    print(coefs)
    print(' - cost: %.6f' % cost)
    print(' - run time: %.6f' % run_time)

X, y = datasets.make_classification(n_samples=500,
                                    n_features=10,
                                    n_informative=8,
                                    n_redundant=0,
                                    n_repeated=0,
                                    n_classes=4,
                                    n_clusters_per_class=2)
C = 0.01
alts = [[0, 1], [2, 3]]

scaler = LogitEstimator.scaler(X)
X_scaled = scaler.transform(X)

start = time.clock()
my_nl = LogitEstimator.estimate_nested_model(X_scaled, y, C, alts)
my_nl_time = time.clock() - start

start = time.clock()
my_mnl = LogitEstimator.estimate_multinomial_model(X_scaled, y, C)
my_mnl_time = time.clock() - start

print_run_results('MNL', my_mnl.theta, my_mnl.cost, my_mnl_time)
print_run_results('NL', my_nl.theta, my_nl.cost, my_nl_time)
