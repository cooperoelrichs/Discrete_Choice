from sklearn import datasets
from logit_estimation_runner import LogisticRegressionEstimator
import time


def print_run_results(model_name, coefs, cost, run_time, lambdas='not_given'):
    print('%s results' % model_name)
    print(' - coefficients:')
    print(coefs)
    print(' - lambdas: %s' % str(lambdas))
    print(' - cost: %.6f' % cost)
    print(' - run time: %.6f' % run_time)

X, y = datasets.make_classification(n_samples=1000,
                                    n_features=10,
                                    n_informative=8,
                                    n_redundant=0,
                                    n_repeated=0,
                                    n_classes=6,
                                    n_clusters_per_class=2)
C = 0.01
alts = [[0, 1, 5], [2], [3, 4]]

scaler = LogisticRegressionEstimator.scaler(X)
X_scaled = scaler.transform(X)

start = time.clock()
my_nl = LogisticRegressionEstimator.estimate_nested_model(X_scaled, y, C, alts)
my_nl_time = time.clock() - start

start = time.clock()
my_mnl = LogisticRegressionEstimator.estimate_multinomial_model(X_scaled, y, C)
my_mnl_time = time.clock() - start

print_run_results('MNL', my_mnl.theta, my_mnl.cost, my_mnl_time)
print_run_results('NL', my_nl.theta, my_nl.cost, my_nl_time, my_nl.lambdas)
