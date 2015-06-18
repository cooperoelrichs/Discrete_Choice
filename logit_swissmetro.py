# from sklearn import datasets
from logit_estimator import LogitEstimator
import time
from numpy import genfromtxt
from numpy import unique
import numpy as np


def print_run_results(model_name, coefs, cost, run_time, lambdas='not_given'):
    print('%s results' % model_name)
    print(' - coefficients:')
    print(coefs)
    print(' - lambdas: ' + str(lambdas))
    print(' - cost: %.6f' % cost)
    print(' - run time: %.6f' % run_time)

# X, y = datasets.make_classification(n_samples=1000,
#                                     n_features=10,
#                                     n_informative=8,
#                                     n_redundant=0,
#                                     n_repeated=0,
#                                     n_classes=6,
#                                     n_clusters_per_class=2)

file_name = 'swissmetro.dat'
data = genfromtxt(file_name, delimiter='\t', skip_header=1)
headers = np.array(open(file_name, 'r').readline().rstrip().split('\t'))
data = data[data[:, -1] != 0]  # choice != 0
data = data[data[:, 15] != 0]  # car_av
data = data[data[:, 16] != 0]  # train_av
data = data[data[:, 2] != 0]  # sp

columns = [18, 19, 21, 22, 25, 26]
y = data[:, -1] - 1
X = data[:, columns]
X /= 100  # scale the costs and travel times
#          [4, 5, 6, 7, 8, 9, 10, 11, 12,
#           15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]

print(headers[columns])
print(unique(y))
print(X[:4])

C = 0.01
alts = [[0, 2], [1]]

scaler = LogitEstimator.scaler(X)
X_scaled = scaler.transform(X)

start = time.clock()
my_nl = LogitEstimator.estimate_nested_model(X_scaled, y, C, alts)
my_nl_time = time.clock() - start

start = time.clock()
my_mnl = LogitEstimator.estimate_multinomial_model(X_scaled, y, C)
my_mnl_time = time.clock() - start

print_run_results('MNL', my_mnl.theta, my_mnl.cost, my_mnl_time)
print_run_results('NL', my_nl.theta, my_nl.cost, my_nl_time, my_nl.lambdas)