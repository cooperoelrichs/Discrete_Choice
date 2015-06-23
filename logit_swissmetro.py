# from sklearn import datasets
from logit_estimation_runner import LogitEstimationRunner
import time
from numpy import genfromtxt
from numpy import unique
import numpy as np

# X, y = datasets.make_classification(n_samples=1000,
#                                     n_features=10,
#                                     n_informative=8,
#                                     n_redundant=0,
#                                     n_repeated=0,
#                                     n_classes=6,
#                                     n_clusters_per_class=2)

file_name = 'biogeme_files/swissmetro.dat'
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

scaler = LogitEstimationRunner.scaler(X)
X_scaled = scaler.transform(X)

# Experimental utility function specification
# ASC_CAR = Beta('ASC_CAR',-0.167,-10,10,0)
# ASC_TRAIN = Beta('ASC_TRAIN',-0.512,-10,10,0)
# ASC_SM = Beta('ASC_SM',0,-10,10,1)
# B_TIME = Beta('B_TIME',-0.899,-10,10,0)
# B_COST = Beta('B_COST',-0.857,-10,10,0)
# MU = Beta('MU',2.05,1,10,0)

initial_parameters = np.array([
    np.random.rand(),
    np.random.rand(),
    np.random.rand(),
    np.random.rand(),
    np.random.rand(),
    1.0,
    1.0,
])

fixed_parameters = set([6])  # Set of parameter numbers

def u1(x_i, params):
    return np.dot(x_i[[0, 1, 2]], params[[0, 3, 4]])

def u2(x_i, params):
    return np.dot(x_i[[0, 3, 4]], params[[1, 3, 4]])

def u3(x_i, params):
    return np.dot(x_i[[0, 5, 6]], params[[2, 3, 4]])

utility_functions = [u1, u2, u3]

# start = time.clock()
# my_nl = LogitEstimationRunner.estimate_nested_model(X_scaled, y, C, alts, initial_parameters,
#                                                     fixed_parameters, utility_functions)
# my_nl_time = time.clock() - start

start = time.clock()
my_mnl = LogitEstimationRunner.estimate_multinomial_model(X_scaled, y, C,
                                                          initial_parameters, fixed_parameters, utility_functions)
my_mnl_time = time.clock() - start

LogitEstimationRunner.print_run_results('MNL', my_mnl.thetas, my_mnl.cost, my_mnl_time)
# LogitEstimationRunner.print_run_results('NL', my_nl.thetas, my_nl.cost, my_nl_time, my_nl.lambdas)