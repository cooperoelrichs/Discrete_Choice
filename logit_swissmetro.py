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

# TODO: Add availability conditions

data = data[data[:, -1] != 0]  # choice != 0
# data = data[data[:, 15] != 0]  # car_av
# data = data[data[:, 16] != 0]  # train_av
# data = data[data[:, 2] != 0]  # sp
data = data[(data[:, 4] == 1) | (data[:, 4] == 3)]  # purpose == 1 or purpose == 3

columns = [18, 19, 21, 22, 25, 26]
y = data[:, -1] - 1
X = data[:, columns]
X /= 100  # scale the costs and travel times

print(headers[[-1, 15, 16, 2, 4]])
print(headers[columns])
print(unique(y))
print(X[:4])

C = 999999  # 0.01
alts = [[0, 2], [1]]

# scaler = LogitEstimationRunner.scaler(X)
# X_scaled = scaler.transform(X)

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
    # np.random.rand(),
    1,
    1,
])

fixed_parameters = {5}  # Set of parameter numbers

parameter_indices = [[0, 2, 3], [1, 2, 3], [2, 3]]
variable_indices = [[0, 1, 2], [0, 3, 4], [5, 6]]

def u1(x_i, parameters):
    return np.dot(x_i[[0, 1, 2]], parameters[[0, 2, 3]])

def u2(x_i, parameters):
    return np.dot(x_i[[0, 3, 4]], parameters[[1, 2, 3]])

def u3(x_i, parameters):
    return np.dot(x_i[[5, 6]], parameters[[2, 3]])

utility_functions = [u1, u2, u3]

LogitEstimationRunner.print_data_statistics(X, y)


start = time.clock()
my_nl = LogitEstimationRunner.estimate_nested_model(X, y, C, alts,
                                                    initial_parameters,
                                                    fixed_parameters,
                                                    utility_functions,
                                                    parameter_indices, variable_indices)
my_nl_time = time.clock() - start

start = time.clock()
my_mnl = LogitEstimationRunner.estimate_multinomial_model(X, y, C,
                                                          initial_parameters[:-2],
                                                          parameter_indices,
                                                          fixed_parameters,
                                                          variable_indices)
my_mnl_time = time.clock() - start

LogitEstimationRunner.print_run_results('MNL', my_mnl.thetas, my_mnl.cost, my_mnl_time)
LogitEstimationRunner.print_run_results('NL', my_nl.thetas, my_nl.cost, my_nl_time, my_nl.lambdas)
