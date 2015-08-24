# from sklearn import datasets
from logit_estimation_runner import LogitEstimationRunner
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

# 1  11    Bicycle_Cost_Outward
# 2  16    Bicycle_Cost_Return
# 3  14    Car_Cost_Outward
# 4  13    Car_Cost_Return
# 5  9     WAWE_Cost_Outward
# 6  18    WAWE_Cost_Return
# 7  22    PAWE_Cost_Outward
# 8  15    WAPE_Cost_Return
# 9  10    KAWE_Cost_Outward
# 10 24    WAKE_Cost_Return
# 11 19    Walk_Cost_Outward
# 12 12    Walk_Cost_Return
# 13 21    CarOwnershipConstant0
# 14 8     CarOwnershipConstant23
# 15 20    a_CBDCore
# 16 17    a_CBDNonCore
# 17 25    a_CBDFrame
# 18 23    a_OuterFrame

def pt_walk_access_cost_function(x_i, params):
    return (params['constant_pt_walk_access'] +
            params['p_pt_cost'] * x_i[5] +
            params['p_pt_cost'] * x_i[6])
def bicycle_cost_function(x_i, params):
    return (params['constant_bicycle'] +
            params['p_bicycle_cost'] * x_i[1] +
            params['p_bicycle_cost'] * x_i[2])
def walk_cost_function(x_i, params):
    return (params['p_walk_cost'] * x_i[11] +
            params['p_walk_cost'] * x_i[12])
def car_cost_function(x_i, params):
    return (params['constant_car'] +
            params['p_car_cost'] * x_i[3] +
            params['p_car_cost'] * x_i[4] +
            params['p_car_ownership_0'] * x_i[13] +
            params['p_car_ownership_23'] * x_i[14] +
            params['p_car_cbdcore_destination'] * x_i[15] +
            params['p_car_cbdnoncore_destination'] * x_i[16] +
            params['p_car_cbdframe_destination'] * x_i[17] +
            params['p_car_outerframe_destination'] * x_i[18])
def pt_park_access_cost_function(x_i, params):
    return (params['constant_pt_park_access'] +
            params['p_pt_cost'] * x_i[7] +
            params['p_pt_cost'] * x_i[8] +
            params['p_park_access_car_ownership_0'] * x_i[13] +
            params['p_park_access_car_ownership_23'] * x_i[14])
def pt_kiss_access_cost_function(x_i, params):
    return (params['constant_pt_kiss_access'] +
            params['p_pt_cost'] * x_i[9] +
            params['p_pt_cost'] * x_i[10] +
            params['p_kiss_access_car_ownership_0'] * x_i[13] +
            params['p_kiss_access_car_ownership_23'] * x_i[14])

# utility_functions = [u1, u2, u3]

utility_functions = [
    bicycle_cost_function,
    car_cost_function,
    pt_park_access_cost_function,
    pt_kiss_access_cost_function,
    pt_walk_access_cost_function,
    walk_cost_function,
]

parameters = {
    'p_walk_cost': -0.343 / 100,
    'p_kiss_access_car_ownership_0': -0.932,
    'p_park_access_car_ownership_0': -0.81,
    'p_car_ownership_23': 0.516,
    'p_pt_cost': -0.129 / 100,
    'p_car_cbdnoncore_destination': -2.19,
    'p_car_cbdcore_destination': -2.57,
    'constant_car': 0.0637,
    'constant_pt_walk_access': 1.3,
    'p_park_access_car_ownership_23': 0.878,
    'p_bicycle_cost': -0.243 / 100,
    'constant_bicycle': -3.41,
    'constant_pt_park_access': -0.961,
    'constant_pt_kiss_access': -1.62,
    'p_car_cbdframe_destination': -1.38,
    'p_kiss_access_car_ownership_23': 0.646,
    'p_car_outerframe_destination': -1.28,
    'p_car_cost': -0.198 / 100,
    'p_car_ownership_0': -1.4,
    20: 1,
}

# parameters = {
#     # Melbourne MNL parameters
#     'p_walk_cost': -0.00309,
#     'p_kiss_access_car_ownership_0': -0.734,
#     'p_park_access_car_ownership_0': -31.3,
#     'p_car_ownership_23': 1.69,
#     'p_pt_cost': -0.00164,
#     'p_car_cbdnoncore_destination': -1.56,
#     'p_car_cbdcore_destination': -2.36,
#     'constant_car': -1.72,
#     'constant_pt_walk_access': -1.45,
#     'p_park_access_car_ownership_23': 1.35,
#     'p_bicycle_cost': -0.00291,
#     'constant_bicycle': -3.45,
#     'constant_pt_park_access': -3.56,
#     'constant_pt_kiss_access': -4.1,
#     'p_car_cbdframe_destination': -1.09,
#     'p_kiss_access_car_ownership_23': 0.482,
#     'p_car_outerframe_destination': -0.563,
#     'p_car_cost': -0.00273,
#     'p_car_ownership_0': -3.04,
#     20: 1
# }

data_set_file_name = 'HWW_Queensland.dat'
# data = np.genfromtxt(data_set_file_name, delimiter="\t", names=True)  # skip_header=1)
data = np.genfromtxt(data_set_file_name, delimiter="\t", skip_header=1)
headers = np.array(open(data_set_file_name, 'r').readline().rstrip().split('\t'))

# data = data[data['exclude'] != 0]

columns = [11, 16, 14, 13, 9, 18, 22, 15, 10, 24, 19, 12, 21, 8, 20, 17, 25, 23]
av_columns = [-6, -5, -4, -3, -2, -1]
y = data[:, 0] - 1
X = data[:, columns]
av = data[:, av_columns]
weights = data[:, 2]
# X /= 100  # scale the costs and travel times

print(headers[columns])
print(headers[av_columns])
print(unique(y))

C = 0.01
alts = [[0, 1, 2, 3, 4, 5]]
lambda_params = [20]

# scaler = LogitEstimationRunner.scaler(X)
# X_scaled = scaler.transform(X)

fixed_parameters = set([])  # Set of parameter numbers

init_nl_cost = LogitEstimationRunner.calculate_nl_cost(X, y, C, alts, av, weights, parameters,
                                                       fixed_parameters, utility_functions, lambda_params)

print('initial NL results  - cost: %.6f' % init_nl_cost)
# print_run_results('NL', my_nl.thetas, my_nl.cost, my_nl_time, my_nl.lambdas)
