from numpy import unique
import numpy as np

from logit_estimator.logit_estimation_runner import LogitEstimationRunner


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

# parameters = np.array([
#     -0.343 / 100,
#     -0.932,
#     -0.81,
#     0.516,
#     -0.129 / 100,
#     -2.19,
#     -2.57,
#     0.0637,
#     1.3,
#     0.878,
#     -0.243 / 100,
#     -3.41,
#     -0.961,
#     -1.62,
#     -1.38,
#     0.646,
#     -1.28,
#     -0.198 / 100,
#     -1.4,
#     1.0,
#     1.0,
# ])

NONMECH  =  1.00939275
PTNEST   =  0.60172876

PW_COST  = -0.00315173
C_B      = -3.18935301
PB_COST  = -0.00231400
C_C      =  0.28167711
PC_COST  = -0.00184692
PC_C0    = -1.32758479
PC_C23   =  0.45070403
PC_ACORE = -2.58974780
PC_ANCOR = -2.15965129
PC_AFR   = -1.37576479
PC_AOFR  = -1.25140408
C_WA     =  2.63962956
PPT_COST = -0.00210587
C_PA     = -0.13104479
PPA_C0   = -0.90262771
PPA_C23  =  0.94450532
C_KA     = -0.86176905
PKA_C0   = -0.84353184
PKA_C23  =  0.73190917

mu_non_mechanised = 1 / NONMECH
mu_pt = 1 / PTNEST

nonmech_scale = mu_non_mechanised
pt_scale = mu_pt

parameters = np.array([
    C_B * nonmech_scale,
    C_C,
    C_KA * pt_scale,
    C_PA * pt_scale,
    C_WA * pt_scale,
    PB_COST * nonmech_scale / 1000,
    PC_COST / 1000,
    PPT_COST * pt_scale / 1000,
    PW_COST * nonmech_scale / 1000,
    PC_C0,
    PC_C23,
    PKA_C0 * pt_scale,
    PKA_C23 * pt_scale,
    PPA_C0 * pt_scale,
    PPA_C23 * pt_scale,
    PC_ACORE,
    PC_ANCOR,
    PC_AFR,
    PC_AOFR,
    mu_non_mechanised,
    mu_pt,
    1,
])

params_key = {
    'constant_bicycle': 0,
    'constant_car': 1,
    'constant_pt_kiss_access': 2,
    'constant_pt_park_access': 3,
    'constant_pt_walk_access': 4,
    'p_bicycle_cost': 5,
    'p_car_cost': 6,
    'p_pt_cost': 7,
    'p_walk_cost': 8,
    'p_car_ownership_0': 9,
    'p_car_ownership_23': 10,
    'p_kiss_access_car_ownership_0': 11,
    'p_kiss_access_car_ownership_23': 12,
    'p_park_access_car_ownership_0': 13,
    'p_park_access_car_ownership_23': 14,
    'p_car_cbdcore_destination': 15,
    'p_car_cbdnoncore_destination': 16,
    'p_car_cbdframe_destination': 17,
    'p_car_outerframe_destination': 18,
    'mu_non_mechanised': 19,
    'mu_pt': 20,
    'mu_car': 21
}

def pt_walk_access_cost_function(x_i, params):
    return (params[params_key['constant_pt_walk_access']] +
            params[params_key['p_pt_cost']] * x_i[5] +
            params[params_key['p_pt_cost']] * x_i[6])
def bicycle_cost_function(x_i, params):
    return (params[params_key['constant_bicycle']] +
            params[params_key['p_bicycle_cost']] * x_i[1] +
            params[params_key['p_bicycle_cost']] * x_i[2])
def walk_cost_function(x_i, params):
    return (params[params_key['p_walk_cost']] * x_i[11] +
            params[params_key['p_walk_cost']] * x_i[12])
def car_cost_function(x_i, params):
    return (params[params_key['constant_car']] +
            params[params_key['p_car_cost']] * x_i[3] +
            params[params_key['p_car_cost']] * x_i[4] +
            params[params_key['p_car_ownership_0']] * x_i[13] +
            params[params_key['p_car_ownership_23']] * x_i[14] +
            params[params_key['p_car_cbdcore_destination']] * x_i[15] +
            params[params_key['p_car_cbdnoncore_destination']] * x_i[16] +
            params[params_key['p_car_cbdframe_destination']] * x_i[17] +
            params[params_key['p_car_outerframe_destination']] * x_i[18])
def pt_park_access_cost_function(x_i, params):
    return (params[params_key['constant_pt_park_access']] +
            params[params_key['p_pt_cost']] * x_i[7] +
            params[params_key['p_pt_cost']] * x_i[8] +
            params[params_key['p_park_access_car_ownership_0']] * x_i[13] +
            params[params_key['p_park_access_car_ownership_23']] * x_i[14])
def pt_kiss_access_cost_function(x_i, params):
    return (params[params_key['constant_pt_kiss_access']] +
            params[params_key['p_pt_cost']] * x_i[9] +
            params[params_key['p_pt_cost']] * x_i[10] +
            params[params_key['p_kiss_access_car_ownership_0']] * x_i[13] +
            params[params_key['p_kiss_access_car_ownership_23']] * x_i[14])

# utility_functions = [u1, u2, u3]

utility_functions = [
    bicycle_cost_function,
    car_cost_function,
    pt_kiss_access_cost_function,
    pt_park_access_cost_function,
    pt_walk_access_cost_function,
    walk_cost_function,
]

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
alts = [[0, 5], [1], [2, 3, 4]]
lambda_params = [19, 20, 21]

# scaler = LogitEstimationRunner.scaler(X)
# X_scaled = scaler.transform(X)

fixed_parameters = set([])  # Set of parameter numbers

init_nl_cost = LogitEstimationRunner.calculate_nl_cost(X, y, C, alts, av, weights, parameters,
                                                       fixed_parameters, utility_functions, lambda_params)

# start = time.clock()
# my_nl = LogitEstimationRunner.estimate_nested_model(X_scaled, y, C, alts, av, weights, parameters,
#                                                     fixed_parameters, utility_functions, lambda_params)
# my_nl_time = time.clock() - start

print('initial NL results  - cost: %.6f' % init_nl_cost)
# print_run_results('NL', my_nl.thetas, my_nl.cost, my_nl_time, my_nl.lambdas)
