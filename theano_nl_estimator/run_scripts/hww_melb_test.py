import numpy as np
from theano_nl_estimator.nested_logit_estimator import NestedLogitEstimator
from nl_data_loader.nl_data_loader import NLDataLoader


# choice, experiment_id, weight, p_zone, a_zone, outward_period, return_period, purpose,
# CarOwnershipConstant23, WAWE_Cost_Outward, KAWE_Cost_Outward, Bicycle_Cost_Outward,
# Walk_Cost_Return, Car_Cost_Return, Car_Cost_Outward, WAPE_Cost_Return, Bicycle_Cost_Return,
# a_CBDNonCore, WAWE_Cost_Return, Walk_Cost_Outward, a_CBDCore, CarOwnershipConstant0,
# PAWE_Cost_Outward, a_OuterFrame, WAKE_Cost_Return, a_CBDFrame, Bicycle_av, Car_av,
# PT_Kiss_Access_av, PT_Park_Access_av, PT_Walk_Access_av, Walk_av

cost_columns = [
    'Bicycle_Cost_Outward', 'Bicycle_Cost_Return',
    'Car_Cost_Outward', 'Car_Cost_Return',
    'WAWE_Cost_Outward', 'WAWE_Cost_Return',
    'PAWE_Cost_Outward', 'WAPE_Cost_Return',
    'KAWE_Cost_Outward', 'WAKE_Cost_Return',
    'Walk_Cost_Outward', 'Walk_Cost_Return',
]

dl = NLDataLoader('../../data/HWW_Melbourne.dat', '\t',
                  cost_columns,
                  'choice')
dl.data = dl.data[dl.get('Bicycle_av') != 0]
dl.data = dl.data[dl.get('Car_av') != 0]
dl.data = dl.data[dl.get('PT_Walk_Access_av') != 0]
dl.data = dl.data[dl.get('PT_Park_Access_av') != 0]
dl.data = dl.data[dl.get('PT_Kiss_Access_av') != 0]
dl.data = dl.data[dl.get('Walk_av') != 0]
# dl.data = dl.data[dl.get('SP') != 0]
# dl.data = dl.data[(dl.get('PURPOSE') == 1) | (dl.get('PURPOSE') == 3)]
dl.print_data_info()
weights = dl.get('weight')
X, y = dl.get_X_and_y()
y -= 1
y = y.astype('int64')
X /= 1000  # scale the costs and travel times

alternatives = np.array([0, 1, 2, 3, 4, 5], dtype='int64')
nests = np.array([0, 1, 2], dtype='int64')
nest_indices = np.array([0, 1, 2, 2, 2, 0], dtype='int64')

# parameters = (n)
# W = (features, alternatives)
# f(parameters) => W
parameters = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
parameters = np.ones_like(parameters)
# utility_functions = np.array([[0, 0, 0], [1, 0, 0],  # (feature, alternative, parameter)
#                               [2, 1, 1], [3, 1, 1],
#                               [4, 2, 2], [5, 2, 2],
#                               [6, 3, 3], [7, 3, 3],
#                               [8, 4, 4], [9, 4, 4],
#                               [10, 5, 5], [11, 5, 5]])

biases = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
lambdas = np.array([[0, 11], [1, 12], [2, 13]])

W_input = np.zeros((X.shape[1], alternatives.shape[0]), dtype='float64')  # rand
b_input = np.zeros_like(alternatives, dtype='float64')
l_input = np.ones_like(nests, dtype='float64')


import theano
import theano.tensor as T
x_i = T.vector('x_i', dtype='float64')
params = T.vector('params', dtype='float64')
v_0 = theano.function(inputs=[x_i, params], outputs=[T.dot(x_i[[0, 1]], params[[0, 0]])], name='v_0')
v_1 = theano.function(inputs=[x_i, params], outputs=[T.dot(x_i[[2, 3]], params[[1, 1]]) + params[6]], name='v_1')
v_2 = T.dot(x_i[[4, 5]], params[[2, 2]]) + params[7]
v_3 = T.dot(x_i[[6, 7]], params[[3, 3]]) + params[8]
v_4 = T.dot(x_i[[8, 9]], params[[4, 4]]) + params[9]
v_5 = T.dot(x_i[[10, 11]], params[[5, 5]]) + params[10]

nle = NestedLogitEstimator(X, y, W_input, b_input, l_input, nests, nest_indices, alternatives,
                           parameters, utility_functions, biases, lambdas)
cost, error, _ = nle.results(parameters)
print(error)
print(cost)
#
# cost, error, _, W, b, lambdas = nle.estimate()
# print(error)
# print(cost)
# print(b)
# print(W)
# print(lambdas)
#
# print('Accuracy is: %.2f' % ((1 - error) * 100))
