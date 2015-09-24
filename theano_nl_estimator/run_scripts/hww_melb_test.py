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
utility_functions = np.array([[0, 0, 0], [1, 0, 0],  # (feature, alternative, parameter)
                              [2, 1, 1], [3, 1, 1],
                              [4, 2, 2], [5, 2, 2],
                              [6, 3, 3], [7, 3, 3],
                              [8, 4, 4], [9, 4, 4],
                              [10, 5, 5], [11, 5, 5]])

biases = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
lambdas = np.array([[0, 11], [1, 12], [2, 13]])

W_input = np.zeros((X.shape[1], alternatives.shape[0]), dtype='float64')  # rand
b_input = np.zeros_like(alternatives, dtype='float64')
l_input = np.ones_like(nests, dtype='float64')

# import theano
# import theano.tensor as T
# W_input[[utility_functions[:, 0], utility_functions[:, 1]]] = parameters[utility_functions[:, 2]]
# b_input[biases[:, 0]] = parameters[biases[:, 1]]
# l_input[lambdas[:, 0]] = parameters[lambdas[:, 1]]
# def sum_nest_for_nest(i, nest_indices__, exp_V__):
#     indices = T.eq(nest_indices__, i).nonzero()[0]
#     exp_V_i = exp_V__[:, indices]
#     return exp_V_i.sum(axis=1)
# V = (T.dot(X, W_input) + b_input)
# exp_V = T.exp(V / l_input[nest_indices])
# nest_sums_T, _ = theano.scan(sum_nest_for_nest,
#                              sequences=[nests], non_sequences=[nest_indices, exp_V],
#                              name='sum_exp_utilities_by_nests')
# nest_sums = nest_sums_T.T
# print(X.shape)
# print(W_input.shape)
# print(V.eval().shape)
# print(nest_sums.eval().shape)
# print(W_input.broadcastable)
# print(X.broadcastable)
# exit()

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
