import numpy as np
from theano_nl_estimator.nested_logit_estimator import NestedLogitEstimator
from nl_data_loader.nl_data_loader import NLDataLoader


# choice, experiment_id, weight, p_zone, a_zone, outward_period, return_period, purpose,
# CarOwnershipConstant23, WAWE_Cost_Outward, KAWE_Cost_Outward, Bicycle_Cost_Outward,
# Walk_Cost_Return, Car_Cost_Return, Car_Cost_Outward, WAPE_Cost_Return, Bicycle_Cost_Return,
# a_CBDNonCore, WAWE_Cost_Return, Walk_Cost_Outward, a_CBDCore, CarOwnershipConstant0,
# PAWE_Cost_Outward, a_OuterFrame, WAKE_Cost_Return, a_CBDFrame, Bicycle_av, Car_av,
# PT_Kiss_Access_av, PT_Park_Access_av, PT_Walk_Access_av, Walk_av

dl = NLDataLoader('../../data/HWW_Melbourne.dat', '\t',
                  ['Bicycle_Cost_Outward', 'Bicycle_Cost_Return',
                   'Car_Cost_Outward', 'Car_Cost_Return',
                   'WAWE_Cost_Outward', 'WAWE_Cost_Return',
                   'PAWE_Cost_Outward', 'WAPE_Cost_Return',
                   'KAWE_Cost_Outward', 'WAKE_Cost_Return',
                   'Walk_Cost_Outward', 'Walk_Cost_Return',
                   ],
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

alternatives = 6
nests = np.array([0, 1, 2], dtype='int32')
nest_indices = np.array([0, 1, 2, 2, 2, 0])
lambdas = np.array([1, 1, 1])

W_input = np.zeros((X.shape[1], alternatives))  # rand
b_input = np.zeros(alternatives)

nle = NestedLogitEstimator(X, y, W_input, b_input, lambdas, nests, nest_indices, alternatives)
cost, error, _ = nle.results(nle.initial_W, nle.initial_b, nle.initial_lambdas)
print(error)
print(cost)

cost, error, _, W, b, lambdas = nle.estimate()
print(error)
print(cost)
print(b)
print(W)
print(lambdas)

print('Accuracy is: %.2f' % ((1 - error) * 100))
