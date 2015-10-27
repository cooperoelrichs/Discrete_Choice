import numpy as np
import theano
from theano_ml_estimator.mixed_logit_estimator import MixedLogitEstimator
from nl_data_loader.nl_data_loader import NLDataLoader
import time

float_dtype = theano.config.floatX
int_dtype = 'int64'
np.seterr(all='raise')

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

dl = NLDataLoader('../../data/HWW_Melbourne.dat', '\t', cost_columns, 'choice', float_dtype, int_dtype)
dl.data = dl.data[dl.get('Bicycle_av') != 0]
dl.data = dl.data[dl.get('Car_av') != 0]
dl.data = dl.data[dl.get('PT_Walk_Access_av') != 0]
dl.data = dl.data[dl.get('PT_Park_Access_av') != 0]
dl.data = dl.data[dl.get('PT_Kiss_Access_av') != 0]
dl.data = dl.data[dl.get('Walk_av') != 0]
# dl.data = dl.data[:1000]
# dl.data = dl.data[dl.get('SP') != 0]
# dl.data = dl.data[(dl.get('PURPOSE') == 1) | (dl.get('PURPOSE') == 3)]
dl.print_data_info()
weights = dl.get('weight')
X, y = dl.get_X_and_y()
num_alternatives = 6
X /= 100  # scale the costs and travel times

input_parameters = np.zeros(11 + 9)  # np.random.randn(14).astype(float_dtype)

parameter_names = [
    'b_bicycle', 'b_car', 'b_pt1', 'b_pt2', 'b_pt3',
    'cost_bicycle', 'cost_car', 'cost_pt1', 'cost_pt2', 'cost_pt3', 'cost_walk',
    'random_cost_pt', 'random_cost_walk', 'random_cost_car', 'random_cost_bike',
    'error_pt', 'error_car', 'error_non_mech', 'error_bike', 'error_pt_kiss_access',
]

biases = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=int_dtype)
cost_params = np.array(
    [[0, 0, 5], [1, 0, 5],  # (feature, alternative, parameter)
     [2, 1, 6], [3, 1, 6],
     [4, 2, 7], [5, 2, 7],
     [6, 3, 8], [7, 3, 8],
     [8, 4, 9], [9, 4, 9],
     [10, 5, 10], [11, 5, 10]],
    dtype=int_dtype
)

random_error_params = np.array(
    [[2, 15],
     [3, 15],
     [4, 15],
     [1, 16],
     [0, 17],
     [5, 17],
     [0, 18],
     [4, 19],
     ], dtype=int_dtype)
random_cost_params = np.array(
    [[4, 2, 11], [5, 2, 11],
     [6, 3, 11], [7, 3, 11],
     [8, 4, 11], [9, 4, 11],
     [10, 5, 12], [11, 5, 12],
     [2, 1, 13], [3, 1, 13],
     [0, 0, 14], [1, 0, 14],
     ],
    dtype=int_dtype
)

b_input = np.zeros(num_alternatives, dtype=float_dtype)
W_input = np.zeros((X.shape[1], num_alternatives), dtype=float_dtype)  # rand
W_rand_input = np.zeros((X.shape[1], num_alternatives), dtype=float_dtype)
error_input = np.zeros(num_alternatives, dtype=float_dtype)

mle = MixedLogitEstimator(X, y, W_input, b_input, W_rand_input, error_input,
                          input_parameters, cost_params, biases, random_cost_params, random_error_params, weights,
                          num_alternatives)
initial_cost, initial_error, _ = mle.results(input_parameters)

start_time = time.clock()
cost, error, predictions, output_parameters = mle.estimate()
end_time = time.clock()

final_grad = mle.gradient(output_parameters)
# W, b, l = nle.extract_parameters(output_parameters)
# print(b)
# print(W)
# print(l)


def print_params(params_, names_):
    for i, x in enumerate(params_):
        print('%s: %.2f' % (names_[i], x))

print_params(output_parameters, parameter_names)
print('Gradient is: ' + str(final_grad))
print('Estimate time: %.2f' % (end_time - start_time))
print('Initial Cost is: %.2f' % initial_cost)
print('Initial Accuracy is: %.2f' % (1 - initial_error))
print('Cost is: %.2f' % cost)
print('Accuracy is: %.2f' % (1 - error))
