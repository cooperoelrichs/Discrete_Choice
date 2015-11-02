import numpy as np
import theano
from theano_ml_estimator.mixed_logit_estimator import MixedLogitEstimator
from nl_data_loader.nl_data_loader import NLDataLoader
import time
from utility_functions import UtilityFunctions

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

utility_functions = UtilityFunctions(float_dtype).fn_list

input_parameters = np.zeros(11 + 9)  # np.random.randn(14).astype(float_dtype)

mle = MixedLogitEstimator(X, y, input_parameters, utility_functions, weights, num_alternatives)
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
