import numpy as np
import theano
from theano_ml_estimator.mixed_logit_estimator import MixedLogitEstimator
from nl_data_loader.nl_data_loader import NLDataLoader
import time
from theano_ml_estimator.utility_functions import UtilityFunctions, parameter_map, parameter_names


num_draws = 1000

uf = V.parameters.draws

mle = MixedLogitEstimator(X, y, input_parameters, uf, weights, num_alternatives, num_draws, 'float64', 'int64')
initial_cost, initial_error, _ = mle.results(input_parameters)

print(1 - initial_error)

start_time = time.clock()
cost, error, predictions, output_parameters = mle.estimate()
end_time = time.clock()

final_grad = mle.gradient(output_parameters)


def print_params(values, name_map, names):
    for name in names:
        print('%s: %.2f' % (name, values[name_map[name]]))

print_params(output_parameters, parameter_map, parameter_names)
print('Gradient is: ' + str(final_grad))
print('Estimate time: %.2f' % (end_time - start_time))
print('Initial Cost is: %.2f' % initial_cost)
print('Initial Accuracy is: %.2f' % (1 - initial_error))
print('Cost is: %.2f' % cost)
print('Accuracy is: %.2f' % (1 - error))
