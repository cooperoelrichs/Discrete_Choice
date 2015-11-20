import numpy as np
import theano
import theano.tensor as T
from theano_ml_estimator.mixed_logit_estimator import MixedLogitEstimator
import time

import sklearn.datasets as datasets
from collections import OrderedDict


param_map = OrderedDict([
    ('1-bias', 0), ('1-bias-random', 1), ('1-1', 2), ('1-1-random', 3), ('1-2', 4), ('1-2-random', 5), ('1-3', 6), ('1-3-random', 7),
    ('2-bias', 8), ('2-bias-random', 9), ('2-1', 10), ('2-1-random', 11), ('2-2', 12), ('2-2-random', 13), ('2-3', 14), ('2-3-random', 15)
])


class UF(object):
    def __init__(self):
        pass

    def calculate_V(self, V, X, B, R):
        # V[exps, alts, draws]

        # X [obs x features]
        # B [parameters]
        # R [obs x B_r x draws]

        V = T.set_subtensor(V[:, 0, :], (
            B[param_map['1-bias']] +
            B[param_map['1-bias-random']]*R[:, 0, :] +
            B[param_map['1-1']]*X[:, 0, np.newaxis] +
            B[param_map['1-1-random']]*R[:, 1, :]*X[:, 0, np.newaxis] +
            B[param_map['1-2']]*X[:, 1, np.newaxis] +
            B[param_map['1-2-random']]*R[:, 2, :]*X[:, 1, np.newaxis] +
            B[param_map['1-3']]*X[:, 2, np.newaxis] +
            B[param_map['1-3-random']]*R[:, 3, :]*X[:, 2, np.newaxis]
        ))
        V = T.set_subtensor(V[:, 1, :], (
            B[param_map['2-bias']] +
            B[param_map['2-bias-random']]*R[:, 0, :] +
            B[param_map['2-1']]*X[:, 0, np.newaxis] +
            B[param_map['2-1-random']]*R[:, 1, :]*X[:, 0, np.newaxis] +
            B[param_map['2-2']]*X[:, 1, np.newaxis] +
            B[param_map['2-2-random']]*R[:, 2, :]*X[:, 1, np.newaxis] +
            B[param_map['2-3']]*X[:, 2, np.newaxis] +
            B[param_map['2-3-random']]*R[:, 3, :]*X[:, 2, np.newaxis]
        ))
        return V


num_draws = 2000
num_alternatives = 2
input_parameters = np.zeros(len(param_map), dtype='float64')
uf = UF()

X, y = datasets.make_classification(
    n_samples=5000, n_features=3,
    n_informative=2, n_redundant=0, n_repeated=0,
    n_classes=2, n_clusters_per_class=1,
    random_state=1,
    flip_y=0.4
)

X = (X - X.mean(axis=0)) / X.std(axis=0)
print(X.sum(axis=0))
print(X.mean(axis=0))

random_numbers = np.random.random_sample(X[:, 1].shape)
X[:, 1] = X[:, 1] + (random_numbers - random_numbers.mean())

weights = np.ones_like(y)

mle = MixedLogitEstimator(X, y, input_parameters, uf, weights, num_alternatives, num_draws, False, 'float64', 'int64')
initial_cost, initial_error, _ = mle.results(input_parameters)

start_time = time.clock()
cost, error, predictions, output_parameters = mle.estimate()
end_time = time.clock()

final_grad = mle.gradient(output_parameters)


def print_params(values, name_map):
    for name, i in name_map.items():
        print('%s: %.2f' % (name, values[i]))

print('Generated data test.')
print('1 informative feature with some noise, 1 informative feature with lots of noise, 1 pure noise feature')
print_params(output_parameters, param_map)
print('Gradient is: ' + str(final_grad))
print('Estimate time: %.2f' % (end_time - start_time))
print('Initial Cost is: %.2f' % initial_cost)
print('Initial Accuracy is: %.2f' % (1 - initial_error))
print('Cost is: %.2f' % cost)
print('Accuracy is: %.2f' % (1 - error))
