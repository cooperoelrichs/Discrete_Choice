import numpy as np
import theano
import theano.tensor as T
from theano_ml_estimator.mixed_logit_estimator import MixedLogitEstimator
import time

import sklearn.datasets as datasets


parameter_names = [
    '1-bias', '1-cost', '1-random-cost', '1-redundant', '1-random-redundant',
    '2-bias', '2-cost', '2-random-cost', '2-redundant', '2-random-redundant']
parameter_map = {
    '1-bias': 0, '1-scale': 1, '1-random-scale': 2, '1-redundant': 3, '1-random-redundant': 4,
    '2-bias': 5, '2-scale': 6, '2-random-scale': 7, '2-redundant': 8, '2-random-redundant': 9}

class UF(object):
    def __init__(self):
        pass

    def calculate_V(self, V, X, B, R):
        # V[exps, alts, draws]

        # X [obs x features]
        # B [parameters]
        # R [obs x B_r x draws]

        V = T.set_subtensor(V[:, 0, :], (B[0] +
                                         B[1]*X[:, 0, np.newaxis] +
                                         B[2]*R[:, 0, :]*X[:, 0, np.newaxis] +
                                         B[3]*X[:, 1, np.newaxis] +
                                         B[4]*R[:, 1, :]*X[:, 1, np.newaxis]))
        V = T.set_subtensor(V[:, 1, :], (B[5] +
                                         B[6]*X[:, 0, np.newaxis] +
                                         B[7]*R[:, 0, :]*X[:, 0, np.newaxis] +
                                         B[8]*X[:, 1, np.newaxis] +
                                         B[9]*R[:, 1, :]*X[:, 1, np.newaxis]))
        return V


num_draws = 2000
num_alternatives = 2
input_parameters = np.zeros(10, dtype='float64')
uf = UF()

X, y = datasets.make_classification(
    n_samples=5000, n_features=2,
    n_informative=1, n_redundant=1, n_repeated=0,
    n_classes=2, n_clusters_per_class=1,
    random_state=1
)

weights = np.ones_like(y)

mle = MixedLogitEstimator(X, y, input_parameters, uf, weights, num_alternatives, num_draws, False, 'float64', 'int64')
initial_cost, initial_error, _ = mle.results(input_parameters)

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
