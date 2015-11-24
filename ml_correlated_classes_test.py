import numpy as np
import theano
import theano.tensor as T
from theano_ml_estimator.mixed_logit_estimator import MixedLogitEstimator
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.datasets import make_blobs


param_map = OrderedDict([(name, i) for i, name in enumerate([
    '0,1-bias-random',
    '0,2-bias-random',
    '1,2-bias-random',
    '0,1-0-beta-random',
    '0,2-0-beta-random',
    '1,2-0-beta-random',
    '0,1-1-beta-random',
    '0,2-1-beta-random',
    '1,2-1-beta-random',
    '0-bias', '0-1-beta', '0-2-beta',  # '0-0-random', '0-1-random',
    '1-bias', '1-1-beta', '1-2-beta',  # '1-0-random', '1-1-random',
    '2-bias', '2-1-beta', '2-2-beta',  # '2-0-random', '2-1-random',
])])

draw_map = OrderedDict([(name, i) for i, name in enumerate([
    '0,1-bias-random',
    '0,2-bias-random',
    '1,2-bias-random',
    '0,1-0-beta-random',
    '0,2-0-beta-random',
    '1,2-0-beta-random',
    '0,1-1-beta-random',
    '0,2-1-beta-random',
    '1,2-1-beta-random',
    '0-0-random', '0-1-random',  # '0-3-random',
    '1-0-random', '1-1-random',  # '1-3-random',
    '2-0-random', '2-1-random',  # '2-3-random',
])])


class UF(object):
    def __init__(self):
        pass

    def calculate_V(self, V, X, B, R):
        # V[exps, alts, draws]

        # X [obs x features]
        # B [parameters]
        # R [obs x B_r x draws]

        V = T.set_subtensor(V[:, 0, :], (
            B[param_map['0,1-bias-random']]*R[:, draw_map['0,1-bias-random'], :] +
            B[param_map['0,2-bias-random']]*R[:, draw_map['0,2-bias-random'], :] +
            B[param_map['0,1-0-beta-random']]*R[:, draw_map['0,1-0-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['0,2-0-beta-random']]*R[:, draw_map['0,2-0-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['0,1-1-beta-random']]*R[:, draw_map['0,1-1-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['0,2-1-beta-random']]*R[:, draw_map['0,2-1-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['0-bias']] +
            B[param_map['0-1-beta']]*X[:, 0, np.newaxis] +
            # B[b_map['0-0-random']]*R[:, 0, :]*X[:, 0, np.newaxis] +
            B[param_map['0-2-beta']]*X[:, 1, np.newaxis]  # +
            # B[b_map['0-1-random']]*R[:, 1, :]*X[:, 1, np.newaxis]  # +
        ))
        V = T.set_subtensor(V[:, 1, :], (
            B[param_map['0,1-bias-random']]*R[:, draw_map['0,1-bias-random'], :] +
            B[param_map['1,2-bias-random']]*R[:, draw_map['1,2-bias-random'], :] +
            B[param_map['0,1-0-beta-random']]*R[:, draw_map['0,1-0-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['1,2-0-beta-random']]*R[:, draw_map['1,2-0-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['0,1-1-beta-random']]*R[:, draw_map['0,1-1-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['1,2-1-beta-random']]*R[:, draw_map['1,2-1-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['1-bias']] +
            B[param_map['1-1-beta']]*X[:, 0, np.newaxis] +
            # B[b_map['1-0-random']]*R[:, 0, :]*X[:, 0, np.newaxis] +
            B[param_map['1-2-beta']]*X[:, 1, np.newaxis]  # +
            # B[b_map['1-1-random']]*R[:, 1, :]*X[:, 1, np.newaxis]  # +
        ))
        V = T.set_subtensor(V[:, 2, :], (
            B[param_map['0,2-bias-random']]*R[:, draw_map['0,2-bias-random'], :] +
            B[param_map['1,2-bias-random']]*R[:, draw_map['1,2-bias-random'], :] +
            B[param_map['0,2-0-beta-random']]*R[:, draw_map['0,2-0-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['1,2-0-beta-random']]*R[:, draw_map['1,2-0-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['0,2-1-beta-random']]*R[:, draw_map['0,2-1-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['1,2-1-beta-random']]*R[:, draw_map['1,2-1-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['2-bias']] +
            B[param_map['2-1-beta']]*X[:, 0, np.newaxis] +
            # B[b_map['2-0-random']]*R[:, 0, :]*X[:, 0, np.newaxis] +
            B[param_map['2-2-beta']]*X[:, 1, np.newaxis]  # +
            # B[b_map['2-1-random']]*R[:, 1, :]*X[:, 1, np.newaxis] #  +
        ))
        return V


# Make data
# Based on:
# http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html#example-calibration-plot-calibration-py
n_samples = 5000
centers = [(5, 5), (-5.5, -4.5), (-4.5, -5.5)]
X, y = make_blobs(
    n_samples=n_samples, n_features=2, cluster_std=1.0,
    centers=centers, shuffle=True, random_state=1
)

plt.figure()
y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
for this_y, color in zip(y_unique, colors):
    this_X = X[y == this_y]
    plt.scatter(this_X[:, 0], this_X[:, 1], c=color, alpha=0.5, label="Class %s" % this_y)

plt.legend(loc="best")
title = 'correlated_classes_test_data'
plt.title("Data")
plt.savefig(title)

# Estimate model
num_draws = 2000
num_alternatives = 3
input_parameters = np.zeros(len(param_map), dtype='float64')
uf = UF()

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

print('Classes 2 and 3 are overlapping, class 0 is separated')

print_params(output_parameters, param_map)
print('Gradient is: ' + str(final_grad))
print('Estimate time: %.2f' % (end_time - start_time))
print('Initial Cost is: %.2f' % initial_cost)
print('Initial Accuracy is: %.2f' % (1 - initial_error))
print('Cost is: %.2f' % cost)
print('Accuracy is: %.2f' % (1 - error))
