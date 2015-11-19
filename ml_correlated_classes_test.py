import numpy as np
import theano
import theano.tensor as T
from theano_ml_estimator.mixed_logit_estimator import MixedLogitEstimator
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.datasets import make_blobs


b_map = OrderedDict([
    ('1-bias', 0), ('1-correlation', 1), ('1-1', 2), ('1-1-random', 3), ('1-2', 4), ('1-2-random', 5),  # ('1-3', 5), ('1-3-random', 6),
    ('2-bias', 6), ('2-correlation', 7), ('2-1', 8), ('2-1-random', 9), ('2-2', 10), ('2-2-random', 11),  # ('2-3', 12), ('2-3-random', 13),
    ('3-bias', 12), ('3-1', 13), ('3-1-random', 14), ('3-2', 15), ('3-2-random', 16),  # ('3-3', 19), ('3-3-random', 20),
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
            B[b_map['1-bias']] +
            B[b_map['1-1']]*X[:, 0, np.newaxis] +
            B[b_map['1-1-random']]*R[:, 0, :]*X[:, 0, np.newaxis] +
            B[b_map['1-2']]*X[:, 1, np.newaxis] +
            B[b_map['1-2-random']]*R[:, 1, :]*X[:, 1, np.newaxis]  # +
            # B[b_map['1-3']]*X[:, 2, np.newaxis] +
            # B[b_map['1-3-random']]*R[:, 2, :]*X[:, 2, np.newaxis]
        ))
        V = T.set_subtensor(V[:, 1, :], (
            B[b_map['2-bias']] +
            B[b_map['2-1']]*X[:, 0, np.newaxis] +
            B[b_map['2-1-random']]*R[:, 0, :]*X[:, 0, np.newaxis] +
            B[b_map['2-2']]*X[:, 1, np.newaxis] +
            B[b_map['2-2-random']]*R[:, 1, :]*X[:, 1, np.newaxis]  # +
            # B[b_map['2-3']]*X[:, 2, np.newaxis] +
            # B[b_map['2-3-random']]*R[:, 2, :]*X[:, 2, np.newaxis]
        ))
        V = T.set_subtensor(V[:, 2, :], (
            B[b_map['3-bias']] +
            B[b_map['3-1']]*X[:, 0, np.newaxis] +
            B[b_map['3-1-random']]*R[:, 0, :]*X[:, 0, np.newaxis] +
            B[b_map['3-2']]*X[:, 1, np.newaxis] +
            B[b_map['3-2-random']]*R[:, 1, :]*X[:, 1, np.newaxis] #  +
            # B[b_map['3-3']]*X[:, 2, np.newaxis] +
            # B[b_map['3-3-random']]*R[:, 2, :]*X[:, 2, np.newaxis]
        ))
        return V


# Make data
n_samples = 5000
centers = [(-5.5, -5.5), (-4.5, -4.5), (5, 5)]
X, y = make_blobs(
    n_samples=n_samples, n_features=2, cluster_std=1.0,
    centers=centers, shuffle=True, random_state=42
)

# y[:n_samples // 2] = 0
# y[n_samples // 2:] = 1

plt.figure()
y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
for this_y, color in zip(y_unique, colors):
    this_X = X[y == this_y]
    plt.scatter(this_X[:, 0], this_X[:, 1], c=color, alpha=0.5,
                label="Class %s" % this_y)
plt.legend(loc="best")
plt.title("Data")
# plt.show()

# exit()

# Estimate model
num_draws = 1000
num_alternatives = 3
input_parameters = np.zeros(len(b_map), dtype='float64')
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

print('Generated data test.')
print('1 informative feature with some noise, 1 informative feature with lots of noise, 1 pure noise feature')
print_params(output_parameters, b_map)
print('Gradient is: ' + str(final_grad))
print('Estimate time: %.2f' % (end_time - start_time))
print('Initial Cost is: %.2f' % initial_cost)
print('Initial Accuracy is: %.2f' % (1 - initial_error))
print('Cost is: %.2f' % cost)
print('Accuracy is: %.2f' % (1 - error))

plt.show()
