import numpy as np
import theano
import theano.tensor as T
from theano_ml_estimator.mixed_logit_estimator import MixedLogitEstimator
import time

import sklearn.datasets as datasets
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs


param_map = OrderedDict([(name, i) for i, name in enumerate([
    '0-bias-random',
    # '2-bias-random',
    '--0-beta-random',
    '--1-beta-random',
    '--2-beta-random',

    '0-bias',
    # '2-bias',
    '0-0-beta',
    '0-1-beta',
    '0-2-beta',
    '1-0-beta',
    '1-1-beta',
    '1-2-beta',
])])

draw_map = OrderedDict([(name, i) for i, name in enumerate([
    '0-bias-random',
    # '2-bias-random',
    '--0-beta-random',
    '--1-beta-random',
    '--2-beta-random',
    # '1-0-beta-random',
    # '1-1-beta-random',
    # '2-2-beta-random',
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
            B[param_map['0-bias']] +
            B[param_map['0-bias-random']]*R[:, draw_map['0-bias-random'], :] +
            B[param_map['0-0-beta']]*X[:, 0, np.newaxis] +
            B[param_map['--0-beta-random']]*R[:, draw_map['--0-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['0-1-beta']]*X[:, 1, np.newaxis] +
            B[param_map['--1-beta-random']]*R[:, draw_map['--1-beta-random'], :]*X[:, 1, np.newaxis] +
            B[param_map['0-2-beta']]*X[:, 2, np.newaxis] +
            B[param_map['--2-beta-random']]*R[:, draw_map['--2-beta-random'], :]*X[:, 2, np.newaxis]
        ))
        V = T.set_subtensor(V[:, 1, :], (
            # B[param_map['2-bias']] +
            # B[param_map['2-bias-random']]*R[:, draw_map['2-bias-random'], :] +
            B[param_map['1-0-beta']]*X[:, 0, np.newaxis] +
            B[param_map['--0-beta-random']]*R[:, draw_map['--0-beta-random'], :]*X[:, 0, np.newaxis] +
            B[param_map['1-1-beta']]*X[:, 1, np.newaxis] +
            B[param_map['--1-beta-random']]*R[:, draw_map['--1-beta-random'], :]*X[:, 1, np.newaxis] +
            B[param_map['1-2-beta']]*X[:, 2, np.newaxis] +
            B[param_map['--2-beta-random']]*R[:, draw_map['--2-beta-random'], :]*X[:, 2, np.newaxis]
        ))
        return V

# X, y = datasets.make_classification(
#     n_samples=5000, n_features=3,
#     n_informative=2, n_redundant=0, n_repeated=0,
#     n_classes=2, n_clusters_per_class=1,
#     random_state=1,
#     # flip_y=0.4
# )

n_samples = 5000
centers = [(-2, -2, -2), (2, 2, 2)]
X, y = make_blobs(
    n_samples=n_samples, n_features=3, cluster_std=[0.8, 2, 1],
    centers=centers, shuffle=True, random_state=1
)

print(X.shape)
print(y.shape)

X[:, 2] = (np.random.uniform(size=X.shape[0]) - 0.5) * 20

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
limits = (-15, 15)
y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
for this_y, color in zip(y_unique, colors):
    this_X = X[y == this_y]
    # plt.legend(loc="best")
    for ax, x1, x2 in [(ax1, 0, 1), (ax2, 0, 2), (ax3, 1, 2)]:
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_title("X%i - X%i" % (x1, x2))
        ax.scatter(this_X[:, x1], this_X[:, x2], c=color, alpha=0.5, label="Class %s" % this_y)

plot_name = 'noisy_data_test'
fig.suptitle(plot_name)
# plt.show()
plt.savefig(plot_name)

# exit()

num_draws = 2000
num_alternatives = 2
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

print('Generated data test.')
print('1 informative feature with some noise, 1 informative feature with lots of noise, 1 pure noise feature')
print_params(output_parameters, param_map)
print('Gradient is: ' + str(final_grad))
print('Estimate time: %.2f' % (end_time - start_time))
print('Initial Cost is: %.2f' % initial_cost)
print('Initial Accuracy is: %.2f' % (1 - initial_error))
print('Cost is: %.2f' % cost)
print('Accuracy is: %.2f' % (1 - error))
