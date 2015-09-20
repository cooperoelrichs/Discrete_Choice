import numpy as np
from theano_nl_estimator.nested_logit_estimator import NestedLogitEstimator
from nl_data_loader.nl_data_loader import NLDataLoader


dl = NLDataLoader('biogeme_files/swissmetro.dat', '\t',
                  ['TRAIN_TT', 'TRAIN_CO', 'SM_TT', 'SM_CO', 'CAR_TT', 'CAR_CO'],
                  'CHOICE')
dl.data = dl.data[dl.get('CHOICE') != 0]
dl.data = dl.data[dl.get('SP') != 0]
dl.data = dl.data[(dl.get('PURPOSE') == 1) | (dl.get('PURPOSE') == 3)]
dl.print_data_info()
X, y = dl.get_X_and_y()
y -= 1
y = y.astype('int64')
X /= 100  # scale the costs and travel times

alternatives = 3
nests = np.array([0, 1], dtype='int32')
nest_indices = np.array([0, 1, 0])
# alt_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
lambdas = np.array([2, 1])
# num_nests = lambdas.shape[0]
# lambdas_expanded = lambdas[nest_indices]

W_input = np.zeros((X.shape[1], alternatives))  # rand
b_input = np.zeros(alternatives)
# P = np.zeros((data_shape[0], alternatives))

nle = NestedLogitEstimator(X, y, W_input, b_input, lambdas, nests, nest_indices, alternatives)
cost, error, _ = nle.results(nle.initial_W, nle.initial_b, nle.initial_lambdas)
print(error)
print(cost)

cost, error, _, W, b, lambdas = nle.estimate()
print(error)
print(cost)
print(b)
print(lambdas)

print('Error is: %.2f' % (error * 100))

# mnl_probs = T.nnet.softmax(V)
# mnl_pred = T.argmax(mnl_probs, axis=1)
# mnl_error = T.mean(T.neq(mnl_pred, digits.target))

# params = [W, b]
# input = input
