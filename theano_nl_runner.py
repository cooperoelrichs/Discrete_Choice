import numpy as np
# from sklearn.datasets import load_digits
from nested_logit_estimator import NestedLogitEstimator

# Setup
alternatives = 3
nests = np.array([0, 1], dtype='int32')
nest_indices = np.array([0, 1, 0])
# alt_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
lambdas = np.array([2, 1])
# num_nests = lambdas.shape[0]
# lambdas_expanded = lambdas[nest_indices]

# digits = load_digits()
# data_shape = digits.data.shape
# X = digits.data
# y = digits.target

file_name = 'biogeme_files/swissmetro.dat'
data = np.genfromtxt(file_name, delimiter='\t', skip_header=1)
headers = np.array(open(file_name, 'r').readline().rstrip().split('\t'))
data = data[data[:, -1] != 0]  # choice != 0
data = data[data[:, 2] != 0]  # sp
data = data[(data[:, 4] == 1) | (data[:, 4] == 3)]  # purpose == 1 or purpose == 3
columns = [18, 19, 21, 22, 25, 26]   # ['TRAIN_TT' 'TRAIN_CO' 'SM_TT' 'SM_CO' 'CAR_TT' 'CAR_CO']
y = (data[:, -1] - 1).astype('int64')
X = data[:, columns]
X /= 100  # scale the costs and travel times
data_shape = X.shape

print(headers[columns])
print(np.unique(y))
print(data_shape)

W_input = np.zeros((data_shape[1], alternatives))  # rand
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
