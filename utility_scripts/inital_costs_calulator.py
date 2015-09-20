# from sklearn import datasets
from numpy import unique
import numpy as np


data_set_file_name = 'HWW_Brisbane.dat'
data = np.genfromtxt(data_set_file_name, delimiter="\t", names=True)  # skip_header=1)
data = np.genfromtxt(data_set_file_name, delimiter="\t", skip_header=1)
headers = np.array(open(data_set_file_name, 'r').readline().rstrip().split('\t'))

# data = data[data['exclude'] != 0]

columns = [11, 16, 14, 13, 9, 18, 22, 15, 10, 24, 19, 12, 21, 8, 20, 17, 25, 23, 2]
av_columns = [-1, -2, -3, -4 , -5, -6]
y = data[:, 0] - 1
X = data[:, columns]
av = data[:, av_columns]

print(headers[columns])
print(headers[av_columns])
print(unique(y))

ps = np.zeros(len(y))
costs = np.zeros_like(ps)
for i, choice in enumerate(y):
    ps[i] = 1.0 / np.sum(av[i])
    costs[i] = np.log(ps[i])

cost_w = np.dot(costs, np.transpose(X[:, 18]))

print(len(y))
print(cost_w)
print(cost_w - -2461970.734)
