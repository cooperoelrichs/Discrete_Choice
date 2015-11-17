import theano
import theano.tensor as T
import numpy as np


parameter_map = {
    'bias_bicycle': 0,
    'cost_bicycle': 1,
    'error_bicycle': 2,
    'random_cost_bicycle': 3
}
feature_map = {
    'Bicycle_Cost_Outward': 0,
}
draw_map = {
    'error_bicycle': 0,
    'random_cost_bicycle': 1
}

def bicycle():
    X = T.matrix('X', dtype='float64')
    draws = T.tensor3('draws', dtype='float64')
    parameters = T.vector('parameters', dtype='float64')

    bias = parameters[parameter_map['bias_bicycle']]
    cost = X[:, feature_map['Bicycle_Cost_Outward']] * parameters[parameter_map['cost_bicycle']]
    error = parameters[parameter_map['error_bicycle']] * draws[:, draw_map['error_bicycle'], :]
    random_cost = ((X[:, feature_map['Bicycle_Cost_Outward']] *
                    parameters[parameter_map['random_cost_bicycle']])[:, np.newaxis] *
                   draws[:, draw_map['random_cost_bicycle'], :])
    utility = bias + cost[:, np.newaxis] + random_cost + error
    return theano.function([X, draws, parameters], utility, name='bicycle_utility')

X = np.array(
    [[1],
     [2]]
)
parameters = np.array([1, 2, 3, 4])
draws = np.array(  # 2x5x1
    [[[0.5],
      [0.5],
      [0.5],
      [0.5]],
     [[0.5],
      [0.5],
      [0.5],
      [0.5]]]
)

bicycle_utility_fn = bicycle()
u = bicycle_utility_fn(X, draws, parameters)

print([1 + 2*1 + 3*0.5 + 4*1*0.5, 1 + 2*2 + 3*0.5 + 4*2*0.5])
print(u)
