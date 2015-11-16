import theano
import theano.tensor as T
import numpy as np


parameter_map = {
    'bias_bicycle': 0,
    'cost_bicycle': 1,
    'error_bicycle': 2,
    'error_non_mech': 3,
    'random_cost_bicycle': 4
}
feature_map = {
    'Bicycle_Cost_Outward': 0,
    'Bicycle_Cost_Return': 1
}
draw_map = {
    'error_bicycle': 0,
    'error_non_mech': 1,
    'random_cost_bicycle': 2
}

def bicycle():
    X = T.matrix('X', dtype='float64')
    draws = T.tensor3('draws', dtype='float64')
    parameters = T.vector('parameters', dtype='float64')

    bias = parameters[parameter_map['bias_bicycle']]
    cost = (X[:, feature_map['Bicycle_Cost_Outward']] * parameters[parameter_map['cost_bicycle']] +
            X[:, feature_map['Bicycle_Cost_Return']] * parameters[parameter_map['cost_bicycle']])
    error = (parameters[parameter_map['error_bicycle']] * draws[:, draw_map['error_bicycle'], :] +
             parameters[parameter_map['error_non_mech']] * draws[:, draw_map['error_non_mech'], :])
    random_cost = ((X[:, feature_map['Bicycle_Cost_Outward']] *
                    parameters[parameter_map['random_cost_bicycle']])[:, np.newaxis] *
                   draws[:, draw_map['random_cost_bicycle'], :] +
                   (X[:, feature_map['Bicycle_Cost_Return']] *
                    parameters[parameter_map['random_cost_bicycle']])[:, np.newaxis] *
                   draws[:, draw_map['random_cost_bicycle'], :])
    return theano.function([X, draws, parameters], bias + cost[:, np.newaxis] + random_cost + error)

X = np.array(
    [[1, 1],
     [2, 2]]
)
parameters = np.array([0, 1, 2, 3, 4])
draws = np.array([0.5, 0.2, 0.1])

bicycle_utility_fn = bicycle()
u = bicycle_utility_fn(X, draws, parameters)
print(u)

