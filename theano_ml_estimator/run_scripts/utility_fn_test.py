import numpy as np


X = np.array([[1, 5], [1, 5], [1, 5]])
draws = np.ones((3, 2, 2)) * 200
parameters = [2, 0.5, 0.2, 0.001]

print(2 + 1 * 0.5 + 0.2 * 200 + 0.001 * 5 * 200)

parameter_map = {
    'bias_0': 0,
    'cost_0': 1,
    'bias_rp_0': 2,
    'cost_rp_1': 3,
}

draws_map = {
    'bias_rp_0': 0,
    'cost_rp_1': 1,
}

feature_map = {
    'cost_0': 0,
    'cost_1': 1,
}


def v_0(X_, draws_, parameters_):
    # X[exps, features]
    # draws[exps, biases_rp + parameters_rp, draws]
    # biases[biases]
    # parameters[parameters]
    # biases_rp[biases_rp]
    # parameters_rp[parameters_rp]
    # return[exps, draws]

    bias = parameters_[parameter_map['bias_0']]
    cost = X_[:, feature_map['cost_0']] * parameters_[parameter_map['cost_0']]
    bias_rp = parameters_[parameter_map['bias_rp_0']] * draws_[:, draws_map['bias_rp_0'], :]
    cost_rp = ((X_[:, feature_map['cost_1']] * parameters_[parameter_map['cost_rp_1']])[:, np.newaxis] *
               draws_[:, draws_map['cost_rp_1'], :])
    return bias + cost[:, np.newaxis] + bias_rp + cost_rp


v_output = v_0(X, draws, parameters)

print(v_output)
print(v_output.shape)
