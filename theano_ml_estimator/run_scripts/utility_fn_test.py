import numpy as np


X = np.ones((14, 3))
draws = np.random.randn(14, 6, 10)
parameters = [2, 0.5, 0.1, 0.01]

parameter_map = {
    'bias_0': 0,
    'cost_0': 1,
    'bias_rp_0': 2,
    'cost_rp_0': 3,
}

feature_map = {
    'cost_0': 0
}


def v_0(X_, draws_, parameters_):
    # X[exps, features]
    # draws[exps, biases_rp + parameters_rp, draws]
    # biases[biases]
    # parameters[parameters]
    # biases_rp[biases_rp]
    # parameters_rp[parameters_rp]
    # return[exps, 1, draws]

    bias = parameters_[parameter_map['bias_0']]
    cost = X_[:, feature_map['cost_0']] * parameters_[parameter_map['cost_0']]
    bias_rp = parameters_[parameter_map['bias_rp_0']] * draws_[:, parameter_map['bias_rp_0'], :]
    cost_rp = (X_[:, feature_map['cost_0']] * parameters_[parameter_map['cost_rp_0']])[:, np.newaxis] * draws_[:, parameter_map['cost_rp_0'], :]
    return (bias +
            cost[:, np.newaxis] +
            bias_rp +
            cost_rp)


v_output = v_0(X, draws, parameters)

print(v_output)
print(v_output.shape)
