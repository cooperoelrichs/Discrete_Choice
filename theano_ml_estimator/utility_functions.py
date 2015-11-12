import numpy as np
import theano.tensor as T


parameter_map = {
    'bias_bicycle': 0,
    'bias_car': 1,
    'bias_pt_walk_access': 2,
    'bias_pt_park_access': 3,
    'bias_pt_kiss_access': 4,
    'cost_bicycle': 5,
    'cost_car': 6,
    'cost_pt': 7,
    'cost_walk': 8,
    'random_cost_pt': 9,
    'random_cost_walk': 10,
    'random_cost_car': 11,
    'random_cost_bicycle': 12,
    'error_pt': 13,
    'error_car': 14,
    'error_non_mech': 15,
    'error_bicycle': 16,
    # 'error_pt_kiss_access': 19,
}

parameter_names = [
    'bias_bicycle',
    'bias_car',
    'bias_pt_walk_access',
    'bias_pt_park_access',
    'bias_pt_kiss_access',
    'cost_bicycle',
    'cost_car',
    'cost_pt',
    'cost_walk',
    'random_cost_pt',
    'random_cost_walk',
    'random_cost_car',
    'random_cost_bicycle',
    'error_pt',
    'error_car',
    'error_non_mech',
    'error_bicycle',
    # 'error_pt_kiss_access': 19,
]

draw_map = {
    'random_cost_pt': 0,
    'random_cost_walk': 1,
    'random_cost_car': 2,
    'random_cost_bicycle': 3,
    'error_pt': 4,
    'error_car': 5,
    'error_non_mech': 6,
    'error_bicycle': 7,
    # 'error_pt_kiss_access': 8,
}

feature_map = {
    'Bicycle_Cost_Outward': 0,
    'Bicycle_Cost_Return': 1,
    'Car_Cost_Outward': 2,
    'Car_Cost_Return': 3,
    'WAWE_Cost_Outward': 4,
    'WAWE_Cost_Return': 5,
    'PAWE_Cost_Outward': 6,
    'WAPE_Cost_Return': 7,
    'KAWE_Cost_Outward': 8,
    'WAKE_Cost_Return': 9,
    'Walk_Cost_Outward': 10,
    'Walk_Cost_Return': 11,
}

# X[exps, features]
# draws[exps, biases_rp + parameters_rp, draws]
# biases[biases]
# parameters[parameters]
# biases_rp[biases_rp]
# parameters_rp[parameters_rp]
# return[exps, draws]


class UtilityFunctions(object):
    def __init__(self):
        self.input_parameters = np.random.randn(len(parameter_map))  # np.random.randn(14).astype(float_dtype)

    def calculate_V(self, V, X, parameters, draws):
        # V[exps, alts, draws]
        V = T.set_subtensor(V[:, 0, :], self.bicycle(X, parameters, draws))
        V = T.set_subtensor(V[:, 1, :], self.car(X, parameters, draws))
        V = T.set_subtensor(V[:, 2, :], self.pt_walk_access(X, parameters, draws))
        V = T.set_subtensor(V[:, 3, :], self.pt_park_access(X, parameters, draws))
        V = T.set_subtensor(V[:, 4, :], self.pt_kiss_access(X, parameters, draws))
        V = T.set_subtensor(V[:, 5, :], self.walk(X, parameters, draws))
        return V

    def bicycle(self, X, parameters, draws):
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

        return bias + cost[:, np.newaxis] + random_cost + error

    def car(self, X, parameters, draws):
        bias = parameters[parameter_map['bias_car']]
        cost = (X[:, feature_map['Car_Cost_Outward']] * parameters[parameter_map['cost_car']] +
                X[:, feature_map['Car_Cost_Return']] * parameters[parameter_map['cost_car']])
        error = parameters[parameter_map['error_car']] * draws[:, draw_map['error_car'], :]
        random_cost = ((X[:, feature_map['Car_Cost_Outward']] *
                        parameters[parameter_map['random_cost_car']])[:, np.newaxis] *
                       draws[:, draw_map['random_cost_car'], :] +
                       (X[:, feature_map['Car_Cost_Return']] *
                        parameters[parameter_map['random_cost_car']])[:, np.newaxis] *
                       draws[:, draw_map['random_cost_car'], :])

        return bias + cost[:, np.newaxis] + random_cost + error

    def pt_walk_access(self, X, parameters, draws):
        bias = parameters[parameter_map['bias_pt_walk_access']]
        cost = (X[:, feature_map['WAWE_Cost_Outward']] * parameters[parameter_map['cost_pt']] +
                X[:, feature_map['WAWE_Cost_Return']] * parameters[parameter_map['cost_pt']])
        error = parameters[parameter_map['error_pt']] * draws[:, draw_map['error_pt'], :]
        random_cost = ((X[:, feature_map['WAWE_Cost_Outward']] *
                        parameters[parameter_map['random_cost_pt']])[:, np.newaxis] *
                       draws[:, draw_map['random_cost_pt'], :] +
                       (X[:, feature_map['WAWE_Cost_Return']] *
                        parameters[parameter_map['random_cost_pt']])[:, np.newaxis] *
                       draws[:, draw_map['random_cost_pt'], :])

        return bias + cost[:, np.newaxis] + random_cost + error

    def pt_park_access(self, X, parameters, draws):
        bias = parameters[parameter_map['bias_pt_park_access']]
        cost = (X[:, feature_map['PAWE_Cost_Outward']] * parameters[parameter_map['cost_pt']] +
                X[:, feature_map['WAPE_Cost_Return']] * parameters[parameter_map['cost_pt']])
        error = parameters[parameter_map['error_pt']] * draws[:, draw_map['error_pt'], :]
        random_cost = ((X[:, feature_map['PAWE_Cost_Outward']] *
                        parameters[parameter_map['random_cost_pt']])[:, np.newaxis] *
                       draws[:, draw_map['random_cost_pt'], :] +
                       (X[:, feature_map['WAPE_Cost_Return']] *
                        parameters[parameter_map['random_cost_pt']])[:, np.newaxis] *
                       draws[:, draw_map['random_cost_pt'], :])

        return bias + cost[:, np.newaxis] + random_cost + error

    def pt_kiss_access(self, X, parameters, draws):
        bias = parameters[parameter_map['bias_pt_kiss_access']]
        cost = (X[:, feature_map['KAWE_Cost_Outward']] * parameters[parameter_map['cost_pt']] +
                X[:, feature_map['WAKE_Cost_Return']] * parameters[parameter_map['cost_pt']])
        error = parameters[parameter_map['error_pt']] * draws[:, draw_map['error_pt'], :]
        random_cost = ((X[:, feature_map['KAWE_Cost_Outward']] *
                        parameters[parameter_map['random_cost_pt']])[:, np.newaxis] *
                       draws[:, draw_map['random_cost_pt'], :] +
                       (X[:, feature_map['WAKE_Cost_Return']] *
                        parameters[parameter_map['random_cost_pt']])[:, np.newaxis] *
                       draws[:, draw_map['random_cost_pt'], :])

        return bias + cost[:, np.newaxis] + random_cost + error

    def walk(self, X, parameters, draws):
        cost = (X[:, feature_map['Walk_Cost_Outward']] * parameters[parameter_map['cost_walk']] +
                X[:, feature_map['Walk_Cost_Return']] * parameters[parameter_map['cost_walk']])
        error = parameters[parameter_map['error_non_mech']] * draws[:, draw_map['error_non_mech'], :]
        random_cost = ((X[:, feature_map['Walk_Cost_Outward']] *
                        parameters[parameter_map['random_cost_walk']])[:, np.newaxis] *
                       draws[:, draw_map['random_cost_walk'], :] +
                       (X[:, feature_map['Walk_Cost_Return']] *
                        parameters[parameter_map['random_cost_walk']])[:, np.newaxis] *
                       draws[:, draw_map['random_cost_walk'], :])

        return cost[:, np.newaxis] + random_cost + error
