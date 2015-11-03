import numpy as np
import theano
import theano.tensor as T


parameter_map = {
    'bias_bicycle': 0,
    'bias_car': 1,
    'bias_pt_walk_access': 2,
    'bias_pt_park_access': 3,
    'bias_pt_kiss_access': 4,
    'cost_bicycle': 5,
    'cost_car': 6,
    'cost_pt_walk_access': 7,
    'cost_pt_park_access': 8,
    'cost_pt_kiss_access': 9,
    'cost_walk': 10,
    'random_cost_pt': 11,
    'random_cost_walk': 12,
    'random_cost_car': 13,
    'random_cost_bicycle': 14,
    'error_pt': 15,
    'error_car': 16,
    'error_non_mech': 17,
    'error_bicycle': 18,
    # 'error_pt_kiss_access': 19,
}

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
# draws[exps, biases_rp + self.parametersrp, draws]
# biases[biases]
# parameters[parameters]
# biases_rp[biases_rp]
# self.parametersrp[self.parametersrp]
# return[exps, draws]


class UtilityFunctions(object):
    def __init__(self, float_dtype):
        self.X = T.matrix('X', dtype=float_dtype)
        self.draws = T.tensor3('draws', dtype=float_dtype)
        self.parameters = T.vector('parameters', dtype=float_dtype)
        self.input_parameters = np.zeros(len(parameter_map))  # np.random.randn(14).astype(float_dtype)

    def bicycle(self):
        bias = self.parameters[parameter_map['bias_bicycle']]
        cost = (self.X[:, feature_map['Bicycle_Cost_Outward']] * self.parameters[parameter_map['cost_bicycle']] +
                self.X[:, feature_map['Bicycle_Cost_Return']] * self.parameters[parameter_map['cost_bicycle']])
        error = (self.parameters[parameter_map['error_bicycle']] * self.draws[:, draw_map['error_bicycle'], :] +
                 self.parameters[parameter_map['error_non_mech']] * self.draws[:, draw_map['error_non_mech'], :])
        random_cost = ((self.X[:, feature_map['Bicycle_Cost_Outward']] *
                        self.parameters[parameter_map['random_cost_bicycle']])[:, np.newaxis] *
                       self.draws[:, draw_map['random_cost_bicycle'], :] +
                       (self.X[:, feature_map['Bicycle_Cost_Return']] *
                        self.parameters[parameter_map['random_cost_bicycle']])[:, np.newaxis] *
                       self.draws[:, draw_map['random_cost_bicycle'], :])

        utility = bias + cost[:, np.newaxis] + random_cost + error
        fn = theano.function([self.X, self.parameters, self.draws], utility, name='bicycle_utility')
        return fn

    def car(self):
        bias = self.parameters[parameter_map['bias_car']]
        cost = (self.X[:, feature_map['Car_Cost_Outward']] * self.parameters[parameter_map['cost_car']] +
                self.X[:, feature_map['Car_Cost_Return']] * self.parameters[parameter_map['cost_car']])
        error = self.parameters[parameter_map['error_car']] * self.draws[:, draw_map['error_car'], :]
        random_cost = ((self.X[:, feature_map['Car_Cost_Outward']] *
                        self.parameters[parameter_map['random_cost_car']])[:, np.newaxis] *
                       self.draws[:, draw_map['random_cost_car'], :] +
                       (self.X[:, feature_map['Car_Cost_Return']] *
                        self.parameters[parameter_map['random_cost_car']])[:, np.newaxis] *
                       self.draws[:, draw_map['random_cost_car'], :])

        utility = bias + cost[:, np.newaxis] + random_cost + error
        fn = theano.function([self.X, self.parameters, self.draws], utility, name='car_utility')
        return fn

    def pt_walk_access(self):
        bias = self.parameters[parameter_map['bias_pt_walk_access']]
        cost = (self.X[:, feature_map['WAWE_Cost_Outward']] * self.parameters[parameter_map['cost_pt_walk_access']] +
                self.X[:, feature_map['WAWE_Cost_Return']] * self.parameters[parameter_map['cost_pt_walk_access']])
        error = self.parameters[parameter_map['error_pt']] * self.draws[:, draw_map['error_pt'], :]
        random_cost = ((self.X[:, feature_map['WAWE_Cost_Outward']] *
                        self.parameters[parameter_map['random_cost_pt']])[:, np.newaxis] *
                       self.draws[:, draw_map['random_cost_pt'], :] +
                       (self.X[:, feature_map['WAWE_Cost_Return']] *
                        self.parameters[parameter_map['random_cost_pt']])[:, np.newaxis] *
                       self.draws[:, draw_map['random_cost_pt'], :])

        utility = bias + cost[:, np.newaxis] + random_cost + error
        fn = theano.function([self.X, self.parameters, self.draws], utility, name='pt_walk_access_utility')
        return fn

    def pt_park_access(self):
        bias = self.parameters[parameter_map['bias_pt_park_access']]
        cost = (self.X[:, feature_map['PAWE_Cost_Outward']] * self.parameters[parameter_map['cost_pt_park_access']] +
                self.X[:, feature_map['WAPE_Cost_Return']] * self.parameters[parameter_map['cost_pt_park_access']])
        error = self.parameters[parameter_map['error_pt']] * self.draws[:, draw_map['error_pt'], :]
        random_cost = ((self.X[:, feature_map['PAWE_Cost_Outward']] *
                        self.parameters[parameter_map['random_cost_pt']])[:, np.newaxis] *
                       self.draws[:, draw_map['random_cost_pt'], :] +
                       (self.X[:, feature_map['WAPE_Cost_Return']] *
                        self.parameters[parameter_map['random_cost_pt']])[:, np.newaxis] *
                       self.draws[:, draw_map['random_cost_pt'], :])

        utility = bias + cost[:, np.newaxis] + random_cost + error
        fn = theano.function([self.X, self.parameters, self.draws], utility, name='pt_park_access_utility')
        return fn

    def pt_kiss_access(self):
        bias = self.parameters[parameter_map['bias_pt_kiss_access']]
        cost = (self.X[:, feature_map['KAWE_Cost_Outward']] * self.parameters[parameter_map['cost_pt_kiss_access']] +
                self.X[:, feature_map['WAKE_Cost_Return']] * self.parameters[parameter_map['cost_pt_kiss_access']])
        error = self.parameters[parameter_map['error_pt']] * self.draws[:, draw_map['error_pt'], :]
        random_cost = ((self.X[:, feature_map['KAWE_Cost_Outward']] *
                        self.parameters[parameter_map['random_cost_pt']])[:, np.newaxis] *
                       self.draws[:, draw_map['random_cost_pt'], :] +
                       (self.X[:, feature_map['WAKE_Cost_Return']] *
                        self.parameters[parameter_map['random_cost_pt']])[:, np.newaxis] *
                       self.draws[:, draw_map['random_cost_pt'], :])

        utility = bias + cost[:, np.newaxis] + random_cost + error
        fn = theano.function([self.X, self.parameters, self.draws], utility, name='pt_kiss_access_utility')
        return fn

    def walk(self):
        cost = (self.X[:, feature_map['Walk_Cost_Outward']] * self.parameters[parameter_map['cost_walk']] +
                self.X[:, feature_map['Walk_Cost_Return']] * self.parameters[parameter_map['cost_walk']])
        error = self.parameters[parameter_map['error_non_mech']] * self.draws[:, draw_map['error_non_mech'], :]
        random_cost = ((self.X[:, feature_map['Walk_Cost_Outward']] *
                        self.parameters[parameter_map['random_cost_walk']])[:, np.newaxis] *
                       self.draws[:, draw_map['random_cost_walk'], :] +
                       (self.X[:, feature_map['Walk_Cost_Return']] *
                        self.parameters[parameter_map['random_cost_walk']])[:, np.newaxis] *
                       self.draws[:, draw_map['random_cost_walk'], :])

        utility = cost[:, np.newaxis] + random_cost + error
        fn = theano.function([self.X, self.parameters, self.draws], utility, name='walk_utility')
        return fn
    
    def fn_list(self):
        return [
            self.bicycle(),
            self.car(),
            self.pt_walk_access(),
            self.pt_park_access(),
            self.pt_kiss_access(),
            self.walk(),
        ]
