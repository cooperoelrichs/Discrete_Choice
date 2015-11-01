import numpy as np


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

draws_map = {
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

def v_bicycle(X_, draws_, parameters_):
    bias = parameters_[parameter_map['bias_bicycle']]
    cost = (X_[:, feature_map['Bicycle_Cost_Outward']] * parameters_[parameter_map['cost_bicycle']] +
            X_[:, feature_map['Bicycle_Cost_Return']] * parameters_[parameter_map['cost_bicycle']])
    error = (parameters_[parameter_map['error_bicycle']] * draws_[:, draws_map['error_bicycle'], :] +
             parameters_[parameter_map['error_non_mech']] * draws_[:, draws_map['error_non_mech'], :])
    random_cost = ((X_[:, feature_map['Bicycle_Cost_Outward']] *
                    parameters_[parameter_map['random_cost_bicycle']])[:, np.newaxis] *
                   draws_[:, draws_map['random_cost_bicycle'], :] +
                   (X_[:, feature_map['Bicycle_Cost_Return']] *
                    parameters_[parameter_map['random_cost_bicycle']])[:, np.newaxis] *
                   draws_[:, draws_map['random_cost_bicycle'], :])
    return bias + cost[:, np.newaxis] + random_cost + error

def v_car(X_, draws_, parameters_):
    bias = parameters_[parameter_map['bias_car']]
    cost = (X_[:, feature_map['Car_Cost_Outward']] * parameters_[parameter_map['cost_car']] +
            X_[:, feature_map['Car_Cost_Return']] * parameters_[parameter_map['cost_car']])
    error = parameters_[parameter_map['error_car']] * draws_[:, draws_map['error_car'], :]
    random_cost = ((X_[:, feature_map['Car_Cost_Outward']] *
                    parameters_[parameter_map['random_cost_car']])[:, np.newaxis] *
                   draws_[:, draws_map['random_cost_car'], :] +
                   (X_[:, feature_map['Car_Cost_Return']] *
                    parameters_[parameter_map['random_cost_car']])[:, np.newaxis] *
                   draws_[:, draws_map['random_cost_car'], :])
    return bias + cost[:, np.newaxis] + random_cost + error

def v_pt_walk_access(X_, draws_, parameters_):
    bias = parameters_[parameter_map['bias_pt_walk_access']]
    cost = (X_[:, feature_map['WAWE_Cost_Outward']] * parameters_[parameter_map['cost_pt_walk_access']] +
            X_[:, feature_map['WAWE_Cost_Return']] * parameters_[parameter_map['cost_pt_walk_access']])
    error = parameters_[parameter_map['error_pt']] * draws_[:, draws_map['error_pt'], :]
    random_cost = ((X_[:, feature_map['WAWE_Cost_Outward']] *
                    parameters_[parameter_map['random_cost_pt']])[:, np.newaxis] *
                   draws_[:, draws_map['random_cost_pt'], :] +
                   (X_[:, feature_map['WAWE_Cost_Return']] *
                    parameters_[parameter_map['random_cost_pt']])[:, np.newaxis] *
                   draws_[:, draws_map['random_cost_pt'], :])
    return bias + cost[:, np.newaxis] + random_cost + error