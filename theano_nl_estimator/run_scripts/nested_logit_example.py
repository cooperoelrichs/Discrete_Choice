import numpy as np
from nl_data_loader.nl_data_loader import NLDataLoader


def sum_nest_for_nest(i, nest_indices, exp_V):
    # return exp_V[:, np[nest_indices == i][0]].sum(axis=1)
    return exp_V[:, nest_indices == i].sum(axis=1)


def calculate_nest_sums(exp_V, nests, nest_indices):
    nest_sums_T = np.array([sum_nest_for_nest(i, nest_indices, exp_V) for i in nests])

    # nest_sums_T, _ = theano.scan(
    #     sum_nest_for_nest,
    #     sequences=[nests],
    #     non_sequences=[nest_indices, exp_V],
    #     name='sum_exp_utilities_by_nests'
    # )
    return nest_sums_T.T


def calculate_probability_for_alternative(alt, lambdas, nest_indices, exp_V, nest_sums, denominator):
    numerator = exp_V[:, alt] * np.power(nest_sums[:, nest_indices[alt]], lambdas[nest_indices][alt] - 1)
    return numerator / denominator


def calculate_probabilities(exp_V, lambdas, alternatives, nest_indices):
    nest_sums = calculate_nest_sums(exp_V, nests, nest_indices)
    denominator = np.power(nest_sums, lambdas).sum(axis=1)

    P_T = np.array([calculate_probability_for_alternative(alt, lambdas, nest_indices, exp_V, nest_sums, denominator)
                    for alt in alternatives])

    # P_T, _ = theano.scan(
    #     calculate_probability_for_alternative,
    #     sequences=[alternatives],
    #     non_sequences=[lambdas, nest_indices, exp_V, nest_sums, denominator],
    #     name='calculate_probabilities_by_alternatives'
    # )
    return P_T.T


def nl_calculations(X, y, biases, lambdas, parameters, utility_functions, W, b, l):
    W[[utility_functions[:, 0], utility_functions[:, 1]]] = parameters[utility_functions[:, 2]]
    b[biases[:, 0]] = parameters[biases[:, 1]]
    l[lambdas[:, 0]] = parameters[lambdas[:, 1]]
    
    V = np.dot(X, W) + b  # calculate utilities
    # V = V - V.mean(axis=1, keepdims=True)  # numerical stability
    # V = np.clip(V, -80, 80)  # numerical stability
    exp_V = np.exp(V / l[nest_indices])  # exp of the scaled utilities
    P = calculate_probabilities(exp_V, l, alternatives, nest_indices)
    
    predictions = np.argmax(P, axis=1)
    error = np.mean(predictions[predictions.neq(y)])
    cost = -np.mean(np.log(P)[np.arange(y.shape[0]), y] * weights)
    return error, cost


float_dtype = 'float64'
int_dtype = 'int64'

cost_columns = [
    'Bicycle_Cost_Outward', 'Bicycle_Cost_Return',
    'Car_Cost_Outward', 'Car_Cost_Return',
    'WAWE_Cost_Outward', 'WAWE_Cost_Return',
    'PAWE_Cost_Outward', 'WAPE_Cost_Return',
    'KAWE_Cost_Outward', 'WAKE_Cost_Return',
    'Walk_Cost_Outward', 'Walk_Cost_Return',
]

dl = NLDataLoader('../../data/HWW_Melbourne.dat', '\t', cost_columns, 'choice', float_dtype, int_dtype)
dl.data = dl.data[dl.get('Bicycle_av') != 0]
dl.data = dl.data[dl.get('Car_av') != 0]
dl.data = dl.data[dl.get('PT_Walk_Access_av') != 0]
dl.data = dl.data[dl.get('PT_Park_Access_av') != 0]
dl.data = dl.data[dl.get('PT_Kiss_Access_av') != 0]
dl.data = dl.data[dl.get('Walk_av') != 0]
# dl.data = dl.data[dl.get('SP') != 0]
# dl.data = dl.data[(dl.get('PURPOSE') == 1) | (dl.get('PURPOSE') == 3)]
dl.print_data_info()
weights = dl.get('weight')
X, y = dl.get_X_and_y()
X /= 1000  # scale the costs and travel times

alternatives = np.array([0, 1, 2, 3, 4, 5], dtype=int_dtype)
nests = np.array([0, 1, 2], dtype=int_dtype)
nest_indices = np.array([0, 1, 2, 2, 2, 0], dtype=int_dtype)

# input_parameters = np.zeros(14)  # np.random.randn(14).astype(float_dtype)
# input_parameters[[11, 12, 13]] = 1
input_parameters = [
    -3.33116253, -3.45647051, -1.08080119, -1.65132431, -3.61162397,
    -3.02946516, -3.06731321, -1.10874701, -7.08716716, -4.28436567,
     1.33832895,  0.86023659,  1.        ,  2.12032055
]

utility_functions = np.array(
    [[0, 0, 0], [1, 0, 0],  # (feature, alternative, parameter)
     [2, 1, 1], [3, 1, 1],
     [4, 2, 2], [5, 2, 2],
     [6, 3, 3], [7, 3, 3],
     [8, 4, 4], [9, 4, 4],
     [10, 5, 5], [11, 5, 5]],
    dtype=int_dtype
)

biases = np.array([[0, 6], [1, 7], [2, 8], [3, 9], [4, 10]], dtype=int_dtype)
lambdas = np.array([[0, 11], [1, 12], [2, 13]], dtype=int_dtype)

W_input = np.zeros((X.shape[1], alternatives.shape[0]), dtype=float_dtype)  # rand
b_input = np.zeros_like(alternatives, dtype=float_dtype)
l_input = np.ones_like(nests, dtype=float_dtype)

cost, error = nl_calculations(X, y, biases, lambdas, input_parameters, utility_functions,
                              W_input, b_input, l_input)

print(cost)
print(error)