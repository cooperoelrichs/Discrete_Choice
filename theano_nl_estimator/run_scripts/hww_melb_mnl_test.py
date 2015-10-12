import numpy as np
import theano
import theano.tensor as T
from nl_data_loader.nl_data_loader import NLDataLoader
from scipy import optimize
import time

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
X, y = dl.get_X_and_y()
X /= 1000  # scale the costs

class MNL(object):
    def __init__(self, X, y, W, b):
        self.X = X
        self.y = y
        self.W = W
        self.b = b

        self.cost_function, self.grad_function = self.cost_and_grad()

    @staticmethod
    def cost_and_grad():
        X = T.matrix('X', dtype=float_dtype)
        y = T.vector('y', dtype=int_dtype)
        W = T.matrix('W_input', dtype=float_dtype)
        b = T.vector('b_input', dtype=float_dtype)

        p_y_given_x = T.nnet.softmax(T.dot(input, W) + b)
        y_pred = T.argmax(p_y_given_x, axis=1)

        cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
        errors = T.mean(T.neq(y_pred, y))

        cost_function = theano.function(inputs=[X, y, W, b], outputs=[cost, errors])
        grad_function = theano.function(inputs=[X, y, W, b], outputs=[T.grad(cost, wrt=[W, b])])

        return cost_function, grad_function

    def cost(self, parameters):
        W, b = np.unravel(parameters)
        self.cost_function(self.X, self.y, )
        return np.ravel([W, b])

    def grad(self, parameters):
        W, b = np.unravel(parameters)
        self.grad_function(self.X, self.y, )
        return np.ravel([W, b])

    def estimate(self):
        parameters = np.ravel([self.W, self.b])
        parameters = optimize.fmin_bfgs(self.cost,
                                        parameters,
                                        fprime=self.grad,
                                        gtol=0.0000001, disp=True)

W = np.zeros((X.shape[1], y.shape[1]))
b = np.zeros(X.shape[1])
mnl = MNL(X, y, W, b)

start_time = time.clock()
cost, error, W, b = mnl.estimate()
end_time = time.clock()

print(X)
print(b)
print('Estimate time: %.2f' % (end_time - start_time))
print('Cost is: %.2f' % cost)
print('Accuracy is: %.2f' % (1 - error))
