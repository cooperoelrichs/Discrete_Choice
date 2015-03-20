# Prototype logit estimation class
#
# This is a prototype class for estimating discrete choice logit models
# Python 3.4 64 bit with SciPy

from sklearn.linear_model import LogisticRegression


class LogitEstimator:
    """A prototype class for logit estimation"""
    def estimate_model(data_x, data_y):
        lr = LogisticRegression(penalty='l1', dual=False, tol=0.0001,
                                C=1.0, fit_intercept=True,
                                class_weight='auto')
        lr.fit(data_x, data_y)
        print(lr.coef_)
