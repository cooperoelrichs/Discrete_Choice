# Prototype logit estimation class
#
# This is a prototype class for estimating discrete choice logit models
# Python 3.4 64 bit with SciPy

from sklearn.linear_model import LogisticRegression


class LogitEstimator:
    """A prototype class for logit estimation"""
    def estimate_model(data_x, data_y):
        lr_n = LogisticRegression(dual=False, tol=0.0001,
                                  fit_intercept=True,
                                  class_weight='auto')
        lr_r = LogisticRegression(penalty='l1', dual=False, tol=0.0001,
                                  C=20, fit_intercept=True,
                                  class_weight='auto')

        lr_n.fit(data_x, data_y)
        lr_r.fit(data_x, data_y)
        print('Coefficients for MNL')
        print(lr_n.coef_)
        print('Coefficients for regularised MNL')
        print(lr_r.coef_)
