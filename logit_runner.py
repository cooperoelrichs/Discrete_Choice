# Script for running the logit_estimator
#
# Python 3.4 64 bit with SciPy

import numpy
import csv
from logit_estimator import LogitEstimator
# import pandas

project_location = 'D:\Cooper\Python\Discrete_Choice'
data_csv_file = project_location + '\\' + 'data.csv'

# Reading the csv using Pandas might be faster
# data_mat = pandas.io.parsers.read_csv(data_csv_file)

with open(data_csv_file) as file:
    reader = csv.reader(file)
    headers = next(reader)

data_mat = numpy.loadtxt(open(data_csv_file, 'rb'),
                         delimiter=',', skiprows=1, dtype='float')

print(headers)

# class_numbers = numpy.transpose(numpy.matrix([1, 2, 3]))
data_x = data_mat[:, [1, 2, 3, 4, 5, 6, 7]]
data_y = data_mat[:, 7]

# print(data_x)
# print(data_y)

logit_model = LogitEstimator.estimate_scikit_learn_model(data_x, data_y)
logit_model = LogitEstimator.estimate_logit_model(data_x, data_y)
