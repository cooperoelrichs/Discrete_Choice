import numpy as np


class NLDataLoader(object):
    def __init__(self, file_dir, delimiter, X_columns, y_column):
        self.file_dir = file_dir
        self. X_columns = X_columns
        self.y_column = y_column
        self.delimiter = delimiter

        self.data, self.original_shape = self.load()
        self.headers = self.load_headers()
        self.header_indices = self.generate_header_indices()

    def load(self):
        data = np.genfromtxt(self.file_dir, delimiter=self.delimiter, skip_header=1)
        return data, data.shape

    def load_headers(self):
        return open(self.file_dir, 'r').readline().rstrip().split(self.delimiter)

    def generate_header_indices(self):
        return dict([(name, i) for i, name in enumerate(self.headers)])

    def get(self, name):
        return self.data[:, self.header_indices[name]]

    def get_X_and_y(self):
        X_indices = [self.header_indices[name] for name in self.X_columns]
        y_index = self.header_indices[self.y_column]
        return self.data[:, X_indices], self.data[:, y_index]

    def print_data_info(self):
        print('Original data shape: ' + str(self.original_shape))
        print('Current data shape: ' + str(self.data.shape))
        print('Current X data shape: ' + str(self.get_X_and_y()[0].shape))
        print('Unique y values: ' + str(np.unique(self.get_X_and_y()[1])))
