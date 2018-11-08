import pickle

import numpy as np


class Classifications:
    IS_NAC = 0
    IS_CLASS_1 = 1
    IS_CLASS_2 = 2
    IS_CLASS_3 = 3
    IS_CLASS_4 = 4
    IS_CLASS_5 = 5
    IS_CLASS_6 = 6
    IS_CLASS_7 = 7
    IS_CLASS_8 = 8
    IS_CLASS_9 = 9
    IS_CLASS_10 = 10
    IS_CLASS_11 = 11
    IS_CLASS_12 = 12
    IS_CLASS_13 = 13
    IS_CLASS_14 = 14
    IS_CLASS_15 = 15
    IS_CLASS_16 = 16
    IS_CLASS_17 = 17
    IS_CLASS_18 = 18
    IS_CLASS_19 = 19
    IS_CLASS_20 = 20

    @staticmethod
    def name(index):
        return 'NAC' if index == 0 else 'CLASS_{}'.format(index)


class NearestNeighbor:
    FILE_NAME = 'nearest_neighbor.pickle'

    def __init__(self):
        self.Xtr = None
        self.ytr = None

    def __len__(self):
        return 0 if self.Xtr is None else self.Xtr.size

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        for i in range(num_test):
            distance = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distance)
            Ypred[i] = self.ytr[min_index]
        return Ypred

    def save(self):
        with open(self.FILE_NAME,'wb') as outfile:
            pickle.dump(self.Xtr, outfile)
            pickle.dump(self.ytr, outfile)

    def load(self):
        with open(self.FILE_NAME,'rb') as infile:
            self.Xtr = pickle.load(infile)
            self.ytr = pickle.load(infile)
