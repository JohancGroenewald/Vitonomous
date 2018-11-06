import numpy as np


class Classifications:
    IS_NAC = 0
    IS_PATH = 1
    IS_LIMIT = 2
    IS_ENVIRONMENT = 3

    @staticmethod
    def name(index):
        return ['NAC', 'PATH', 'LIMIT', 'ENVIRONMENT'][index]


class NearestNeighbor:
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
