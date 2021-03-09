import numpy as np


class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        num_test = x.shape[0]
        y_predict = np.zeros(num_test)
        for i in range(num_test):
            l1_distance = np.sum(np.sum(np.abs(self.x - x[i]), axis=1), axis=1)
            max_similarity_data_index = np.argmin(l1_distance)
            y_predict[i] = self.y[max_similarity_data_index]
        return y_predict
