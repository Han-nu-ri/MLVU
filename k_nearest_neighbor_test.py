import unittest
from k_nearest_neighbor import KNearestNeighbor
import numpy as np


class KNearestNeighborTest(unittest.TestCase):
    @staticmethod
    def get_train_data():
        x = np.array([
            [[10, 10],
             [10, 10]],
            [[15, 15],
             [15, 15]],
            [[-12, -12],
             [-12, -12]],
            [[-15, -15],
             [-15, -15]]
        ])
        y = np.array([1, 1, 0, 0])
        return x, y

    @staticmethod
    def get_test_data():
        x = np.array([
            [[8, 8],
             [8, 8]],
            [[-10, -10],
             [-10, -10]]
        ])
        y = np.array([1, 0])
        return x, y

    def test_train(self):
        knn_model = KNearestNeighbor()
        x_train, y_train = self.get_train_data()
        knn_model.train(x_train, y_train)
        self.assertEqual(np.testing.assert_array_equal(x_train, knn_model.x), None)
        self.assertEqual(np.testing.assert_array_equal(y_train, knn_model.y), None)

    def test_predict(self):
        knn_model = KNearestNeighbor()
        x_train, y_train = self.get_train_data()
        knn_model.train(x_train, y_train)
        x_test, y_test = self.get_test_data()
        y_actual = knn_model.predict(x_test)
        self.assertEqual(np.testing.assert_array_equal(y_actual, y_test), None)


if __name__ == '__main__':
    unittest.main()
