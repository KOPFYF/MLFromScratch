import numpy as np 
from collections import Counter


class KNN(object):
    """docstring for KNN"""
    def __init__(self, k=2):
        super(KNN, self).__init__()
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):

    	# Define distance metric, we can also use weighted, that is inverse of distance to do majority voting
        def euclidean_distance(x1, x2):
        	# return np.linalg.norm(x1 - x2)
            return np.sqrt(np.sum((x1 - x2)**2))

        # Compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Sort by distances and get k nearest neighbors
        idxs = np.argsort(distances)[:self.k]
        k_neighbor_labels = list(self.y_train[idxs])

        # Majority voting
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0] 

        