# https://github.com/geodra/Articles/blob/master/K-Means_scratch.ipynb

import numpy as np
from scipy.spatial.distance import cdist 

# def euclidean_distance(x1, x2):
#     return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, X, k=2, max_iters=100):
        self.X = X 
        self.k = k 
        self.max_iters = max_iters

    def fit(self):
        '''
        1. random select k points as starting centroids, with no replacement
        2. calculate dist between each point and assign it to cloest centroids
        3. for each group, recalculate the centroids
        4. repeat 2-3 until it converges
        '''

        idx = np.random.choice(len(self.X), self.k, replace=False)
        centroids = self.X[idx, :]

        distances = cdist(self.X, centroids, 'euclidean')
        points = np.argmin(distances, axis=1) # labels

        for step in range(self.max_iters):
            prev_points = points
            centroids = []
            for label in range(self.k):
                sub_group = self.X[points == label]
                centroids.append(sub_group.mean(axis=0)) # mean on all points

            centroids = np.vstack(centroids)
            distances = cdist(self.X, centroids, 'euclidean')
            points = np.argmin(distances, axis=1) # labels
            if np.array_equal(prev_points, points):
                print('early stop at step {}'.format(step))
                break

        print('final labels:', points)
        # print('final centroids', centroids)
        self.centroids = centroids

        return points

    def predict(self, x):
        distances = cdist(x, self.centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        print('predicted labels:', labels)
        return labels




X = np.array([[1, 2], [1, 4], [1, 0], [1, 2], [10, 5],[10, 2], [10, 4], [10, 0]])
kmeans = KMeans(X)
kmeans.fit()

kmeans.predict([[0, 0], [12, 3]])



# from sklearn.datasets import load_digits
# data = load_digits().data
# kmeans2 = KMeans(data, 10)

# label = kmeans2.fit()
# print(len(label))




