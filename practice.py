import numpy as np
import random

class PR_CURVE:
	def precision_recall(self, y_trues, y_probs, n_partitions=10):
		'''
		   1  0 (ground truth)
		1  TP FP
		0  FN TN
		precision = tp / (tp + fp)
		recall = tpr = tp / (tp + fn)

		'''
		y_trues = np.array(y_trues)
		y_probs = np.array(y_probs)

		thresholds = np.array([1.0 * i * y_probs.max() / n_partitions for i in range(n_partitions)])
		precisions, recalls = []

		for threshold in thresholds:
			y_preds = (y_probs >= threshold).astype(int)
			matching = (y_preds == y_trues).astype(int)
			tp = (matching * y_preds).sum()
			tn = (matching * (1- y_preds)).sum()
			fp = ((1 - matching) * y_preds).sum()
			fn = ((1 - matching) * y_trues).sum()

			precision, recall = tp / (tp + fp), tp / (tp + fn)
			precisions.append(precision)
			recalls.append(recall)

		return precisions, recalls, thresholds

pr = PR_CURVE()
y_trues = [0, 0, 1, 1]
y_probs = [0.1, 0.4, 0.35, 0.8]
print(pr.precision_recall(y_trues, y_probs, 3))


class LogisticRegression:
	def __init__(self, lr=0.01, n_iters=1000):
		self.lr = lr 
		self.n_iters = n_iters
		self.W = None
		self.b = None

	def _sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def fit(self, X, y):
		'''
		X: n_samples * n_features
		W: n_features * 1
		b: 1 * 1
		y: n_samples * 1

		y = XW + b

		'''
		n_samples, n_features = X.shape

		self.W = np.random.rand(n_features)
		self.b = random.random()

		for i in range(self.n_iters):
			# linear regression is almost the same, except that no sigmoid here
			y_pred = self._sigmoid(np.dot(X, self.W) + self.b)

			dW = (1 / n_samples) * np.dot(X.T, (y_pred - y)) # n_features * 1
			db = (1 / n_samples) * np.sum(y_pred - y)

			self.W -= self.lr * dW
			self.b -= self.lr * db

	def predict(self, X):
		y_pred = self._sigmoid(np.dot(X, self.W) + self.b)
		y_pred_label = [int(p > 0.5) for p in y_pred]
		return np.array(y_pred_label)



class KNN:
	# K nearest neighbor. prediction based on the neighbors
	def __init__(self, k=3):
		self.k = k

	def fit(self, X, y):
		self.X_train = X 
		self.y_train = y 

	def predict(self, X):
		y_pred = [self._predict(x) for x in X]
		return np.array(y_pred)

	def l2norm(self, x1, x2):
		return np.sqrt(np.sum((x1 - x2) ** 2))


	def _predict(self, x):
		# predict a single data point
		distances = [l2norm(x, x_train) for x_train in self.X_train]

		# K nearest neighbor
		idxs = np.argsort(distances)[:self.k] # get top k indexes
		k_labels = list(self.y_train[idxs])

		# majority vote
		freqs = collections.Counter(k_labels)
		return freqs.most_common(1)[0][0]



class KMeans:
	'''
	clustering:

	1. init k centroids 
	2. for each datapoint, find the closest centroid and assign it to that group
	3. update the centroids using within group data untill it converges or early stop
	'''


	def __init__(self, k, max_iters):
		self.k = k 
		self.max_iters = max_iters
		self.clusters = [[] for _ in range(self.k)] # list of group, len=k
		self.centroids = [] # len=k

	def predict(self, X):
		self.X = X
		self.n_samples, self.n_features = X.shape

		random_sample_k = np.random.choice(self.n_samples, self.k = k, replace=False)
		self.centroids = [self.X[idx] for idx in random_sample_k]

		for i in range(self.max_iters):
			self.clusters = self._create_cluster(self.centroids)

			prev_centroids = self.centroids
			self.centroids = self._centroid_update(self.centroids)
			if self._converge(prev_centroids, self.centroids):
				break

		return self._get_cluster_labels(self.clusters)


	def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        pass

	def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        pass

    def _closest_centroid(self, sample, centroids):
        # get the index of the closest centroid of the current sample 
        pass

    def _centroid_update(self, clusters):
        # assign mean value of clusters to centroids









