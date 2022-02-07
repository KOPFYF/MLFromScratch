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
		precisions, recalls = [], []

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
# print(pr.precision_recall(y_trues, y_probs, 3))



class LogisticRegression:
	def __init__(self, lr=0.01, n_iters=1000):
		'''
		X : (n_samples, n_features)
		Y:  (n_samples, 1)
		W:  (n_features, 1)
		b:  (1, 1)
		Z = W*X + b (n_features, 1)
		Y = sigmoid(Z) = sigmoid(W*X + b)
		dY/dW = 1 / n_samples *  X^T (y_pred - y) = (n_features, 1)
		dY/db = 1 / n_samples * (y_pred - y)


		'''
		self.W = None
		self.b = None
		self.lr = lr 
		self.n_iters = n_iters

	def _sigmoid(self, x):
		return 1 / (1 + np.exp(-x))


	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.W = np.random.rand(n_features)
		self.b = np.random.rand(1)

		for i in range(self.n_iters):
			y_pred = self._sigmoid(np.dot(X, self.W) + self.b)

			dW = 1 / n_samples *  np.dot(X.T, (y_pred - y))
			db = 1 / n_samples *  np.sum((y_pred - y))

			self.W -= dW * self.lr
			self.b -= db * self.lr

			# could add some early stop based on precision, recall, etc


	def predict_prob(self, X):
		return self._sigmoid(np.dot(X, self.W) + self.b)

	def predict(self, X):
		y_probs = self._sigmoid(np.dot(X, self.W) + self.b)
		y_preds = (y_probs > 0.5).astype(int)
		return y_preds


class KNN:
	# K nearest neighbor. prediction based on the neighbors
	def __init__(self, k=3):
		self.k = k

	def fit(self, X, y):
		self.X_train = X 
		self.y_train = y 


	def predict(self, X):
		y_pred = [self._predict(x) for x in X]


	def euclidian_dist(self, x1, x2):
		return np.sqrt(np.sum(x1 - x2) ** 2)


	def _predict(self, x):
		dists = [self.euclidian_dist(x, x_train) for x_train in self.X_train]
		idxs = np.argsorts(dists)[:self.k]

		labels = list(self.y_train[idxs])
		freqs = collections.Counter(labels)

		return freqs.most_common(1)[0][0]



class PCA:
	def __init__(self, n_components):
		self.n_components = n_components
		self.mean = None 

	def fit(self, X):
		'''
		X: m * n ( samples * features)
		Cov: n * n, X.T * X
		eigenvalues: n * 1
		eigenvectors: n * n


		'''
		# Normalization
		self.mean = np.mean(X, axis=0)
		X -= self.mean

		# decompose COV
		cov = np.cov(X.T)
		eigenvalues, eigenvectors = np.linalg.eig(cov)

		# sort eigens
		idxs = np.argsort(eigenvalues)[::-1]
		eigenvalues = eigenvalues[idxs]
		eigenvectors = eigenvectors[idxs]
		self.components = eigenvectors[:self.n_components]


	def transform(self, X):
		# X: m * n
		# self.components: n * n_components
		X -= self.mean
		return np.dot(X, self.components.T) 































