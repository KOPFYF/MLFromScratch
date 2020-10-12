import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class LogisticRegression(object):
	"""
	docstring for LogisticRegression
	https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
	"""
	def __init__(self, learning_rate=0.001, n_iters=1000, print_training=False):
		super(LogisticRegression, self).__init__()
		self.lr = learning_rate
		self.n_iters = n_iters
		self.W = None
		self.b = None
		self.print_training = print_training

	def _sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def fit(self, X, y):
		'''
		X: n_samples by n_feature
		W: n_feature by 1
		y: n_samples by 1
		'''
		n_samples, n_feature = X.shape

		# Initilze parameters with random number
		self.W = np.random.rand(n_feature)
		self.W = self.W.astype('float128')
		self.b = np.random.rand(1)[0]
		self.b = self.b.astype('float128')

		# Gradient descent to minimize the log loss function
		for i in range(self.n_iters):
			y_pred = self._sigmoid(np.dot(X, self.W) + self.b)

			dW = (1 / n_samples) * np.dot(X.T, (y_pred - y))
			db = (1 / n_samples) * np.sum(y_pred - y)

			self.W -= self.lr * dW
			self.b -= self.lr * db

			if self.print_training:
				if i % 500 == 0:
					y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]
					f1 = f1_score(y, y_pred_binary)
					print(f"Training... f1 score in iteration {i} is {f1}")

	def predict(self, X):
		y_pred_prob = self._sigmoid(np.dot(X, self.W) + self.b)
		y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred_prob]
		return np.array(y_pred_binary)





