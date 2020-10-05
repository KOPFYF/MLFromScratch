import numpy as np
from sklearn.metrics import mean_squared_error


class LinearRegression(object):
	"""
	docstring for LinearRegression
	https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931
	"""
	def __init__(self, learning_rate=0.01, n_iters=1000, print_mse=False):
		super(LinearRegression, self).__init__()
		self.learning_rate = learning_rate
		self.n_iters = n_iters
		self.print_mse = print_mse
		self.W = None
		self.b = None

	def fit(self, X, y):
		n_samples, n_features = X.shape

		# Init parameters
		self.W = np.zeros(n_features)
		self.b = 0

		# Gradient descent
		for iter in range(self.n_iters):
			y_pred = np.dot(X, self.W) + self.b

			# Compute gradient
			# MSE = 1/n * (y - (WX + b)) ^ 2
			dW = (1 / n_samples) * np.dot(X.T, (y_pred - y))
			db = (1 / n_samples) * np.sum(y_pred - y)

			# Update paras
			self.W -= self.learning_rate * dW
			self.b -= self.learning_rate * db

			# Print MSE
			if self.print_mse and iter % 100 == 0:
				y_pred = np.dot(X, self.W) + self.b
				mse = mean_squared_error(y, y_pred)
				print(f"In iteration {iter} with Mean Squared Error = {mse}")

	def predict(self, X):
		return np.dot(X, self.W) + self.b


		