import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class SVM(object):
	"""
	docstring for SVM

	https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47

	for all xi in training data
	xi.w + b <= -1   if yi = -1 (belongs to -ve class)
	xi.w + b >= +1	if yi = +1 (belongs to +ve class)
	or
	yi(xi.w+b) >= 1

	for all support vectors(SV) (data points which decides margin)
	xi.w+b = -1    here xi is -ve SV and yi is -1
	xi.w+b = +1    here xi is +ve SV and yi is +1

	decision Boundary:
	yi(xi.w+b)=0

	objective is to maximize Width W
	W = ((X+ - X-).w)/|w|, or we can say minimize |w|

	"""
	def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
		super(SVM, self).__init__()
		self.lr = learning_rate
		self.lambda_param = lambda_param
		self.n_iters = n_iters
		self.w = None
		self.b = None

	def fit(self, X, y):
		n_samples, n_features = X.shape
		# make binary labels
		y_ = np.where(y <= 0, -1, 1)

		# parameter inited as 0
		self.w = np.zeros(n_features)
		self.b = 0

		# Gradient descent with Hinge Loss & L2 regularization
		for _ in range(self.n_iters):
			for i, x_i in enumerate(X):
				flag = y_[i] * (np.dot(x_i, self.w) - self.b) >= 1
				if flag:
					# hinge loss is zero because it's correctly classified, only L2 term
					self.w -= self.lr * (2 * self.lambda_param * self.w)
				else:
					self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[i]))
					self.b -= self.lr * y_[i]

	def predict(self, X):
		y_pred_prob = np.dot(X, self.w) - self.b
		y_pred = np.sign(y_pred_prob)

		return y_pred

