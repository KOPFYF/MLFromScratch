import numpy as np 

class LDA(object):
	"""docstring for LDA"""
	def __init__(self, n_components):
		super(LDA, self).__init__()
		self.n_components = n_components
		self.linear_discriminants = None 

	def fit(self, X, y):
		n_features = X.shape[1] # 150 by 4
		class_labels = np.unique(y)

		# S_W: within group, S_B, between group
		mean_overall = np.mean(X, axis=0)
		S_W = np.zeros((n_features, n_features)) # 4 by 4
		S_B = np.zeros((n_features, n_features)) # 4 by 4
		for c in class_labels:
			X_c = X[y == c]
			mean_c = np.mean(X_c, axis=0)
			# 4 by n_c, n_c by 4
			S_W += (X_c - mean_c).T.dot(X_c - mean_c)
			# 4 by 1, 1 by 4
			n_c = X_c.shape[0]
			mean_diff = (mean_c - mean_overall).reshape(n_features, 1) # (4, 1)
			S_B += n_c * (mean_diff).dot(mean_diff.T)

		A = np.linalg.inv(S_W).dot(S_B)
		eigenvalues, eigenvectors = np.linalg.eig(A)
		eigenvectors = eigenvectors.T
		idxs = np.argsort(abs(eigenvalues))[::-1]
		eigenvalues, eigenvectors = eigenvalues[idxs],eigenvectors[idxs]
		self.linear_discriminants = eigenvectors[:self.n_components]


	def transform(self, X):
		# project the data
		return np.dot(X, self.linear_discriminants.T)
