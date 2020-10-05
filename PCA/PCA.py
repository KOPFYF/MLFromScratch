import numpy as np

class PCA:

	def __init__(self, n_components):
		self.n_components = n_components
		self.mean = None

	
	def fit(self, X):
		'''
		X: m * n (m rows & n features)
		Covariance Matrix: n * n
		Eigenvalues: 1 * n
		Eigenvectors: n * n
		self.components: self.n_components * n

		'''
		# Normalization
		self.mean = np.mean(X, axis=0)
		X = X - self.mean

		# Empirical sample covariance matrix of X
		cov = np.cov(X.T)

		# Eigenvalues & Eigenvectors, cov * eigenvectors = eigenvalues * eigenvectors
		eigenvalues, eigenvectors = np.linalg.eig(cov)

		# Sort eigenvectors
		eigenvectors = eigenvectors.T
		idxs = np.argsort(eigenvalues)[::-1]
		eigenvalues = eigenvalues[idxs]
		eigenvectors = eigenvectors[idxs]
		self.components = eigenvectors[0: self.n_components]


	def transform(self, X):
		# project data
		X = X - self.mean
		return np.dot(X, self.components.T) # m by n times n by n_components

