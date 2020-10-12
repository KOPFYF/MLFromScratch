import numpy as np

class NaiveBayes(object):
	"""
	docstring for NaiveBayes

	Posterior = likelihood * prior / evidence
	P(class|data) = (P(data|class) * P(class)) / P(data)
	we can ignore the marginal probability / denominator P(data)
	"""
	def __init__(self):
		super(NaiveBayes, self).__init__()
		# NBC is non-parametric model, so no need to init
		pass

	def fit(self, X, y):
		n_samples, n_features = X.shape
		# Multi-class
		self._classes = np.unique(y)
		n_classes = len(self._classes)

		# Calculate mean, variance, and priors for each class
		self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
		self._var = np.zeros((n_classes, n_features), dtype=np.float64)
		self._priors =  np.zeros(n_classes, dtype=np.float64)

		for i, c in enumerate(self._classes):
			Xc = X[y == c]
			self._mean[i, :] = Xc.mean(axis=0)
			self._var[i, :] = Xc.var(axis=0)
			self._priors[i] = Xc.shape[0] / float(n_samples)

	def predict(self, X):
		y_pred = [self._predict(x) for x in X]
		return np.array(y_pred)

	def _predict(self, x):
		posteriors = []

		# calculate posterior probability for each class
		for i, c in enumerate(self._classes):
			prior = np.log(self._priors[i])
			posterior = np.sum(np.log(self._pdf(i, x)))
			posterior = prior + posterior
			posteriors.append(posterior)

		# return class with highest posterior probability
		return self._classes[np.argmax(posteriors)]


	def _pdf(self, class_idx, x):
		# Gaussian Naive Bayes Classifier
		mean = self._mean[class_idx]
		var = self._var[class_idx]
		numerator = np.exp(- (x-mean)**2 / (2 * var))
		denominator = np.sqrt(2 * np.pi * var)
		return numerator / denominator




		
