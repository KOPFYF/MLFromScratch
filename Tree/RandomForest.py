import numpy as np
from collections import Counter
from DecisionTree import DecisionTree


class RandomForest(object):
	"""docstring for RandomForest"""
	def __init__(self, n_trees=5, min_samples_split=3, max_depth=8, n_features=None):
		super(RandomForest, self).__init__()
		self.n_trees = n_trees
		self.min_samples_split = min_samples_split
		self.max_depth = max_depth
		self.n_features = n_features
		self.trees = None


	def fit(self, X, y):
		self.trees = []
		n_samples, n_features = X.shape[0], X.shape[1]
		for _ in range(self.n_trees):
			dt =  DecisionTree(min_samples_split=self.min_samples_split,
                	max_depth=self.max_depth, n_features=self.n_features)

			idxs = np.random.choice(n_samples, n_samples, replace=True)
			dt.fit(X[idxs], y[idxs])
			self.trees.append(dt)


	def predict(self, X):
		tree_preds = np.array([tree.predict(X) for tree in self.trees])
		# print('tree_preds:', tree_preds, tree_preds.shape)
		tree_preds = np.swapaxes(tree_preds, 0, 1)

		# majority voting
		y_pred = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
		return np.array(y_pred)
