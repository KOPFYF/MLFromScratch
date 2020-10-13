import numpy as np
from collections import Counter

class Node(object):
	"""docstring for Node"""
	def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
		super(Node, self).__init__()
		self.feature = feature
		self.threshold = threshold
		self.left = left
		self.right = right
		self.value = value

	def is_leaf_node(self):
		return self.value is not None


class DecisionTree(object):
	"""docstring for DecisionTree"""
	def __init__(self, min_samples_split=3, max_depth=8, n_features=None):
		super(DecisionTree, self).__init__()
		self.min_samples_split = min_samples_split
		self.max_depth = max_depth
		self.n_features = n_features
		self.root = None


	def entropy(self, y):
		hist = np.bincount(y)
		prob = hist / len(y)

		return -np.sum([p * np.log2(p) for p in prob if p > 0])


	def fit(self, X, y):
		if not self.n_features:
			self.n_features = X.shape[1] 
		else:
			self.n_features = min(self.n_features, X.shape[1])
		self.root = self._grow_tree(X, y)


	def predict(self, X):
		return np.array([self._traverse_tree(x, self.root) for x in X])


	def _grow_tree(self, X, y, depth=0):
		# DFS to grow the tree recursively
		n_samples, n_features = X.shape
		n_labels = len(np.unique(y))

		# stopping criteria to prevent overfitting
		if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
			leaf_value = Counter(y).most_common(1)[0][0]
			return Node(value=leaf_value)

		# greedy search the best split by info gain
		feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
		best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

		# recursively grow the tree until meeting stopping criteria
		left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
		left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
		right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

		# return current Node
		return Node(best_feat, best_thresh, left, right)


	def _best_criteria(self, X, y, feat_idxs):
		# greedy search function
		best_gain = 0
		split_idx, split_thresh = None, None
		for i in feat_idxs:
			# loop through all features
			X_i = X[:, i]
			thresholds = np.unique(X_i)
			for threshold in thresholds:
				# loop through all cutoffs, whatever the feature is categorical or numerical
				# definitely can be improved here, this is really costly
				gain = self._information_gain(y, X_i, threshold)
				if gain > best_gain:
					best_gain = gain
					split_idx = i
					split_thresh = threshold

		return split_idx, split_thresh


	def _information_gain(self, y, X_i, split_thresh):
		# calculate info gain based on entropy

		parent_entropy = self.entropy(y)

		left_idxs, right_idxs = self._split(X_i, split_thresh)
		if len(left_idxs) == 0 or len(right_idxs) == 0:
			return 0 # pure in this case

		children_entropy = 1 / len(y) * (len(left_idxs) * self.entropy(y[left_idxs]) + \
										len(right_idxs) * self.entropy(y[right_idxs]))

		return parent_entropy - children_entropy


	def _split(self, X_i, split_thresh):
		# return split indexes
		left_idxs = np.argwhere(X_i <= split_thresh).flatten()
		right_idxs = np.argwhere(X_i > split_thresh).flatten()

		return left_idxs, right_idxs


	def _traverse_tree(self, x, node):
		if node.is_leaf_node():
			return node.value

		if x[node.feature] <= node.threshold:
			return self._traverse_tree(x, node.left)
		else:
			return self._traverse_tree(x, node.right)





