import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from KNearestNeighbors import KNN

# The iris dataset is a classic and very easy multi-class classification dataset.
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape,)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=20)
plt.savefig('Original_Multiclass_dataset.png')

knn = KNN(3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("KNN classification accuracy", accuracy_score(y_test, y_pred))

plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], s=100*X_train[:, 2], c=y_train, cmap='viridis', alpha=0.2)
plt.scatter(X_test[:, 0], X_test[:, 1], s=100*X_test[:, 2], c=y_pred, cmap='viridis', alpha=0.2)
plt.savefig('KNN_visualization.png')
