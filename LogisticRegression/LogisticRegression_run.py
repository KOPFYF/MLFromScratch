import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from LogisticRegression import LogisticRegression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

LR = LogisticRegression(learning_rate=0.001, n_iters=10000, print_training=True)
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)

print("LR classification accuracy:", accuracy_score(y_test, y_pred))
print("LR classification f1 score:", f1_score(y_test, y_pred))

