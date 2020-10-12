import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from NaiveBayes import NaiveBayes

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nbc = NaiveBayes()
nbc.fit(X_train, y_train)
y_pred = nbc.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("accuracy_score:", acc)
print("f1 score:", f1)