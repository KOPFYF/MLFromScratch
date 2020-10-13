import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from DecisionTree import DecisionTree

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print('Shape:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

dt = DecisionTree(min_samples_split=3, max_depth=10, n_features=20)
dt.fit(X_train, y_train)

print('Parameters:', dt.min_samples_split, dt.max_depth, dt.n_features)
    
y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print ("Accuracy score:", acc)
print ("F1 score:", f1)