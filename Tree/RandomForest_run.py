import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from DecisionTree import DecisionTree
from RandomForest import RandomForest

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print('Shape:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

dt = DecisionTree(min_samples_split=3, max_depth=10, n_features=20)
dt.fit(X_train, y_train)

rf = RandomForest(n_trees=3, min_samples_split=3, max_depth=10, n_features=15)
rf.fit(X_train, y_train)

print('Parameters:', dt.min_samples_split, dt.max_depth, dt.n_features)
    
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print ("Accuracy score of Decision Tree:", acc_dt)
print ("Accuracy score of Random Forest:", acc_rf)
print ("F1 score of Decision Tree:", f1_dt)
print ("F1 score of Random Forest:", f1_rf) # random forest overfiting for this small data