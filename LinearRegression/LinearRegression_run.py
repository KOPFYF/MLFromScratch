import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import matplotlib.pyplot as plt

from LinearRegression import LinearRegression

X, y = datasets.make_regression(n_samples=50, n_features=1, noise=20, random_state=4)
print('X shape: ', X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LR = LinearRegression(learning_rate=0.01, n_iters=1000)
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

y_pred_line = LR.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.2), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.9), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.savefig('LinearRegression_visulization.png')