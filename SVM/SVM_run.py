import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from SVM import SVM

X, y =  datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.5, random_state=42)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVM()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print("SVM classification accuracy:", accuracy_score(y_test, y_pred))
print("SVM classification f1 score:", f1_score(y_test, y_pred))


def visualize_svm():
    '''
    xi.w+b = -1    here xi is -ve SV and yi is -1
    xi.w+b = +1    here xi is +ve SV and yi is +1
    
    decision Boundary:
    yi(xi.w+b)=0
    '''
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X[:,0], X[:,1], marker='o',c=y)

    # plot lines based on 2 ends
    x0_1 = np.amin(X[:,0])
    x0_2 = np.amax(X[:,0])

    # Decision boundary
    x1_1 = get_hyperplane_value(x0_1, svm.w, svm.b, 0)
    x1_2 = get_hyperplane_value(x0_2, svm.w, svm.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, svm.w, svm.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, svm.w, svm.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, svm.w, svm.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, svm.w, svm.b, 1)
    
    # Decision boundary
    ax.plot([x0_1, x0_2],[x1_1, x1_2], 'y--')

    # Support vector lines
    # not necessary that support vector lines always pass through support vectors
    ax.plot([x0_1, x0_2],[x1_1_m, x1_2_m], 'k')
    ax.plot([x0_1, x0_2],[x1_1_p, x1_2_p], 'k')
    x1_min = np.amin(X[:,1])
    x1_max = np.amax(X[:,1])
    ax.set_ylim([x1_min - 3, x1_max + 3])
    plt.savefig('SVM_visualization.png')

visualize_svm()