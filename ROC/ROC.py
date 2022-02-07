from typing import Sequence, Tuple
import numpy as np


def roc_curve(
    y_trues: Sequence[int], y_probs: Sequence[float], n_partitions: int = 10
    ) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
    """Description
    Assuming binary label and positive class is 1.
    Args:
        XXX -
        XXX -
    Return:
        XXX -
    """
    assert len(y_trues) == len(y_probs), "y_true should have the same length as y_prob"
    assert (
        len(set(y_trues)) <= 2
    ), "More than 2 classes detected! This vanilla ROC only supports binary classification." 

    y_trues = np.array(y_trues)
    y_probs = np.array(y_probs)
    threds = (
        np.array(
            [1.0 * i * (y_probs.max() + 0.001) for i in range(0, n_partitions + 1)]
        )
        / n_partitions
    )
    
    fpr, tpr = [], []

    for thred in threds:
        y_preds = (y_probs >= thred).astype(int)
        matching = (y_preds == y_trues).astype(int)
        n_tp = (matching * y_trues).sum()
        n_fp = ((1 - matching) * y_preds).sum()
        n_p = y_trues.sum()
        n_n = (1 - y_trues).sum()
        fpr.append(1.0 * n_fp / n_n)
        tpr.append(1.0 * n_tp / n_p)
    return np.array(fpr), np.array(tpr), threds


def auc(fpr: Sequence[float], tpr: Sequence[float]) -> float:
    return 0.5 * (np.multiply(tpr[1:] + tpr[:-1], fpr[:-1] - fpr[1:])).sum()




# https://towardsdatascience.com/roc-curve-and-auc-from-scratch-in-numpy-visualized-2612bb9459ab

def true_false_positive(threshold_vector, y_test):
    true_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 1)
    true_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 0)
    false_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 0)
    false_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr


def roc_from_scratch(probabilities, y_test, partitions=100):
    roc = np.array([])
    for i in range(partitions + 1):
        
        threshold_vector = np.greater_equal(probabilities, i / partitions).astype(int)
        tpr, fpr = true_false_positive(threshold_vector, y_test)
        roc = np.append(roc, [fpr, tpr])
        
    return roc.reshape(-1, 2)



import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
plt.figure(figsize=(15,7))

ROC = roc_from_scratch(prob_vector,y_test,partitions=10)
plt.scatter(ROC[:,0],ROC[:,1],color='#0F9D58',s=100)
plt.title('ROC Curve',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=16)
plt.ylabel('True Positive Rate',fontsize=16)


from celluloid import Camera
camera = Camera(plt.figure(figsize=(17,9)))
for i in range(30):
    ROC = roc_from_scratch(prob_vector,y_test,partitions=(i+1)*5)
    plt.scatter(ROC[:,0],ROC[:,1],color='#0F9D58',s=100)
    plt.title('ROC Curve',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    camera.snap()
anim = camera.animate(blit=True,interval=300)
anim.save('scatter.gif')


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, prob_vector)

plt.figure(figsize=(15, 7))
plt.scatter(fpr, tpr, s=100, alpha=0.5, color="blue", label="Scikit-learn")
plt.scatter(
    ROC[:, 0], ROC[:, 1], color="red", s=100, alpha=0.3, label="Our implementation"
)
plt.title("ROC Curve", fontsize=20)
plt.xlabel("False Positive Rate", fontsize=16)
plt.ylabel("True Positive Rate", fontsize=16)
plt.legend()