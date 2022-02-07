'''
手写precision, recall

电话面试， 题目是用python手撸一个P/R出来，楼主从没写过。已挂。
补充一下。PR的意思是p‍‍‍‌‌‍recision recall.  输入是label和probability

iterate 不同的threshold 算 precision-recall 就好了


'''

import numpy as np
from sklearn.metrics import precision_recall_curve

def roc_curve(y_trues, y_probs, n_partitions=10):
    '''
    TPR = TP / (TP + FN) = Recall
    FPR = FP / (FP + TN)
    '''
    y_trues = np.array(y_trues)
    y_probs = np.array(y_probs)

    threds = np.array([1.0 * i * y_probs.max() + 0.001 for i in range(n_partitions + 1)]) / n_partitions
    print(len(y_trues), len(threds), threds)


    fpr, tpr = [], []
    for thred in threds:
        y_preds = (y_probs >= thred).astype(int)
        matching = (y_preds == y_trues).astype(int)
        n_tp = (matching * y_trues).sum() # y_true is 1 and predict 1
        n_tn = (matching * (1 - y_trues)).sum() # y_true is 0 and predict 0
        n_fp = ((1 - matching) * y_preds).sum() # y_true is 0 and predict 1
        n_fn = ((1 - matching) * y_trues).sum() # y_true is 1 and predict 0
        n_p = y_trues.sum()
        n_n = (1 - y_trues).sum()

        fpr.append(1.0 * n_fp / n_n)
        tpr.append(1.0 * n_tp / n_p)

    print('roc_curve')
    print(np.array(fpr), np.array(tpr), threds)
    return np.array(fpr), np.array(tpr), threds


def precision_recall(y_trues, y_probs, n_partitions=10):
    '''
    Precision = TP / (FP + TP)
    Recall    = TP / (FN + TP)
    '''
    y_trues = np.array(y_trues)
    y_probs = np.array(y_probs)

    threds = np.array([1.0 * i * y_probs.max() + 0.001 for i in range(n_partitions + 1)]) / n_partitions
    print(len(y_trues), len(threds), threds)

    precison, recall = [], []
    for thred in threds:
        y_preds = (y_probs >= thred).astype(int)
        matching = (y_preds == y_trues).astype(int)
        n_tp = (matching * y_trues).sum() # y_true is 1 and predict 1
        n_tn = (matching * (1 - y_trues)).sum() # y_true is 0 and predict 0
        n_fp = ((1 - matching) * y_preds).sum() # y_true is 0 and predict 1
        n_fn = ((1 - matching) * y_trues).sum() # y_true is 1 and predict 0
        n_p = y_trues.sum()
        n_n = (1 - y_trues).sum()

        precison.append(1.0 * n_tp / (n_tp + n_fp))
        recall.append(1.0 * n_tp / (n_tp + n_fn))
       
    print('precision_recall')
    print(np.array(precison), np.array(recall), threds) 
    return np.array(precison), np.array(recall), threds


y_trues = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]
precision, recall, thresholds = precision_recall_curve(y_trues, y_scores)

# print(precision, recall, thresholds) # [0.66666667 0.5        1.         1.        ] [1.  0.5 0.5 0. ] [0.35 0.4  0.8 ]


roc_curve(y_trues, y_scores, 3) # [1.  0.5 0.  0. ] [1.  1.  0.5 0. ] [3.33333333e-04 2.67000000e-01 5.33666667e-01 8.00333333e-01]
precision_recall(y_trues, y_scores, 3) # [0.5        0.66666667 1.                nan] [1.  1.  0.5 0. ] [3.33333333e-04 2.67000000e-01 5.33666667e-01 8.00333333e-01]









