import numpy as np
from sklearn.metrics import (accuracy_score, 
                             confusion_matrix, 
                             precision_score, 
                             recall_score, 
                             roc_auc_score,
                             roc_curve,
                             precision_recall_curve, 
                             auc,)
import math

def auprc(y_true, predictions):
    precision, recall, _ = precision_recall_curve(y_true, predictions)
    return (auc(recall, precision))

def sensitivity(y_true, predictions):
    fpr, tpr, threshold = roc_curve(y_true,predictions)

    gmeans_square = tpr * (1 - fpr)
    gmeans = [math.sqrt(x) for x in gmeans_square]
    optimal_idx = np.argmax(gmeans)

    optimal_threshold = threshold[optimal_idx]

    predictions = [0 if item < optimal_threshold else 1 for item in predictions]
    tn, fp, fn, tp = confusion_matrix(y_true,predictions).ravel()
    return tp/(tp+fn) 

def specificity(y_true, predictions):
    fpr, tpr, threshold = roc_curve(y_true,predictions)

    gmeans_square = tpr * (1 - fpr)
    gmeans = [math.sqrt(x) for x in gmeans_square]
    optimal_idx = np.argmax(gmeans)

    optimal_threshold = threshold[optimal_idx]

    predictions = [0 if item < optimal_threshold else 1 for item in predictions]
    tn, fp, fn, tp = confusion_matrix(y_true,predictions).ravel()
    return tn/(tn+fp)
