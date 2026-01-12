import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np

def confusion_matrix(y_true, y_pred, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(cm, average="macro"):
    """
    Docstring for precision_recall_f1
    :param cm: [num_classes, num_classes]
    :param average: string
    """
    TP = torch.diag(cm).float()         # true positive
    FP = cm.sum(dim=0).float() - TP     # false positive
    FN = cm.sum(dim=1).float() - TP     # false negative
    support = cm.sum(dim=1).float()

    # calculate precision, recall and f1 score
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    if average == "macro":
        return precision.mean(), recall.mean(), f1.mean()
    
    elif average == "weighted":
        weights = support / support.sum() + 0.001
        return (
            (precision * weights).sum(),
            (recall * weights).sum(),
            (f1 * weights).sum()
        )
    
    elif average == "micro":
        # micro-F1 == accuracy for single-label multiclass
        accuracy = TP.sum() / cm.sum()
        return accuracy, accuracy, accuracy

    else:
        raise ValueError("average must be 'macro' or 'micro'!")
    
"""
def multiclass_roc_auc(y_true, y_prob, average="macro"):
    
    Function for multiclass_roc_auc
    :param y_true: [N]
    :param y_prob: [N, C]
    :param average: string
    
    num_classes = y_prob.size(1)
    # one-hot encode labels
    y_true_bin = F.one_hot(
        y_true, num_classes=num_classes
    ).numpy()

    # return roc auc score
    return roc_auc_score(
        y_true_bin,
        y_prob.numpy(),
        average=average,
        multi_class="ovr"
    )
"""

def multiclass_roc_auc(y_true, y_prob):
    """
    Function for multiclass_roc_curve
    :param y_true: [N]
    :param y_prob: [N, C]
    :param average: string
    """
    #Binarize labels
    num_classes = y_prob.size(1)
    y_true_bin = F.one_hot(
        y_true, num_classes=num_classes
    ).numpy()

    # Compute ROC per class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for c in range(num_classes):
        fpr[c], tpr[c], _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(num_classes)]))

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])

    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr["macro"], tpr["macro"], roc_auc["macro"]




def evaluate_metrics(y_true, y_pred, y_prob, num_classes):
    """
    Function for evaluating model precision, recall, f1 score and roc auc macro
    
    :param y_true (Tensor): true labels 
    :param y_pred (Tesnor): 
    :param y_prob: Description
    :param num_classes: Description
    """
    # confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=num_classes)

    metrics = {}

    # evaluate macro and micro precision, recall and f1 score
    for avg in ["macro", "weighted", "micro"]:
        p, r, f1 = precision_recall_f1(cm, average=avg)
        metrics[f"precision_{avg}"] = p.item()
        metrics[f"recall_{avg}"] = r.item()
        metrics[f"f1_{avg}"] = f1.item()

    # evaluate roc auc macro
    fpr, tpr, roc_auc = multiclass_roc_auc(y_true, y_prob)
    metrics["fpr_macro"] = fpr
    metrics["tpr_macro"] = tpr
    metrics["roc_auc_macro"] = roc_auc

    return metrics


