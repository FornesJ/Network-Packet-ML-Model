import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

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
    #support = cm.sum(dim=1).float()

    # calculate precision, recall and f1 score
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    if average == "macro":
        return precision.mean(), recall.mean(), f1.mean()
    
    elif average == "micro":
        # micro-F1 == accuracy for single-label multiclass
        accuracy = TP.sum() / cm.sum()
        return accuracy, accuracy, accuracy

    else:
        raise ValueError("average must be 'macro' or 'micro'!")
    

def multiclass_roc_auc(y_true, y_prob, average="macro"):
    """
    Function for multiclass_roc_auc
    :param y_true: [N]
    :param y_prob: [N, C]
    :param average: string
    """
    # one-hot encode labels
    y_true_bin = F.one_hot(
        y_true, num_classes=y_prob.size(1)
    ).numpy()

    # return roc auc score
    return roc_auc_score(
        y_true_bin,
        y_prob.numpy(),
        average=average,
        multi_class="ovr"
    )


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
    for avg in ["macro", "micro"]:
        p, r, f1 = precision_recall_f1(cm, average=avg)
        metrics[f"precision_{avg}"] = p.item()
        metrics[f"recall_{avg}"] = r.item()
        metrics[f"f1_{avg}"] = f1.item()

    # evaluate roc auc macro
    metrics["roc_auc_macro"] = multiclass_roc_auc(
        y_true, y_prob, average="macro"
    )

    return metrics


