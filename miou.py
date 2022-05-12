import numpy as np


def MIOU(GT, Pred, num_class=19):
    confusion_matrix = np.zeros((num_class,) * 2)
    mask = (GT >= 0) & (GT < num_class)
    label = num_class * GT[mask].astype('int') + Pred[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix += count.reshape(num_class, num_class)
    return confusion_matrix


def MIOU_score(GT, Pred, num_class=19):
    confusion_matrix = MIOU(GT, Pred, num_class)
    score = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
    score = np.nanmean(score)
    return score