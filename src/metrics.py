"""Shared evaluation metrics.

Seizure windows are a small minority of the dataset, so accuracy is a poor
headline number — a model that predicts "no seizure" every time still scores
well. Sensitivity, specificity and AUC are reported alongside it throughout.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def _to_numpy(x):
    """Accept torch tensors or array-likes interchangeably."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_metrics(y_true, y_pred, y_scores=None):
    """Binary classification metrics.

    Args:
        y_true: Ground-truth labels (0 = non-seizure, 1 = seizure).
        y_pred: Predicted labels.
        y_scores: Optional positive-class probabilities or decision scores,
            required for AUC.

    Returns:
        dict with accuracy, sensitivity, specificity and auc. AUC is NaN when
        scores are absent or the split contains only one class.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    metrics = {"accuracy": accuracy_score(y_true, y_pred)}

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Sensitivity (recall / true positive rate): of the real seizures, how many
    # did we catch? This is the number that matters clinically.
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (true negative rate): how well we avoid false alarms.
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics["auc"] = float("nan")
    if y_scores is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, _to_numpy(y_scores))
        except ValueError:
            # Raised when y_true has a single class — leave AUC as NaN.
            pass

    return metrics
