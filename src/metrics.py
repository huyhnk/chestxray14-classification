from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score


def fit_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Fit one F1-optimal threshold per class. Fit on validation data only."""
    thresholds = []
    for i in range(y_prob.shape[1]):
        if np.unique(y_true[:, i]).size < 2:
            thresholds.append(0.5)
            continue
        precision, recall, threshold_values = precision_recall_curve(y_true[:, i], y_prob[:, i])
        if len(threshold_values) == 0:
            thresholds.append(0.5)
            continue
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        thresholds.append(float(threshold_values[int(np.nanargmax(f1))]))
    return np.asarray(thresholds, dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray):
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)
    aucs, f1s = [], []
    for i in range(y_true.shape[1]):
        auc = np.nan
        if np.unique(y_true[:, i]).size >= 2:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        aucs.append(auc)
        f1s.append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))

    return {
        "macro_auroc": float(np.nanmean(aucs)),
        "macro_f1": float(np.mean(f1s)),
        "per_class_auroc": [None if np.isnan(x) else float(x) for x in aucs],
        "per_class_f1": [float(x) for x in f1s],
    }
