"""
Comprehensive Evaluation Metrics for Imbalanced Financial Crime Detection.

Calculates:
- Precision-Recall Area Under Curve (PR-AUC / Average Precision)
- Receiver Operating Characteristic Area Under Curve (ROC-AUC)
- Confusion Matrix (TP, FP, TN, FN)
- Threshold Tuning (optimal F-beta trade-off)
- Regulatory Efficiency (False Alarm Ratio and Investigator Alert Burden)
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _trapezoidal_auc(x: List[float], y: List[float]) -> float:
    """Trapezoidal integration for AUC without external libraries."""
    if len(x) < 2:
        return 0.0
    auc = 0.0
    for i in range(len(x) - 1):
        h = abs(x[i + 1] - x[i])
        avg_height = (y[i + 1] + y[i]) / 2.0
        auc += h * avg_height
    return auc


def compute_binary_metrics(
    y_true: List[int],
    y_probs: List[float],
    threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Computes forensic evaluation metrics across illicit and licit accounts.
    """
    n = len(y_true)
    if n == 0:
        return {}

    # Discretize predictions at threshold
    y_pred = [1 if p >= threshold else 0 for p in y_probs]

    # Confusion Matrix Counts
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    num_pos = tp + fn
    num_neg = tn + fp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

    # Scikit-learn or numerical calculation for PR-AUC and ROC-AUC
    if _HAS_SKLEARN:
        try:
            pr_auc = float(average_precision_score(y_true, y_probs))
        except Exception:
            pr_auc = precision
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            roc_auc = 0.5
    else:
        # Approximate PR-AUC by sorting probabilities
        sorted_pairs = sorted(zip(y_probs, y_true), key=lambda x: x[0], reverse=True)
        precisions = []
        recalls = []
        cur_tp = 0
        cur_fp = 0
        for p_val, y_val in sorted_pairs:
            if y_val == 1:
                cur_tp += 1
            else:
                cur_fp += 1
            precisions.append(cur_tp / max(cur_tp + cur_fp, 1))
            recalls.append(cur_tp / max(num_pos, 1))
        pr_auc = _trapezoidal_auc(recalls, precisions)
        roc_auc = 0.5 + (recall + specificity - 1.0) / 2.0

    # Operational Investigator Metrics
    alert_rate = (tp + fp) / max(n, 1)
    investigator_burden = fp / max(tp, 1)  # False alarms per true positive

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "operational_metrics": {
            "total_accounts_evaluated": n,
            "total_mules": num_pos,
            "total_licit": num_neg,
            "decision_threshold": threshold,
            "alert_rate_pct": round(alert_rate * 100.0, 2),
            "false_alarm_ratio": round(investigator_burden, 2),
        },
    }


def find_optimal_threshold(
    y_true: List[int],
    y_probs: List[float],
    beta: float = 1.0,
    search_steps: int = 50,
) -> Tuple[float, float]:
    """
    Performs grid search to discover the decision threshold maximizing F-beta score.
    In AML, beta > 1 (e.g. beta=2) places more weight on recall (catching mules).
    """
    best_thresh = 0.5
    best_f_beta = -1.0
    beta_sq = beta ** 2

    for step in range(1, search_steps):
        thresh = step / float(search_steps)
        metrics = compute_binary_metrics(y_true, y_probs, threshold=thresh)
        p = metrics["precision"]
        r = metrics["recall"]
        denom = (beta_sq * p) + r
        if denom > 0:
            f_b = ((1.0 + beta_sq) * p * r) / denom
        else:
            f_b = 0.0

        if f_b > best_f_beta:
            best_f_beta = f_b
            best_thresh = thresh

    return round(best_thresh, 4), round(best_f_beta, 4)
