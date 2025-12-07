"""Richer metrics utilities with domain-aware primary metric selection."""

from typing import Dict, List, Optional

import numpy as np
from sklearn import metrics as sk_metrics


def regression_metrics(y_true, y_pred, settings=None) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "R2": float(sk_metrics.r2_score(y_true, y_pred)) if len(y_true) > 0 else float("nan"),
        "RMSE": float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred))) if len(y_true) > 0 else float("nan"),
        "MAE": float(sk_metrics.mean_absolute_error(y_true, y_pred)) if len(y_true) > 0 else float("nan"),
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if (y_true != 0).any() else np.nan
    out["MAPE"] = float(mape)
    return out


def classification_metrics(y_true, y_pred, y_proba=None, settings=None) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(sk_metrics.accuracy_score(y_true, y_pred)) if len(y_true) > 0 else float("nan"),
        "balanced_accuracy": float(sk_metrics.balanced_accuracy_score(y_true, y_pred)) if len(y_true) > 0 else float("nan"),
        "macro_f1": float(sk_metrics.f1_score(y_true, y_pred, average="macro")) if len(y_true) > 0 else float("nan"),
        "micro_f1": float(sk_metrics.f1_score(y_true, y_pred, average="micro")) if len(y_true) > 0 else float("nan"),
    }
    if y_proba is not None:
        try:
            # Binary: use positive class column when available; multi-class falls back to OVR.
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                out["roc_auc"] = float(sk_metrics.roc_auc_score(y_true, y_proba[:, 1]))
            else:
                out["roc_auc"] = float(sk_metrics.roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except Exception:
            pass
        try:
            precision, recall, _ = sk_metrics.precision_recall_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            out["pr_auc"] = float(sk_metrics.auc(recall, precision))
        except Exception:
            pass
    return out


def choose_primary_metrics(domain: str, problem_type: str) -> List[str]:
    domain = (domain or "Generic").lower()
    if problem_type == "regression":
        if domain == "finance":
            return ["RMSE", "MAE", "MAPE"]
        if domain == "science":
            return ["R2", "RMSE"]
        return ["RMSE", "R2"]
    if problem_type == "classification":
        if domain == "business":
            return ["accuracy", "roc_auc", "macro_f1"]
        return ["accuracy", "macro_f1"]
    if problem_type == "forecasting":
        return ["RMSE", "MAE", "MAPE"]
    return []
