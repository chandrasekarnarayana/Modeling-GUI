"""
Explainability helpers with optional SHAP integration.
Falls back to model importances/coefficients when SHAP is unavailable.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import shap

    _HAVE_SHAP = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_SHAP = False


@dataclass
class GlobalExplanation:
    feature_names: List[str]
    importance_values: List[float]
    method: str
    summary_text: str


@dataclass
class LocalExplanation:
    feature_names: List[str]
    contributions: List[float]
    base_value: Optional[float]
    predicted_value: float
    method: str


def _fallback_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        vals = np.abs(np.asarray(model.coef_).flatten())
    else:
        vals = np.zeros(len(feature_names))
    order = np.argsort(vals)[::-1]
    return list(np.array(feature_names)[order]), list(vals[order])


def explain_global(model, X, feature_names) -> GlobalExplanation:
    feature_names = list(feature_names)
    if _HAVE_SHAP:
        try:
            explainer = shap.Explainer(model, np.asarray(X))
            shap_vals = explainer(np.asarray(X))
            mean_abs = np.mean(np.abs(shap_vals.values), axis=0)
            order = np.argsort(mean_abs)[::-1]
            feats = list(np.array(feature_names)[order])
            vals = list(mean_abs[order])
            summary = f"Top drivers: {', '.join(feats[:3])}."
            return GlobalExplanation(feats, vals, "shap", summary)
        except Exception:
            pass
    feats, vals = _fallback_importances(model, feature_names)
    summary = f"Top drivers: {', '.join(feats[:3])}." if feats else "No feature importances available."
    return GlobalExplanation(feats, vals, "native_importance", summary)


def explain_local(model, X_sample, feature_names, index: int = 0) -> LocalExplanation:
    feature_names = list(feature_names)
    x_row = np.asarray(X_sample)[index : index + 1]
    predicted = float(model.predict(x_row)[0]) if hasattr(model, "predict") else float("nan")
    baseline = None
    contributions = np.zeros(len(feature_names))
    if _HAVE_SHAP:
        try:
            explainer = shap.Explainer(model, np.asarray(X_sample))
            vals = explainer(x_row)
            contributions = np.asarray(vals.values).reshape(-1)
            baseline = float(getattr(vals, "base_values", [np.nan])[0])
            return LocalExplanation(feature_names, list(contributions), baseline, predicted, "shap")
        except Exception:
            pass
    return LocalExplanation(feature_names, list(contributions), baseline, predicted, "native_importance")


def compute_partial_dependence(model, X, feature_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a simple 1D partial dependence for a feature by varying it over its range.
    """
    if feature_name not in X.columns:
        return np.array([]), np.array([])
    grid = np.linspace(X[feature_name].min(), X[feature_name].max(), num=20)
    X_copy = X.copy()
    preds = []
    for val in grid:
        X_copy[feature_name] = val
        preds.append(np.mean(model.predict(X_copy)))
    return grid, np.array(preds)
