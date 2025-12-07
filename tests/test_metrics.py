import numpy as np
from modeling_gui.metrics import regression_metrics, classification_metrics, choose_primary_metrics


def test_regression_metrics_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    mets = regression_metrics(y_true, y_pred, {})
    assert "R2" in mets and "RMSE" in mets


def test_classification_metrics_basic():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    mets = classification_metrics(y_true, y_pred)
    assert "accuracy" in mets and mets["accuracy"] >= 0


def test_choose_primary_metrics():
    finance = choose_primary_metrics("Finance", "regression")
    assert "RMSE" in finance
    business = choose_primary_metrics("Business", "classification")
    assert "accuracy" in business


def test_classification_metrics_with_proba_includes_auc():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2]])
    mets = classification_metrics(y_true, y_pred, y_proba)
    for key in ("roc_auc", "pr_auc", "macro_f1", "micro_f1"):
        assert key in mets
        assert 0 <= mets[key] <= 1
