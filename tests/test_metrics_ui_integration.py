import numpy as np

from modeling_gui.metrics import classification_metrics, choose_primary_metrics


def test_classification_metrics_with_proba_include_auc():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2]])
    mets = classification_metrics(y_true, y_pred, y_proba)
    assert "roc_auc" in mets
    assert "pr_auc" in mets


def test_primary_metric_selection_by_domain():
    business = choose_primary_metrics("Business", "classification")
    assert "accuracy" in business and "roc_auc" in business
    finance = choose_primary_metrics("Finance", "regression")
    assert business[0] == "accuracy"
    assert "RMSE" in finance
