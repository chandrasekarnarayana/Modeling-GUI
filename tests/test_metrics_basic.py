import numpy as np
from modeling_gui.metrics import regression_metrics, classification_metrics


def test_regression_mape_handles_zero():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 1.1])
    mets = regression_metrics(y_true, y_pred, {})
    assert "MAPE" in mets


def test_classification_prob_optional():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    mets = classification_metrics(y_true, y_pred, y_proba=None)
    assert "accuracy" in mets
