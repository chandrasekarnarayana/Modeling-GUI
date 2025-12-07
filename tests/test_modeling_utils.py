import numpy as np
from modeling_gui.utils import (
    compute_classification_metrics,
    compute_regression_metrics,
    is_regression_target,
    split_data,
)


def test_split_data_shapes():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=0)
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_regression_and_classification_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.1, 2.9])
    metrics = compute_regression_metrics(y_true, y_pred, y_true, y_pred)
    assert "r2_test" in metrics and metrics["rmse_test"] >= 0

    y_class_true = np.array([0, 1, 1, 0])
    y_class_pred = np.array([0, 1, 0, 0])
    class_metrics = compute_classification_metrics(y_class_true, y_class_pred)
    assert "accuracy" in class_metrics and class_metrics["accuracy"] >= 0
    assert class_metrics["confusion_matrix"].shape == (2, 2)


def test_is_regression_target_detection():
    assert is_regression_target(np.array([1.0, 2.0]))
    assert not is_regression_target(np.array(["a", "b"]))
