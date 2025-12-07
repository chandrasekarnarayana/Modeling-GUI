import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def compute_regression_metrics(y_train, y_train_pred, y_test, y_test_pred):
    """
    Return regression metrics dictionary.
    """
    return {
        "r2_train": r2_score(y_train, y_train_pred),
        "r2_test": r2_score(y_test, y_test_pred),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
    }


def compute_classification_metrics(y_test, y_test_pred):
    """
    Return classification metrics dictionary.
    """
    return {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "report": classification_report(y_test, y_test_pred),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred),
    }


def is_regression_target(y):
    """
    Determine if a target is regression-like based on dtype.
    """
    dtype = getattr(y, "dtype", np.asarray(y).dtype)
    return dtype.kind in "if"
