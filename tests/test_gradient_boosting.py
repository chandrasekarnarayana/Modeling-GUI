import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from modeling_gui.metrics import regression_metrics, classification_metrics


def test_gradient_boosting_regression_basic():
    X, y = make_regression(n_samples=60, n_features=5, random_state=42, noise=0.1)
    model = GradientBoostingRegressor(random_state=42, n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)
    mets = regression_metrics(y, preds, {})
    assert preds.shape[0] == X.shape[0]
    for key in ("R2", "RMSE", "MAE", "MAPE"):
        assert key in mets
        assert np.isfinite(mets[key])


def test_gradient_boosting_classifier_metrics():
    X, y = make_classification(n_samples=80, n_features=6, n_informative=4, random_state=42)
    model = GradientBoostingClassifier(random_state=42, n_estimators=60, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    mets = classification_metrics(y, preds, proba, {})
    assert preds.shape[0] == X.shape[0]
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        assert key in mets
    if "roc_auc" in mets:
        assert np.isfinite(mets["roc_auc"])
    if "pr_auc" in mets:
        assert np.isfinite(mets["pr_auc"])
