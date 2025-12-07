import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from modeling_gui.explain import explain_global, explain_local, compute_partial_dependence
from modeling_gui.visualization import plot_feature_importance, plot_local_contributions


def _tiny_rf():
    X = pd.DataFrame({"feature_0": [1, 2, 3, 4], "feature_1": [2, 1, 0, -1]})
    y = np.array([1.0, 1.5, 2.5, 4.0])
    model = RandomForestRegressor(random_state=0, n_estimators=10).fit(X, y)
    return model, X, y


def test_explain_global_tree():
    model, X, _ = _tiny_rf()
    exp = explain_global(model, X, X.columns)
    assert len(exp.feature_names) == X.shape[1]
    assert len(exp.importance_values) == X.shape[1]
    assert all(np.isfinite(exp.importance_values))
    fig = plot_feature_importance(exp.feature_names, exp.importance_values)
    assert fig is not None
    assert len(fig.axes) > 0


def test_explain_local_tree():
    model, X, _ = _tiny_rf()
    exp = explain_local(model, X, X.columns, index=0)
    assert len(exp.contributions) == X.shape[1]
    assert np.isfinite(exp.predicted_value)
    fig = plot_local_contributions(exp.feature_names, exp.contributions, baseline=exp.base_value, predicted=exp.predicted_value)
    assert fig is not None
    assert len(fig.axes) > 0


def test_partial_dependence_basic():
    model, X, _ = _tiny_rf()
    grid, preds = compute_partial_dependence(model, X, "feature_0")
    assert len(grid) > 1
    assert len(grid) == len(preds)
