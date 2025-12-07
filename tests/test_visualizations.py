import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

from modeling_gui.visualization import (
    plot_regression,
    plot_residuals,
    plot_confusion_from_predictions,
    plot_tree_diagram,
    plot_forecast,
)


def test_regression_and_residual_plots(tmp_path: Path):
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    y = 2 * X.flatten() + 1
    model = LinearRegression().fit(X, y)
    fig_reg = plot_regression(X, y, model)
    assert fig_reg is not None
    assert len(fig_reg.axes) > 0
    fig_reg.savefig(tmp_path / "reg.png")

    preds = model.predict(X)
    fig_res = plot_residuals(y, preds)
    assert fig_res is not None
    assert len(fig_res.axes) > 0
    fig_res.savefig(tmp_path / "res.png")


def test_confusion_and_tree_plots(tmp_path: Path):
    X = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 1, 0, 0]})
    y = np.array([0, 1, 0, 1])
    clf = RandomForestClassifier(random_state=0, n_estimators=5, max_depth=2)
    clf.fit(X, y)
    preds = clf.predict(X)
    fig_cm = plot_confusion_from_predictions(y, preds, labels=[0, 1])
    assert fig_cm is not None
    assert len(fig_cm.axes) > 0
    fig_cm.savefig(tmp_path / "cm.png")

    fig_tree = plot_tree_diagram(clf)
    assert fig_tree is not None
    assert len(fig_tree.axes) > 0
    fig_tree.savefig(tmp_path / "tree.png")


def test_forecast_plot(tmp_path: Path):
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    y_train = pd.Series(np.arange(5), index=idx[:5])
    y_valid = pd.Series(np.arange(5, 8), index=idx[5:8])
    y_pred = pd.Series(np.arange(5, 8) + 0.5, index=idx[5:8])
    y_forecast = pd.Series(np.arange(8, 10), index=range(8, 10))
    fig = plot_forecast(idx, y_train, y_valid, y_pred, y_forecast)
    assert fig is not None
    assert len(fig.axes) > 0
    fig.savefig(tmp_path / "forecast.png")
