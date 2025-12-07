import numpy as np

from modeling_gui.models import ModelManager
from modeling_gui.visualization import plot_curve_fit


def test_gaussian_fit_parameters_and_plot():
    X = np.linspace(-3, 3, 30)
    Y = 2 * np.exp(-(X - 0.5) ** 2 / (2 * 1.2**2)) + 0.1
    mgr = ModelManager()
    mean, std_dev, _ = mgr.gaussian_fit(X, Y)
    assert np.isfinite(mean)
    assert np.isfinite(std_dev)
    fig = plot_curve_fit(X, Y, (2, mean, std_dev), "gaussian")
    assert fig is not None
    assert len(fig.axes) > 0


def test_exponential_fit_parameters_and_plot():
    X = np.linspace(0, 2, 20)
    Y = 1.5 * np.exp(0.8 * X) + 0.2
    mgr = ModelManager()
    params = mgr.exponential_fit(X, Y)
    assert all(np.isfinite(params))
    fig = plot_curve_fit(X, Y, params, "exponential")
    assert fig is not None
    assert len(fig.axes) > 0
