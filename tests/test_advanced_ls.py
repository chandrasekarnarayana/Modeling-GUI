import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from modeling_gui.models import ModelManager


def test_wls_differs_from_ols_with_weights():
    X = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    y = pd.Series([1, 2, 3, 4, 10])  # last point is influential
    weights = pd.Series([1, 1, 1, 1, 0.1])  # down-weight outlier
    mgr = ModelManager()
    ols = mgr.ols(X, y)
    wls = mgr.wls(X, y, weights)
    coef_ols = ols.params[1] if hasattr(ols, "params") else ols._result.params[1]
    coef_wls = wls.params[1] if hasattr(wls, "params") else wls._result.params[1]
    assert coef_wls != coef_ols


def test_gls_runs_and_params_finite():
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([1, 1.9, 3.1, 4.05])
    sigma = np.eye(len(X))
    mgr = ModelManager()
    gls = mgr.gls(X, y, sigma=sigma)
    assert np.isfinite(gls.params).all()


def test_recursive_ls_close_to_ols():
    X = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    y = pd.Series([1, 2, 3, 4, 5])
    mgr = ModelManager()
    rec = mgr.recursive_ls(X, y)
    ols = mgr.ols(X, y)
    assert_allclose(rec.params, ols.params, atol=1e-1)


def test_rolling_ls_window_output_length():
    X = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    y = pd.Series([1, 2, 3, 4, 5])
    mgr = ModelManager()
    df_params = mgr.rolling_ls(X, y, window=3)
    assert not df_params.empty
    # For n=5, window=3 => 3 rows of params
    assert df_params.shape[0] == 3


def test_rlm_less_sensitive_to_outlier():
    X = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    y = pd.Series([1, 2, 3, 4, 100])  # strong outlier
    mgr = ModelManager()
    ols = mgr.ols(X, y)
    rlm = mgr.rlm(X, y)
    coef_ols = ols.params[1] if hasattr(ols, "params") else ols._result.params[1]
    coef_rlm = rlm.params[1] if hasattr(rlm, "params") else rlm._result.params[1]
    assert abs(coef_rlm) < abs(coef_ols)
