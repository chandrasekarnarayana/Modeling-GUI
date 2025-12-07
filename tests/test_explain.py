import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from modeling_gui.explain import explain_global, explain_local


def test_explain_global_fallback():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
    y = np.array([1, 2, 3, 4])
    model = LinearRegression().fit(X, y)
    exp = explain_global(model, X, X.columns)
    assert len(exp.feature_names) == len(exp.importance_values)


def test_explain_local_fallback():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
    y = np.array([1, 2, 3, 4])
    model = LinearRegression().fit(X, y)
    exp = explain_local(model, X, X.columns, index=0)
    assert len(exp.feature_names) == len(exp.contributions)
