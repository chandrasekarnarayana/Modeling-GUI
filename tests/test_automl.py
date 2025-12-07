import pandas as pd
import pytest
from modeling_gui import automl
from modeling_gui.automl import run_automl


def test_run_automl_synthetic_regression():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [2, 4, 6, 8, 10, 12]})
    if not getattr(automl, "_HAVE_FLAML", False):
        pytest.skip("AutoML optional dependency not installed")
    result = run_automl(df, ["x"], "y", settings={"time_budget": 5, "domain": "Generic"}, problem_type="regression")
    assert result.best_candidate is not None
    assert result.candidate_models
    assert isinstance(result.best_candidate.metrics, dict)
