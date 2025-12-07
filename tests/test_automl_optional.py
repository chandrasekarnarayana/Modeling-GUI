import numpy as np
import pandas as pd
import pytest

from modeling_gui import automl
from modeling_gui.automl import run_automl, detect_problem_type
HAVE_FLAML = getattr(automl, "_FLAML_AVAILABLE", False)


@pytest.mark.skipif(not HAVE_FLAML, reason="AutoML optional dependency not installed")
def test_run_automl_small_dataset():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [2, 4, 6, 8, 10, 12]})
    pt, _ = detect_problem_type(df, "y")
    assert pt == "regression"
    result = run_automl(df, ["x"], "y", settings={"time_budget": 5, "n_splits": 2})
    assert result.best_model is not None
    assert "r2_train" in result.metrics or "accuracy" in result.metrics
