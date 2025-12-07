import os
import tempfile
import numpy as np
import pandas as pd
from modeling_gui.models import ModelManager
from modeling_gui.reporting import build_summary_text
from modeling_gui.utils import save_project, load_project, compute_file_hash


def test_project_save_and_load_roundtrip():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    X = df[["x"]]
    y = df["y"]
    manager = ModelManager()
    model = manager.ols(X, y)

    summary = {
        "data": {"rows": 3, "cols": 2, "types": {"numeric": 2, "categorical": 0}},
        "preprocessing": {"missing": "none", "scaling": "none", "categoricals": "n/a"},
        "modeling": {"problem_type": "regression", "algorithms": ["OLS"], "best_model": "OLS", "cv": "n/a"},
        "evaluation": {"split": "test_size=0.2", "primary": {}, "secondary": {}},
    }
    metrics = {"r2_test": 1.0, "rmse_test": 0.0}
    text = build_summary_text(summary, metrics)
    assert "Rows" in text

    project = {
        "version": "0.0.0",
        "data_path": None,
        "data_hash": None,
        "domain": "Generic",
        "x_columns": ["x"],
        "y_column": "y",
        "automl_settings": {},
        "summary": summary,
        "metrics": metrics,
        "model": model,
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mgui") as tmp:
        path = tmp.name
    try:
        saved_path = save_project(path, project)
        loaded = load_project(saved_path)
        assert loaded["domain"] == "Generic"
        assert loaded["summary"]["data"]["rows"] == 3
        assert loaded["model"] is not None
    finally:
        if os.path.exists(path):
            os.remove(path)
