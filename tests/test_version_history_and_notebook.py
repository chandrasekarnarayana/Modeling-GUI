import tempfile
from pathlib import Path

from modeling_gui.utils.project import save_project, load_project
import pytest
from modeling_gui.notebook_export import export_notebook


def test_run_history_roundtrip(tmp_path):
    run_history = [
        {"model_name": "m1", "metrics": {"accuracy": 0.9}, "domain": "Business", "problem_type": "classification", "timestamp": "2024-01-01T00:00:00Z"},
        {"model_name": "m2", "metrics": {"RMSE": 1.2}, "domain": "Finance", "problem_type": "regression", "timestamp": "2024-01-02T00:00:00Z"},
    ]
    project = {"run_history": run_history}
    path = tmp_path / "proj.mgui"
    save_project(path, project)
    loaded = load_project(path)
    assert len(loaded.get("run_history", [])) == 2
    assert loaded["run_history"][0]["model_name"] == "m1"


def test_notebook_export_contains_cells(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    project = {"data_path": "demo.csv", "x_columns": ["feat1", "feat2"], "y_column": "target", "metrics": {"accuracy": 0.8}}
    nb_path = tmp_path / "export.ipynb"
    export_notebook(nb_path, project)
    nb = nbformat.read(nb_path, as_version=4)
    assert len(nb.cells) >= 5
    cell_sources = "\n".join(cell.source for cell in nb.cells)
    assert "demo.csv" in cell_sources
    assert "feat1" in cell_sources
    assert "target" in cell_sources
    assert "fit" in cell_sources or "train" in cell_sources


def test_run_history_filtering():
    from modeling_gui.utils.history import filter_run_history

    runs = [
        {"model_name": "RF", "problem_type": "classification"},
        {"model_name": "GB", "problem_type": "regression"},
        {"model_name": "Prophet", "problem_type": "forecasting"},
    ]
    assert len(filter_run_history(runs, problem="All", model_filter="All")) == 3
    assert len(filter_run_history(runs, problem="classification", model_filter="All")) == 1
    assert len(filter_run_history(runs, problem="All", model_filter="gb")) == 1
