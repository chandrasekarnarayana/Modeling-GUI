"""
Project save/load helpers for Modeling-GUI.
Stores metadata, model, settings, and summaries for reproducibility.
"""

import base64
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from io import BytesIO

import joblib
from modeling_gui.drift import compute_snapshot
from modeling_gui.preprocessing import infer_column_types


def _model_to_base64(model: Any) -> str:
    buffer = BytesIO()
    joblib.dump(model, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _model_from_base64(data: str) -> Any:
    buffer = BytesIO(base64.b64decode(data))
    return joblib.load(buffer)


def compute_file_hash(path: str) -> Optional[str]:
    """
    Compute SHA256 hash of a file for integrity checks.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None


def save_project(path: str, project_data: Dict[str, Any]) -> str:
    """
    Save a project dictionary (including model) to a .mgui JSON file.
    """
    data = project_data.copy()
    model = data.pop("model", None)
    if model is not None:
        data["model_b64"] = _model_to_base64(model)
    # Add snapshot if not present
    if "training_snapshot" not in data and "dataframe_for_snapshot" in data:
        df = data.pop("dataframe_for_snapshot")
        col_types = infer_column_types(df)
        snap = compute_snapshot(df, col_types)
        data["training_snapshot"] = {
            "rows": snap.rows,
            "feature_stats": snap.feature_stats,
            "col_types": col_types,
        }
    json_path = Path(path)
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(json_path.resolve())


def load_project(path: str) -> Dict[str, Any]:
    """
    Load a project from a .mgui JSON file and rehydrate the model if possible.
    """
    json_path = Path(path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    model_b64 = data.pop("model_b64", None)
    if model_b64:
        try:
            data["model"] = _model_from_base64(model_b64)
        except Exception:
            data["model"] = None
    if "run_history" in data and data["run_history"] is None:
        data["run_history"] = []
    return data
