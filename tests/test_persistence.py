import os
import tempfile
import numpy as np
from modeling_gui.models import ModelManager
from modeling_gui.utils import load_model_bundle, save_model_bundle


def test_model_save_and_load_roundtrip():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    manager = ModelManager()
    model = manager.random_forest(X, y, n_estimators=5, max_depth=2)

    metadata = {"model_choice": "Random Forest", "feature_names": ["x"], "train_test_split": False}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        path = tmp.name
    try:
        save_model_bundle(path, model, metadata)
        bundle = load_model_bundle(path)
        loaded_model = bundle["model"]
        preds = loaded_model.predict(X)
        assert len(preds) == len(y)
    finally:
        if os.path.exists(path):
            os.remove(path)
