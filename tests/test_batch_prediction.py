import pandas as pd

from modeling_gui.models import ModelManager
from modeling_gui.utils.persistence import save_model_bundle, load_model_bundle


def test_batch_prediction_export(tmp_path):
    # Train and save
    X = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [0, 1, 0, 1]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    mgr = ModelManager()
    model = mgr.random_forest(X, y, n_estimators=10, max_depth=3)
    bundle_path = tmp_path / "model.joblib"
    save_model_bundle(bundle_path, model, {"feature_names": list(X.columns)})

    # Load and predict on new data
    bundle = load_model_bundle(bundle_path)
    loaded_model = bundle["model"]
    new_df = pd.DataFrame({"x1": [5, 6], "x2": [1, 0]})
    preds = loaded_model.predict(new_df)
    output = new_df.copy()
    output["prediction"] = preds
    out_path = tmp_path / "preds.csv"
    output.to_csv(out_path, index=False)

    assert out_path.exists()
    loaded_out = pd.read_csv(out_path)
    assert loaded_out.shape[0] == len(new_df)
    assert "prediction" in loaded_out.columns
