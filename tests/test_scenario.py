import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from modeling_gui.models import ModelManager


def test_scenario_prediction_changes():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 11, 12, 13]})
    y = np.array([1.0, 2.0, 2.5, 4.0])
    mgr = ModelManager()
    model = mgr.random_forest(X, y, n_estimators=20, max_depth=3)
    base_row = X.iloc[0].values.reshape(1, -1)
    baseline = float(model.predict(base_row)[0])
    changed_row = base_row.copy()
    changed_row[0, 0] = changed_row[0, 0] + 5
    new_pred = float(model.predict(changed_row)[0])
    assert baseline != new_pred
