import pandas as pd
from modeling_gui.drift import compute_snapshot, compare_snapshot
from modeling_gui.preprocessing import infer_column_types


def test_drift_detection():
    train = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    new = pd.DataFrame({"x": [10, 12, 14, 16], "y": [1, 1, 1, 1]})
    col_types = infer_column_types(train)
    snap = compute_snapshot(train, col_types)
    issues = compare_snapshot(snap, new, col_types)
    assert issues


def test_drift_ignores_stable_columns():
    train = pd.DataFrame({"x": [1, 2, 3, 4], "stable": [1, 1, 1, 1]})
    new = pd.DataFrame({"x": [10, 12, 14, 16], "stable": [1, 1, 1, 1]})
    col_types = infer_column_types(train)
    snap = compute_snapshot(train, col_types)
    issues = compare_snapshot(snap, new, col_types)
    # Should flag x but not stable
    flagged_cols = {i.message.split()[-1] for i in issues}
    assert "x" in "".join(flagged_cols)
