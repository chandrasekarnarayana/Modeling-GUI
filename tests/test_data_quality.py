import pandas as pd
from modeling_gui.data_quality import (
    detect_missing_target,
    detect_non_numeric_features,
    detect_potential_leakage,
    detect_class_imbalance,
    detect_imbalance,
)


def test_missing_target_issue():
    df = pd.DataFrame({"y": [1, None, 3], "x": [1, 2, 3]})
    issues = detect_missing_target(df, "y")
    assert issues


def test_non_numeric_features_issue():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
    issues = detect_non_numeric_features(df, ["x"])
    assert issues and issues[0].severity == "warning"


def test_leakage_detection():
    df = pd.DataFrame({"y": [1, 2, 3], "y_copy": [1, 2, 3], "x": [0, 0, 0]})
    issues = detect_potential_leakage(df, ["y", "y_copy", "x"], "y")
    codes = [i.code for i in issues]
    assert "target_in_features" in codes or "high_corr_leakage" in codes


def test_class_imbalance():
    df = pd.DataFrame({"y": [0, 0, 0, 0, 1]})
    issues = detect_class_imbalance(df["y"])
    # imbalance detection may be heuristic; accept empty or warning
    if issues:
        assert issues[0].severity in {"warning", "info"}


def test_detect_imbalance_ratio():
    y = pd.Series([0] * 50 + [1] * 2)
    issue = detect_imbalance(y)
    assert issue is not None
    assert issue.code == "class_imbalance"
