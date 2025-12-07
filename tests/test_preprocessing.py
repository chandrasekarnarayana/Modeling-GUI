import pandas as pd
import pytest
from modeling_gui.preprocessing import infer_column_types, parse_date_columns, encode_categoricals
from modeling_gui.utils import (
    apply_missing_strategy,
    select_numeric_columns,
    standardize_features,
)


def test_apply_missing_strategy_drop_and_mean():
    df = pd.DataFrame({"a": [1.0, 2.0, None], "b": [1, None, 3], "c": ["x", "y", None]})

    dropped = apply_missing_strategy(df, "drop")
    assert len(dropped) == 1

    mean_filled = apply_missing_strategy(df, "mean")
    assert mean_filled.isna().sum().sum() == 1  # only non-numeric missing remains


def test_standardize_features_and_numeric_selection():
    df = pd.DataFrame({"num1": [1.0, 2.0, 3.0], "num2": [2.0, 4.0, 6.0], "cat": ["a", "b", "c"]})
    numeric_df, dropped = select_numeric_columns(df, ["num1", "num2", "cat"])
    assert dropped == ["cat"]
    scaled, scaler = standardize_features(numeric_df, numeric_df.columns)
    assert pytest.approx(float(scaled.mean().mean()), abs=1e-7) == 0.0
    assert scaler is not None


def test_infer_column_types_and_encode():
    df = pd.DataFrame({"num": [1, 2], "cat": ["a", "b"], "date": ["2023-01-01", "2023-01-02"]})
    types = infer_column_types(df)
    assert types["num"] == "numeric"
    df_parsed = parse_date_columns(df, ["date"])
    df_encoded = encode_categoricals(df_parsed, strategy="one_hot")
    assert any(col.startswith("cat_") for col in df_encoded.columns)
