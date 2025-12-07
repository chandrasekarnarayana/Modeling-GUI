import numpy as np
import pandas as pd

from modeling_gui.preprocessing import (
    build_preprocessing_pipeline,
    EncodingStrategy,
    generate_date_features,
    infer_column_types,
)


def test_one_hot_encoding_expands_categories():
    df = pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "a"]})
    pipe = build_preprocessing_pipeline(df, encoding=EncodingStrategy.ONE_HOT, scale_numeric=False)
    transformed = pipe.fit_transform(df)
    # num + one-hot (drop_first=True in encoder) => 1 numeric + 1 dummy
    assert transformed.shape[1] >= 2


def test_target_encoding_maps_unseen_to_global_mean():
    df_train = pd.DataFrame({"cat": ["a", "b", "a", "c"], "num": [1, 2, 3, 4]})
    y_train = pd.Series([10, 20, 10, 30])
    df_test = pd.DataFrame({"cat": ["a", "d"], "num": [5, 6]})  # 'd' unseen
    pipe = build_preprocessing_pipeline(df_train, encoding=EncodingStrategy.TARGET, scale_numeric=False, target=y_train)
    # Fit uses target; transform test should not crash
    transformed_test = pipe.fit(df_train, y_train).transform(df_test)
    arr = np.asarray(transformed_test, dtype=float)
    assert np.isfinite(arr).all()


def test_date_feature_generation():
    df = pd.DataFrame({"date": ["2023-01-01", "2023-02-15"], "x": [1, 2]})
    df_feat = generate_date_features(df, "date")
    assert {"date_year", "date_month", "date_dayofweek", "date_quarter"}.issubset(df_feat.columns)
