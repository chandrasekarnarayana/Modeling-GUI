"""
Additional preprocessing utilities for automatic type inference and categorical/date handling.
"""

from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class EncodingStrategy(str, Enum):
    ONE_HOT = "one_hot"
    TARGET = "target"
    DROP = "drop"


class TargetMeanEncoder(TransformerMixin, BaseEstimator):
    """Simple mean target encoder with global-mean fallback for unseen categories."""

    def __init__(self, columns: List[str]):
        self.columns = columns
        self.mappings: Dict[str, Dict[object, float]] = {}
        self.global_mean: float = 0.0

    def fit(self, X, y):
        df = pd.DataFrame(X, columns=self.columns)
        self.global_mean = float(pd.Series(y).mean())
        for col in self.columns:
            means = pd.Series(y).groupby(df[col]).mean()
            self.mappings[col] = means.to_dict()
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.columns)
        for col in self.columns:
            mapping = self.mappings.get(col, {})
            df[col] = df[col].map(mapping).fillna(self.global_mean)
        return df.values


class TargetEncodingPipeline(TransformerMixin, BaseEstimator):
    """
    Combine target mean encoding for categorical columns with optional numeric scaling.
    """

    def __init__(self, cat_cols: List[str], num_cols: List[str], scale_numeric: bool):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.scale_numeric = scale_numeric
        self.encoder = TargetMeanEncoder(cat_cols) if cat_cols else None
        self.scaler = StandardScaler() if scale_numeric and num_cols else None

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        if self.encoder and y is not None:
            self.encoder.fit(df[self.cat_cols], y)
        if self.scaler:
            self.scaler.fit(df[self.num_cols])
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        parts = []
        if self.encoder:
            parts.append(self.encoder.transform(df[self.cat_cols]))
        if self.num_cols:
            num_vals = df[self.num_cols].values
            if self.scaler:
                num_vals = self.scaler.transform(num_vals)
            parts.append(num_vals)
        if parts:
            return np.hstack(parts)
        return df.values


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer basic column types: numeric, categorical, datetime, other.
    """
    types: Dict[str, str] = {}
    for col in df.columns:
        dtype = df[col].dtype
        if np.issubdtype(dtype, np.number):
            types[col] = "numeric"
        elif np.issubdtype(dtype, np.datetime64):
            types[col] = "datetime"
        else:
            # Try datetime parsing heuristically
            sample = df[col].astype(str).head(5)
            if sample.str.match(r"\d{4}-\d{2}-\d{2}").any():
                types[col] = "datetime"
            elif df[col].nunique(dropna=True) < max(50, len(df) * 0.2):
                types[col] = "categorical"
            else:
                types[col] = "other"
    return types


def parse_date_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Parse the specified date columns to datetime."""
    df = df.copy()
    columns = columns or []
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def encode_categoricals(df: pd.DataFrame, strategy: str = "one_hot", target: Optional[pd.Series] = None) -> pd.DataFrame:
    """Encode categorical columns using one-hot or simple target encoding."""
    df_enc = df.copy()
    cat_cols = [c for c in df_enc.columns if df_enc[c].dtype == object or str(df_enc[c].dtype) == "category"]
    if not cat_cols:
        return df_enc

    if strategy == "drop":
        return df_enc.drop(columns=cat_cols)

    if strategy == "target" and target is not None:
        global_mean = target.mean()
        for col in cat_cols:
            means = target.groupby(df_enc[col]).mean()
            df_enc[col] = df_enc[col].map(means).fillna(global_mean)
        return df_enc

    # default one-hot
    return pd.get_dummies(df_enc, columns=cat_cols, drop_first=True)


def generate_date_features(df: pd.DataFrame, date_col: str, features=None) -> pd.DataFrame:
    """
    Add simple date-derived features (year, month, dayofweek, quarter).
    """
    if features is None:
        features = ["year", "month", "dayofweek", "quarter"]
    df = df.copy()
    if date_col not in df.columns:
        return df
    date_series = pd.to_datetime(df[date_col], errors="coerce")
    for feat in features:
        if feat == "year":
            df[f"{date_col}_year"] = date_series.dt.year
        elif feat == "month":
            df[f"{date_col}_month"] = date_series.dt.month
        elif feat == "dayofweek":
            df[f"{date_col}_dayofweek"] = date_series.dt.dayofweek
        elif feat == "quarter":
            df[f"{date_col}_quarter"] = date_series.dt.quarter
    return df


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    encoding: EncodingStrategy = EncodingStrategy.ONE_HOT,
    scale_numeric: bool = True,
    handle_missing: str = "keep",
    target: Optional[pd.Series] = None,
) -> TransformerMixin:
    """
    Build a scikit-learn ColumnTransformer for numeric/categorical columns.
    Target encoding is applied inline if requested and target is provided; unseen categories map to global mean.
    """
    col_types = infer_column_types(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    transformers = []

    if num_cols:
        num_steps = []
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", PipelineSafe(num_steps), num_cols))

    if cat_cols:
        if encoding == EncodingStrategy.ONE_HOT:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            )
        elif encoding == EncodingStrategy.DROP:
            pass
        elif encoding == EncodingStrategy.TARGET and target is not None:
            return TargetEncodingPipeline(cat_cols, num_cols, scale_numeric)
        else:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            )

    if not transformers:
        return PipelineSafe([])

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


class PipelineSafe(TransformerMixin, BaseEstimator):
    """
    Minimal passthrough when no scaling is requested; wraps list of steps for ColumnTransformer compatibility.
    """

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        # Fit all sub-steps if any
        for name, transformer in self.steps:
            transformer.fit(X, y)
        return self

    def transform(self, X):
        X_trans = X
        for name, transformer in self.steps:
            X_trans = transformer.transform(X_trans)
        return X_trans
