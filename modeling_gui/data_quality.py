"""
Data quality checks and simple heuristics for leakage/imbalance.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class DataIssue:
    severity: str  # "info" | "warning" | "error"
    code: str
    message: str
    suggestion: Optional[str] = None

# Backwards alias
Issue = DataIssue


def validate_schema(df: pd.DataFrame, expected_schema: Optional[dict] = None) -> List[DataIssue]:
    issues: List[DataIssue] = []
    if expected_schema:
        missing = [c for c in expected_schema if c not in df.columns]
        if missing:
            issues.append(
                DataIssue(
                    severity="error",
                    code="missing_columns",
                    message=f"Missing columns: {', '.join(missing)}",
                    suggestion="Ensure your CSV includes the required columns.",
                )
            )
    return issues


def detect_missing_issues(df: pd.DataFrame, target_col: Optional[str]) -> List[DataIssue]:
    issues: List[DataIssue] = []
    missing_pct = df.isna().mean()
    high_missing = missing_pct[missing_pct > 0.3]
    if not high_missing.empty:
        cols = ", ".join(high_missing.index)
        issues.append(
            DataIssue(
                severity="warning",
                code="high_missing",
                message=f"High missing values in: {cols}",
                suggestion="Consider imputing or dropping columns with many missing values.",
            )
        )
    if target_col and target_col in df.columns:
        tgt_missing = df[target_col].isna().mean()
        if tgt_missing > 0.05:
            issues.append(
                DataIssue(
                    severity="warning",
                    code="target_missing",
                    message=f"Target column '{target_col}' has {tgt_missing:.1%} missing.",
                    suggestion="Drop rows with missing target or choose a different target.",
                )
            )
    return issues


def detect_type_mismatches(df: pd.DataFrame) -> List[DataIssue]:
    issues: List[DataIssue] = []
    for col in df.columns:
        if df[col].dtype == object:
            try:
                pd.to_numeric(df[col])
                issues.append(
                    DataIssue(
                        severity="info",
                        code="possibly_numeric_as_text",
                        message=f"Column '{col}' may be numeric stored as text.",
                        suggestion="Convert to numeric if appropriate.",
                    )
                )
            except Exception:
                continue
    return issues


def detect_potential_leakage(df: pd.DataFrame, x_cols: List[str], y_col: Optional[str]) -> List[DataIssue]:
    issues: List[DataIssue] = []
    if y_col and y_col in x_cols:
        issues.append(
            DataIssue(
                severity="error",
                code="target_in_features",
                message="Target column is present among features.",
                suggestion="Remove the target from the feature list to avoid leakage.",
            )
        )
    if y_col and y_col in df.columns:
        target = df[y_col]
        # Name-based heuristic
        leak_like = [c for c in x_cols if any(tok in c.lower() for tok in ["id", "target", "label"])]
        if leak_like:
            issues.append(
                DataIssue(
                    severity="warning",
                    code="leaky_name",
                    message=f"Columns may leak target: {', '.join(leak_like)}",
                    suggestion="Remove ID/label-like columns from features if they duplicate the target.",
                )
            )
        # Correlation heuristic for numeric
        numeric_cols = [c for c in x_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
        if np.issubdtype(target.dtype, np.number) and numeric_cols:
            for col in numeric_cols:
                corr = target.corr(df[col])
                if pd.notna(corr) and abs(corr) > 0.98:
                    issues.append(
                        DataIssue(
                            severity="warning",
                            code="high_corr_leakage",
                            message=f"Feature '{col}' is almost identical to target (corr={corr:.2f}).",
                            suggestion="Drop or transform this feature to avoid leakage.",
                        )
                    )
    return issues


def detect_imbalance(y: pd.Series) -> Optional[DataIssue]:
    if y is None:
        return None
    if y.dtype.kind in "biu":
        counts = y.value_counts(normalize=False, dropna=True)
        if not counts.empty:
            max_c, min_c = counts.max(), counts.min()
            ratio = max_c / max(min_c, 1)
        else:
            ratio = 1
        if not counts.empty and ratio > 10:
            return DataIssue(
                severity="warning",
                code="class_imbalance",
                message=f"Strong class imbalance detected (max/min ratio ≈ {ratio:.1f}).",
                suggestion="Consider class weights, resampling, or threshold tuning.",
            )
    return None


# Additional simple heuristics
def detect_missing_target(df: pd.DataFrame, target_col: str) -> List[DataIssue]:
    return detect_missing_issues(df, target_col)


def detect_non_numeric_features(df: pd.DataFrame, x_cols: List[str]) -> List[DataIssue]:
    issues: List[DataIssue] = []
    non_numeric = [c for c in x_cols if c in df.columns and not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric:
        issues.append(
            DataIssue(
                severity="warning",
                code="non_numeric_features",
                message=f"Non-numeric features detected: {', '.join(non_numeric)}",
                suggestion="Encode or remove non-numeric features for numeric models.",
            )
        )
    return issues


def detect_class_imbalance(y: pd.Series) -> List[DataIssue]:
    issue = detect_imbalance(y)
    return [issue] if issue else []
