"""
Guided troubleshooting suggestions based on simple heuristics.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class Suggestion:
    short_message: str
    detailed_message: str
    optional_fix: Optional[str] = None


def analyze_failure_context(error: Exception, df: Optional[pd.DataFrame], config: dict) -> List[Suggestion]:
    suggestions: List[Suggestion] = []
    msg = str(error).lower()

    if "target" in msg or "y_column" in msg:
        suggestions.append(Suggestion("Target column not found.", "Check the target column name and ensure it exists in your data."))
    if "could not convert string to float" in msg or "could not convert" in msg:
        suggestions.append(
            Suggestion(
                "Non-numeric data detected.",
                "Some features or target contain text. Enable categorical encoding in preprocessing or clean your data.",
            )
        )
    if "nans" in msg or "missing" in msg:
        suggestions.append(
            Suggestion(
                "Missing values detected.",
                "Handle missing values via the preprocessing settings (drop or impute) or clean the CSV.",
            )
        )
    if df is not None and config.get("x_columns"):
        x_cols = config["x_columns"]
        missing = [c for c in x_cols if c not in df.columns]
        if missing:
            suggestions.append(Suggestion("Feature columns missing.", f"These columns are missing: {', '.join(missing)}. Adjust your selection."))
        non_numeric = [c for c in x_cols if c in df.columns and not np.issubdtype(df[c].dtype, np.number)]
        if non_numeric:
            suggestions.append(Suggestion("Non-numeric features detected.", f"Columns {', '.join(non_numeric)} are non-numeric. Encode or remove them."))
    if df is not None and config.get("y_column") in df.columns:
        y = df[config["y_column"]]
        if y.isna().mean() > 0.2:
            suggestions.append(Suggestion("Too many missing values in target.", "Consider dropping or imputing missing target values."))
    if df is not None and config.get("y_column") in df.columns and set(config.get("x_columns", [])) & {config.get("y_column")}:
        suggestions.append(Suggestion("Target appears in features.", "Remove the target from the feature list to avoid leakage."))
    return suggestions
