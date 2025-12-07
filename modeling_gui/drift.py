"""
Simple data drift utilities.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from modeling_gui.data_quality import DataIssue


@dataclass
class TrainingSnapshot:
    rows: int
    feature_stats: Dict[str, Dict[str, float]]
    col_types: Dict[str, str]
    timestamp: str = ""


def compute_snapshot(df: pd.DataFrame, col_types: Dict[str, str]) -> TrainingSnapshot:
    stats = {}
    for col, ctype in col_types.items():
        if col not in df.columns:
            continue
        if ctype == "numeric":
            series = pd.to_numeric(df[col], errors="coerce")
            stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        elif ctype == "categorical":
            top = df[col].value_counts(normalize=True).head(5)
            stats[col] = top.to_dict()
    return TrainingSnapshot(rows=len(df), feature_stats=stats, col_types=col_types)


def compare_snapshot(train: TrainingSnapshot, df_new: pd.DataFrame, col_types: Dict[str, str]) -> List[DataIssue]:
    issues: List[DataIssue] = []
    new_snap = compute_snapshot(df_new, col_types)
    for col, stat in train.feature_stats.items():
        if col not in new_snap.feature_stats:
            continue
        drift_val = 0.0
        if isinstance(stat, dict) and all(isinstance(v, float) for v in stat.values()):
            train_mean = stat.get("mean", 0.0)
            new_mean = new_snap.feature_stats[col].get("mean", train_mean)
            drift_val = abs(new_mean - train_mean) / (abs(train_mean) + 1e-9)
        else:
            train_counts = stat
            new_counts = new_snap.feature_stats[col]
            keys = set(train_counts) | set(new_counts)
            drift_val = sum(abs(train_counts.get(k, 0) - new_counts.get(k, 0)) for k in keys)
        if drift_val > 0.5:
            issues.append(DataIssue(severity="warning", code="drift", message=f"Strong drift detected for {col}", suggestion="Consider retraining with new data."))
        elif drift_val > 0.2:
            issues.append(DataIssue(severity="info", code="drift", message=f"Moderate drift detected for {col}", suggestion="Monitor predictions; retrain if performance drops."))
    return issues
