"""
Lightweight AutoML integration for tabular data using FLAML when available.
Provides problem type detection, execution, and summarized results for the GUI.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from modeling_gui.utils import (
    is_regression_target,
)
from modeling_gui.metrics import regression_metrics, classification_metrics, choose_primary_metrics

try:
    from flaml import AutoML

    _FLAML_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _FLAML_AVAILABLE = False


class AutoMLDependencyMissing(Exception):
    """Raised when the optional AutoML dependency is not installed."""


@dataclass
class CandidateModelResult:
    name: str
    estimator: Any
    params: Dict[str, Any]
    metrics: Dict[str, float]
    train_time: float
    is_best: bool = False


@dataclass
class AutoMLRunResult:
    problem_type: str
    candidate_models: List[CandidateModelResult]
    best_candidate: CandidateModelResult
    leaderboard_metrics: List[str]


@dataclass
class AutoMLResult:
    best_model: Any
    summary: str
    metrics: Dict[str, Any]
    problem_type: str
    plots: Dict[str, Any]


def detect_problem_type(df: pd.DataFrame, y_col: str) -> Tuple[Optional[str], float]:
    """
    Heuristically detect whether the target column represents regression or classification.

    Returns a tuple of (problem_type, confidence) where problem_type is one of:
    "regression", "classification", or None when unsupported/unknown.
    """
    if y_col not in df.columns:
        return None, 0.0
    y = df[y_col]
    unique_count = y.nunique(dropna=True)
    if unique_count < 2:
        return None, 0.0

    if is_regression_target(y):
        ratio_unique = unique_count / max(len(y), 1)
        # Many unique numeric values: likely regression
        if unique_count > 20 or ratio_unique > 0.05:
            return "regression", 0.8
        # Low unique numeric: could be encoded classes
        return "classification", 0.5
    else:
        return "classification", 0.8


def _ensure_flaml_available():
    if not _FLAML_AVAILABLE:
        raise AutoMLDependencyMissing(
            "AutoML features require the optional dependency 'flaml'. "
            "Install with: pip install modeling-gui[automl]"
        )


def run_automl(
    df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    settings: Optional[Dict[str, Any]] = None,
    problem_type: Optional[str] = None,
) -> AutoMLRunResult:
    """
    Execute an AutoML sweep on the provided dataframe.

    Returns AutoMLResult containing the fitted best model, metrics, and plot-ready data.
    """
    _ensure_flaml_available()
    if settings is None:
        settings = {}

    if y_col not in df.columns:
        raise ValueError("Target column not found in dataframe.")

    y = df[y_col]
    resolved_type, _ = detect_problem_type(df, y_col)
    if problem_type:
        resolved_type = problem_type
    if resolved_type not in ("regression", "classification"):
        raise ValueError("Unsupported or ambiguous problem type for AutoML.")

    task = "regression" if resolved_type == "regression" else "classification"
    metric = "r2" if task == "regression" else "accuracy"

    X = df[x_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.get("test_size", 0.2),
        random_state=settings.get("random_state", 42),
        stratify=y if task == "classification" else None,
    )

    candidates: List[CandidateModelResult] = []
    leaderboard_metrics = choose_primary_metrics(settings.get("domain", "Generic"), task)

    def add_candidate(name, est):
        est.fit(X_train, y_train)
        if task == "regression":
            y_pred = est.predict(X_test)
            mets = regression_metrics(y_test, y_pred)
        else:
            y_pred = est.predict(X_test)
            y_proba = est.predict_proba(X_test) if hasattr(est, "predict_proba") else None
            mets = classification_metrics(y_test, y_pred, y_proba)
        candidates.append(
            CandidateModelResult(
                name=name,
                estimator=est,
                params=getattr(est, "get_params", lambda: {})(),
                metrics=mets,
                train_time=0.0,
            )
        )

    try:
        automl = AutoML()
        automl_settings = {
            "time_budget": settings.get("time_budget", 30),
            "task": task,
            "metric": metric,
            "log_file_name": settings.get("log_file_name", None),
            "verbose": settings.get("verbose", 0),
            "n_splits": settings.get("n_splits", 5),
        }
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            **automl_settings,
        )
        y_pred = automl.predict(X_test)
        if task == "regression":
            mets = regression_metrics(y_test, y_pred)
        else:
            y_proba = automl.predict_proba(X_test) if hasattr(automl, "predict_proba") else None
            mets = classification_metrics(y_test, y_pred, y_proba)
        candidates.append(
            CandidateModelResult(
                name=str(automl.best_estimator),
                estimator=automl.model,
                params=getattr(automl, "best_config", {}),
                metrics=mets,
                train_time=0.0,
            )
        )
    except Exception:
        if not _HAVE_FLAML:
            pass  # optional dependency missing, fall back below
    # Fallback simple candidates
    if task == "regression":
        add_candidate("RandomForestRegressor", RandomForestRegressor(n_estimators=100))
        add_candidate("GradientBoostingRegressor", GradientBoostingRegressor())
        add_candidate("LinearRegression", LinearRegression())
    else:
        add_candidate("RandomForestClassifier", RandomForestClassifier(n_estimators=100))
        add_candidate("GradientBoostingClassifier", GradientBoostingClassifier())
        add_candidate("LogisticRegression", LogisticRegression(max_iter=200))

    # pick best by primary metric
    def score_candidate(cand: CandidateModelResult):
        if leaderboard_metrics:
            return cand.metrics.get(leaderboard_metrics[0], float("-inf"))
        return cand.metrics.get("accuracy", cand.metrics.get("R2", float("-inf")))

    candidates_sorted = sorted(candidates, key=score_candidate, reverse=True)
    if candidates_sorted:
        candidates_sorted[0].is_best = True
    best = candidates_sorted[0] if candidates_sorted else candidates[0]

    # build plots info (best only)
    plots = {}
    if task == "regression":
        y_pred_best = best.estimator.predict(X_test)
        plots["residuals"] = (y_test, y_pred_best)
    else:
        y_pred_best = best.estimator.predict(X_test)
        plots["confusion"] = (y_test, y_pred_best)
        importances = getattr(best.estimator, "feature_importances_", None)
        if importances is not None:
            plots["feature_importance"] = (importances, x_cols)

    summary = f"Best model: {best.name}"

    return AutoMLRunResult(
        problem_type=task,
        candidate_models=candidates_sorted,
        best_candidate=best,
        leaderboard_metrics=leaderboard_metrics or [metric],
    )
