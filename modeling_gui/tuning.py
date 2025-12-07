"""
Guided hyperparameter tuning presets for supported models.
"""

from typing import Any
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import make_scorer, r2_score, accuracy_score

from modeling_gui.automl import CandidateModelResult
from modeling_gui.metrics import regression_metrics, classification_metrics


def _cv(problem_type: str, n_splits: int = 3):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if problem_type == "classification" else KFold(n_splits=n_splits, shuffle=True, random_state=42)


def _space_for(estimator):
    if isinstance(estimator, (RandomForestRegressor, RandomForestClassifier)):
        return {
            "n_estimators": [50, 100, 200, 400],
            "max_depth": [None, 5, 10, 20],
            "max_features": ["auto", "sqrt", 0.5],
        }
    if isinstance(estimator, (GradientBoostingRegressor, GradientBoostingClassifier)):
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
        }
    return {}


def tune_model(estimator, X, y, problem_type: str, preset: str) -> CandidateModelResult:
    """
    Hyperparameter tuning using RandomizedSearchCV with preset budgets.
    preset in {"fast", "balanced", "thorough"}.
    """
    preset_iter = {"fast": 5, "balanced": 15, "thorough": 30}
    n_iter = preset_iter.get(preset, 5)
    space = _space_for(estimator)
    if not space:
        return CandidateModelResult(
            name=estimator.__class__.__name__,
            estimator=estimator.fit(X, y),
            params=getattr(estimator, "get_params", lambda: {})(),
            metrics={},
            train_time=0.0,
        )
    scorer = make_scorer(r2_score) if problem_type == "regression" else make_scorer(accuracy_score)
    search = RandomizedSearchCV(
        estimator,
        space,
        n_iter=n_iter,
        cv=_cv(problem_type),
        scoring=scorer,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X, y)
    best_est = search.best_estimator_
    if problem_type == "regression":
        mets = regression_metrics(y, best_est.predict(X))
    else:
        mets = classification_metrics(y, best_est.predict(X))
    return CandidateModelResult(
        name=best_est.__class__.__name__ + " (tuned)",
        estimator=best_est,
        params=search.best_params_,
        metrics=mets,
        train_time=0.0,
        is_best=False,
    )
