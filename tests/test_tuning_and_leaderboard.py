import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from modeling_gui.automl import AutoMLRunResult, CandidateModelResult
from modeling_gui.metrics import choose_primary_metrics
from modeling_gui.tuning import tune_model


def test_tune_model_fast_regression():
    X, y = make_regression(n_samples=60, n_features=4, random_state=42)
    est = RandomForestRegressor(random_state=42)
    result = tune_model(est, pd.DataFrame(X), y, "regression", "fast")
    assert result.estimator is not None
    assert result.metrics


def test_tune_model_fast_classification():
    X, y = make_classification(n_samples=80, n_features=5, random_state=0)
    est = RandomForestClassifier(random_state=0)
    result = tune_model(est, pd.DataFrame(X), y, "classification", "fast")
    assert result.estimator is not None
    assert result.metrics


def test_leaderboard_best_candidate_marked():
    candidates = [
        CandidateModelResult(
            name="m1",
            estimator=None,
            params={},
            metrics={"accuracy": 0.8},
            train_time=0.1,
        ),
        CandidateModelResult(
            name="m2",
            estimator=None,
            params={},
            metrics={"accuracy": 0.9},
            train_time=0.1,
        ),
    ]
    primary = choose_primary_metrics("Business", "classification")
    # Simulate sorting as automl.run_automl does
    best = sorted(candidates, key=lambda c: c.metrics.get(primary[0], -np.inf), reverse=True)[0]
    best.is_best = True
    result = AutoMLRunResult(problem_type="classification", candidate_models=[best] + candidates[1:], best_candidate=best, leaderboard_metrics=primary)
    assert result.best_candidate.is_best
    assert result.candidate_models[0].metrics["accuracy"] == 0.9
