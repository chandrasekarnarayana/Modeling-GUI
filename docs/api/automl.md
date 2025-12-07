# API: automl

Key functions and classes:

- `run_automl(df, x_cols, y_col, settings, problem_type=None) -> AutoMLRunResult`
- `CandidateModelResult`: name, estimator, params, metrics, train_time, is_best
- `AutoMLRunResult`: problem_type, candidate_models, best_candidate, leaderboard_metrics

Typical usage:

```python
from modeling_gui.automl import run_automl
result = run_automl(df, ["x1", "x2"], "target", settings={"time_budget": 30, "domain": "Generic"})
best = result.best_candidate
print(best.name, best.metrics)
```

Metrics rely on `modeling_gui.metrics` and domain-aware primary metrics.
