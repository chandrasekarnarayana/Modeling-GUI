# API: drift

- `TrainingSnapshot`: rows, feature_stats, col_types, timestamp.
- `compute_snapshot(df, col_types)` → TrainingSnapshot
- `compare_snapshot(train_snapshot, df_new, col_types)` → list[DataIssue] indicating drift severity.

Used to warn about distribution changes during batch prediction compared to training data.
