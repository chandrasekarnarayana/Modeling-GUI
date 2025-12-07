# API: data_quality

- `DataIssue`: severity, code, message, suggestion.
- Checks:
  - `validate_schema(df, expected_schema=None)`
  - `detect_missing_issues(df, target_col)` / `detect_missing_target`
  - `detect_type_mismatches(df)`
  - `detect_potential_leakage(X_cols, y_col)`
  - `detect_imbalance(y)` / `detect_class_imbalance`
  - `detect_non_numeric_features(df, x_cols)`

These helpers power the Data Quality panel and troubleshooting hints.
