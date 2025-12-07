# API: explain

- `GlobalExplanation`: feature_names, importance_values, method ("shap" or fallback), summary_text.
- `LocalExplanation`: feature_names, contributions, base_value, predicted_value, method.
- `explain_global(model, X, feature_names)` → GlobalExplanation
- `explain_local(model, X, feature_names, index=0)` → LocalExplanation
- `compute_partial_dependence(model, X, feature_name)` → (grid, mean_predictions)

SHAP is optional; if unavailable, native importances/coefficients are used.
