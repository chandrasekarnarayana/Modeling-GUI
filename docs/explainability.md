# Explainability

Modeling-GUI offers explainability for both global and local views, with optional SHAP integration when available.

## Global Explanations

- Uses SHAP (if installed) to compute mean |SHAP| per feature.
- Falls back to native feature importances or coefficients when SHAP is unavailable.
- Shown as a bar chart in the **Explain** tab, with a short summary of top drivers.

## Local Explanations

- Select a row (by index) to view per-feature contributions.
- Displays predicted value vs baseline and contributions (SHAP or fallback).

## Partial Dependence

- 1D partial dependence helper (`compute_partial_dependence`) sweeps a feature over its range and plots mean predictions.
- Used in scenario testing and interactive feature importance views.

## Scenario Testing (What-if)

- In Expert mode, adjust sliders for top features (ranges derived from data).
- Predictions update live for a hypothetical row; coach bar describes the delta vs baseline.

### Quick GUI walkthrough

1. Run **Smart Analyze** or train a model manually.
2. Open the **Explain** tab → click “Show global importance plot”.
3. Choose a row index and click “Explain selected row” for local contributions.
4. Switch to **Scenario testing** tab:
   - Pick a base row.
   - Choose a top feature and move the slider.
   - Read the updated prediction and delta text.
5. Optional: click “Show partial dependence” to see how predictions change over the full feature range.

Screenshot placeholders:
- `![Global importance](media/global_importance.png)`
- `![Scenario testing](media/scenario_testing.png)`

## Notes

- SHAP is optional; install with `pip install shap` to enable advanced plots.
- Basic mode hides complexity; Expert mode exposes the Explain tab fully.
- See `examples/explainability_basic.md` and `examples/scenario_testing_basic.md` for full walkthroughs.
