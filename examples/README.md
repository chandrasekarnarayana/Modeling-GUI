# Examples

- **Bundled demo CSV**: `modeling_gui/data/demo_quickstart.csv`
  - Load it via the **Load Demo Dataset** button in the app.
  - Columns:
    - Regression target: `target`
    - Binary classification target: `target_class`
    - Features: `feature_1`, `feature_2`, `feature_3`

To try manually:
1. Start the app (`run_modeling_gui`).
2. Click **Load Demo Dataset**.
3. Choose `target` (regression) or `target_class` (classification) as what you want to predict; leave features preselected.
4. Click **Smart Analyze** and review metrics, plots, and the leaderboard.

Additional walkthroughs:
- `examples/classification_binary_basic.md`
- `examples/classification_multiclass_basic.md`
- `examples/clustering_kmeans_basic.md`
- `examples/forecasting_basic.md`
- `examples/regression_advanced_ls.md`
