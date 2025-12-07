# FAQ

**AutoML missing?**
- Install optional extras: `pip install "modeling-gui[automl]"`.

**SHAP missing?**
- Install with `pip install shap` or proceed with native feature importances.

**PyQt errors on headless servers?**
- Use a virtual display (e.g., `xvfb-run`) or run locally with a GUI.

**Graphviz issues?**
- Ensure system Graphviz is installed for tree diagrams.

**Large CSV performance?**
- Select only relevant columns; consider sampling for quick experiments.

**Model won’t train?**
- Check Data Quality panel for missing targets, non-numeric features, or leakage. Try simplifying features or encoding categoricals.
