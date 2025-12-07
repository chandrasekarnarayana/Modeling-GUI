# Regression & Residual Plots

Use this walkthrough to find and interpret regression and residual plots in Modeling-GUI.

## Steps

1) **Load data**
   - Use the demo CSV (`target` for regression) or your own numeric dataset.

2) **Run a regression model**
   - Basic mode: click **Smart Analyze**.
   - Expert mode: pick **OLS** or **Gradient Boosting (regression)** and click **Run selected model**.

3) **Open plots**
   - After the run, the app shows the regression fit plot (predictions vs feature) and a **Residual Plot** (residuals vs predicted).
   - Use **Save plot…** to export as PNG/PDF.

4) **Interpretation**
   - Regression plot: look for alignment of predictions with observed values.
   - Residual plot: residuals should be centered around 0 without clear patterns; strong trends/patterns indicate model misspecification or heteroscedasticity.

5) **Next steps**
   - Try **Gradient Boosting** for non-linear relationships (see `examples/regression_gradient_boosting_basic.md`).
   - Compare models in the leaderboard and check residual patterns for each.
