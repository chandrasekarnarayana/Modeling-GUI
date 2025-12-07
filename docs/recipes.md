# Recipes

## Business: Predict Customer Churn (Classification)

- Domain preset: **Business**
- Primary metric: **ROC AUC** (or macro F1 if classes imbalanced)
- How-to:
  1) Load a churn-style CSV (or the demo dataset’s `target_class`).
  2) Set target to the binary churn flag; keep other columns as features.
  3) Smart Analyze (or choose Gradient Boosting/Random Forest).
  4) Review confusion matrix, ROC/PR metrics; use Explain tab for top drivers.
  5) Scenario test tenure/spend sliders to see impact on churn risk.
- See `examples/classification_binary_basic.md` for a detailed walkthrough.

## Finance: Forecast Monthly Revenue with ETS/ARIMA

- Domain preset: **Finance**
- Primary metrics: **MAPE**, **RMSE**
- How-to:
  1) Load a time-series CSV (date + revenue).
  2) Forecast mode → choose time column, target, horizon (e.g., 12 months).
  3) Model type: ETS (default) or ARIMA (if installed); enable backtest for fold-averaged metrics.
  4) Review forecast plot and backtest metrics; watch coach bar for drift hints.
- See `examples/forecasting_basic.md` (advanced section) for Prophet/ARIMA notes.

## Science: Calibration Curve with Gaussian/Exponential Fit

- Domain preset: **Science**
- Primary metrics: **R²**, **RMSE**
- How-to:
  1) Load calibration CSV (x = concentration, y = response).
  2) Select **Gaussian** (peak) or **Exponential** (decay/growth) fit.
  3) Run fit; view curve overlay with parameter legend; check residuals.
  4) Export plot if needed; compare against OLS in the leaderboard.
- See `examples/curve_fitting_gaussian_exponential.md` for step-by-step.

## Business/Generic: Customer Segmentation with KMeans

- Domain preset: **Business** or **Generic**
- Goal: group customers by behavior/spend
- How-to:
  1) Load CSV with numeric attributes (spend, visits, age, etc.).
  2) Expert mode → select numeric features → choose **KMeans Clustering**.
  3) Set k (start with 3–5) in the KMeans dialog; run.
  4) Review cluster centers to interpret segments; export labels or plot scatter by cluster.
- See `examples/clustering_kmeans_basic.md` for a detailed walkthrough.
