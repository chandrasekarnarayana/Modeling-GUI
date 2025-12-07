# Forecasting (basic GUI walkthrough)

Dataset: use the bundled `modeling_gui/data/demo_quickstart.csv` or any CSV with a date column and numeric target.

Steps:
1) Open the app → **Load CSV** (or **Load Demo Dataset**).  
2) Select target column (e.g., `target`) and ensure the date column is selected in the Forecast controls.  
3) In Expert mode, choose a forecasting model: **naive**, **ets**, **arima**, or **prophet** (all installed by default).  
4) Set forecast horizon (e.g., 12 steps) and click **Run forecast**.  
5) Read metrics (RMSE/MAE/MAPE) and inspect the forecast plot (train vs validation vs forecast).  
6) Optional: save the project to store the training snapshot for drift checks later.

Notes:
- Prophet is available out of the box; configure growth (linear/logistic), seasonality toggles, and optional country holidays.
- Finance domain highlights RMSE/MAPE; Science emphasizes RMSE/R².***

## Advanced: Prophet + Backtesting

- Select **Prophet** from the forecast model list for smoother trends/seasonality.
- Toggle yearly/weekly/daily seasonality and choose growth (linear/logistic); optionally enable country holidays (e.g., US).
- Check **Run backtest** to compute averaged RMSE/MAE/MAPE across rolling splits; compare to single holdout metrics.
- Use backtest metrics to gauge stability; large gaps may indicate overfitting or non-stationarity.***

## Forecast plot interpretation

- The forecast plot shows train (blue), validation (orange), predicted-on-validation (green dashed), and forward forecast (red dotted).
- Use **Save plot…** to export the figure.
- Check alignment on validation; large gaps suggest model adjustments are needed.
