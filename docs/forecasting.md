# Forecasting

Modeling-GUI includes basic time-series forecasting for tabular data with a timestamp column.

## Workflow

1. Load a CSV with a date/time column and numeric target.
2. Select **Forecast** mode (or allow Smart Analyze to detect forecasting in suitable domains).
3. Configure:
   - Time column (auto-detected when possible)
   - Target column
   - Horizon (steps ahead)
   - Model type (naive, ETS, ARIMA, Prophet)
4. Run and view:
   - Actual vs fitted vs forecast plot
   - Backtest/validation overlay
   - Metrics: RMSE, MAE, MAPE

## Example

```text
Date,Sales
2023-01-01,120
2023-01-02,118
...
```

After selecting Date as time column and Sales as target, choose horizon=14 and run. The forecast plot will show train, validation, and forward forecast.

CLI/API snippet (advanced users):

```python
from modeling_gui.forecasting import ForecastConfig, train_forecast_model
import pandas as pd

df = pd.read_csv("sales.csv")
cfg = ForecastConfig(date_col="Date", target_col="Sales", horizon=14, model_type="ets")
result = train_forecast_model(df, cfg)
print(result.metrics)  # {'RMSE': ..., 'MAE': ..., 'MAPE': ...}
```

See `examples/forecasting_basic.md` for a GUI walkthrough.
For date parsing and categorical handling tips, see `examples/preprocessing_dates_and_categoricals.md`.

## Screenshots

- Forecast plot: `docs/screenshots/main_window.png` (placeholder)

## Notes

- ETS/naive/ARIMA/Prophet are installed out of the box.
- Metrics emphasize RMSE/MAE/MAPE (Finance defaults to RMSE/MAPE; Science emphasizes RMSE/R²).
- The leaderboard will include forecasting candidates when Smart Analyze evaluates them; primary metric follows the active domain preset.
- For single-feature exponential-style behavior, see `examples/curve_fitting_gaussian_exponential.md` for a simple growth/decay fit.
- **Prophet**: included by default; supports growth (linear/logistic), seasonality toggles (yearly/weekly/daily), and optional country holidays from the Forecasting panel.
- **Backtesting**: enable “Run backtest” to compute averaged RMSE/MAE/MAPE over rolling splits; use to assess stability beyond a single holdout.

## Using Prophet

- Available by default. Recommended when trends/seasonality/holidays matter.
- Configure in the Forecasting panel:
  - Growth: **linear** (default) or **logistic**.
  - Seasonality toggles: yearly/weekly/daily; optionally set numeric overrides.
  - Holidays: enable “Use country holidays” and choose a country (e.g., US/UK/DE/FR/IN).
- Outputs: forecast with yhat values; metrics include RMSE/MAE/MAPE plus backtest_* if backtesting is enabled.
