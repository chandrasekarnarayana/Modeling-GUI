import pandas as pd
import pytest

from modeling_gui.forecasting import ForecastConfig, train_forecast_model


def _toy_series():
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    values = pd.Series(range(20)) + 0.5
    return pd.DataFrame({"date": dates, "target": values})


def test_forecast_naive_returns_metrics_and_predictions():
    df = _toy_series()
    cfg = ForecastConfig(date_col="date", target_col="target", horizon=5, model_type="naive")
    result = train_forecast_model(df, cfg)
    assert len(result.y_forecast) == cfg.horizon
    for key in ("RMSE", "MAE", "MAPE"):
        assert key in result.metrics


def test_forecast_ets_has_predictions_when_statsmodels_available():
    df = _toy_series()
    cfg = ForecastConfig(date_col="date", target_col="target", horizon=4, model_type="ets")
    result = train_forecast_model(df, cfg)
    assert len(result.y_forecast) == cfg.horizon
    assert result.metrics


def test_forecast_arima_optional():
    df = _toy_series()
    cfg = ForecastConfig(date_col="date", target_col="target", horizon=3, model_type="arima")
    result = train_forecast_model(df, cfg)
    assert len(result.y_forecast) == cfg.horizon
