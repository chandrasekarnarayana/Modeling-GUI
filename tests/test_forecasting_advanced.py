import pandas as pd
import numpy as np
import pytest

from modeling_gui.forecasting import ForecastConfig, train_forecast_model


def _series():
    dates = pd.date_range("2023-01-01", periods=25, freq="D")
    values = pd.Series(np.linspace(0, 10, 25))
    return pd.DataFrame({"date": dates, "target": values})


def test_backtest_metrics_naive_and_ets():
    df = _series()
    cfg_naive = ForecastConfig(date_col="date", target_col="target", horizon=3, model_type="naive")
    res_naive = train_forecast_model(df, cfg_naive)
    assert any(k.startswith("backtest_") for k in res_naive.metrics)
    cfg_ets = ForecastConfig(date_col="date", target_col="target", horizon=3, model_type="ets")
    res_ets = train_forecast_model(df, cfg_ets)
    assert any(k.startswith("backtest_") for k in res_ets.metrics)


def test_prophet_optional_forecast():
    df = _series()
    cfg = ForecastConfig(date_col="date", target_col="target", horizon=2, model_type="prophet")
    res = train_forecast_model(df, cfg)
    assert len(res.y_forecast) == cfg.horizon
    for key in ("RMSE", "MAE", "MAPE"):
        assert key in res.metrics
