"""
Basic time-series / forecasting helpers.

Supports naive, ETS (ExponentialSmoothing), ARIMA, and Prophet forecasts with simple train/validation split and backtesting.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Literal, Union

import numpy as np
import pandas as pd
from prophet import Prophet  # type: ignore

try:  # Optional statsmodels import
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    _HAVE_STATSMODELS = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_STATSMODELS = False


@dataclass
class ForecastConfig:
    date_col: str
    target_col: str
    horizon: int = 10
    freq: Optional[str] = None
    model_type: Literal["naive", "ets", "arima", "prophet"] = "naive"
    growth: Literal["linear", "logistic"] = "linear"
    yearly_seasonality: Union[bool, int] = True
    weekly_seasonality: Union[bool, int] = True
    daily_seasonality: Union[bool, int] = False
    holidays_country: Optional[str] = None


@dataclass
class ForecastResult:
    model: Optional[object]
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_pred_valid: pd.Series
    y_forecast: pd.Series
    metrics: Dict[str, float]
    info: Dict[str, object]


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Return a likely datetime column name if found."""
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
        if df[col].astype(str).str.match(r"\d{4}-\d{2}-\d{2}").any():
            return col
    return None


def make_time_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse a date column and set it as the index, sorted by time."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    return df.set_index(date_col)


def _simple_train_valid_split(y: pd.Series, horizon: int):
    split_idx = max(len(y) - horizon, 1)
    y_train = y.iloc[:split_idx]
    y_valid = y.iloc[split_idx:]
    return y_train, y_valid


def _naive_forecast(y_train: pd.Series, horizon: int):
    last_val = y_train.iloc[-1]
    return pd.Series([last_val] * horizon, index=pd.RangeIndex(start=0, stop=horizon, step=1))


def _ets_forecast(y_train: pd.Series, horizon: int):
    if not _HAVE_STATSMODELS:
        raise ImportError("statsmodels is required for ETS forecasting")
    model = ExponentialSmoothing(y_train, trend="add", seasonal=None)
    fitted = model.fit(optimized=True)
    forecast = fitted.forecast(horizon)
    return fitted, forecast


def _backtest_series(y: pd.Series, horizon: int, model_type: str):
    """
    Simple rolling-origin backtest with 3 folds.
    """
    folds = 3
    metrics_list = []
    n = len(y)
    fold_size = max(n // (folds + 1), horizon + 1)
    for i in range(folds):
        start = i * fold_size
        end = start + fold_size
        y_train = y.iloc[: end - horizon]
        y_valid = y.iloc[end - horizon : end]
        if len(y_valid) < horizon:
            break
        cfg = ForecastConfig(date_col="ds", target_col="y", horizon=horizon, model_type=model_type)
        df_train = pd.DataFrame({"ds": y_train.index, "y": y_train.values})
        res = train_forecast_model(df_train, cfg)
        metrics_list.append(res.metrics)
    if not metrics_list:
        return {}
    agg = {}
    for key in metrics_list[0]:
        vals = [m.get(key, np.nan) for m in metrics_list]
        agg[f"backtest_{key}"] = float(np.nanmean(vals))
    return agg


def train_forecast_model(df: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
    """
    Train a simple forecast model and return predictions/metrics.
    Currently supports naive and ETS; other types fall back to naive with a note.
    """

    df_ts = make_time_index(df, config.date_col)
    y = df_ts[config.target_col].astype(float)
    y_train, y_valid = _simple_train_valid_split(y, config.horizon)

    y_pred_valid = pd.Series(dtype=float)
    y_forecast = pd.Series(dtype=float)
    model = None
    info = {"model_type": config.model_type}

    try:
        if config.model_type == "naive":
            y_pred_valid = pd.Series([y_train.iloc[-1]] * len(y_valid), index=y_valid.index)
            y_forecast = _naive_forecast(y_train, config.horizon)
        elif config.model_type == "ets":
            model, forecast = _ets_forecast(y_train, config.horizon)
            y_pred_valid = model.forecast(len(y_valid))
            y_forecast = forecast
        elif config.model_type == "arima":
            try:
                from statsmodels.tsa.arima.model import ARIMA  # type: ignore

                model = ARIMA(y_train, order=(1, 1, 1)).fit()
                y_pred_valid = model.predict(start=len(y_train), end=len(y_train) + len(y_valid) - 1)
                y_forecast = model.forecast(config.horizon)
            except Exception as exc:
                info["warning"] = f"ARIMA unavailable ({exc}); falling back to naive forecast."
                y_pred_valid = pd.Series([y_train.iloc[-1]] * len(y_valid), index=y_valid.index)
                y_forecast = _naive_forecast(y_train, config.horizon)
        elif config.model_type == "prophet":
            try:
                df_prophet = (
                    df_ts.reset_index()[[config.date_col, config.target_col]]
                    .rename(columns={config.date_col: "ds", config.target_col: "y"})
                )
                m = Prophet(
                    growth=config.growth,
                    yearly_seasonality=config.yearly_seasonality,
                    weekly_seasonality=config.weekly_seasonality,
                    daily_seasonality=config.daily_seasonality,
                )
                if config.holidays_country:
                    try:
                        m.add_country_holidays(country_name=config.holidays_country)
                    except Exception:
                        pass
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=config.horizon, freq=config.freq or "D")
                forecast_df = m.predict(future)
                y_forecast = forecast_df.set_index("ds")["yhat"].iloc[-config.horizon:]
                y_pred_valid = forecast_df.set_index("ds")["yhat"].iloc[len(y_train) : len(y_train) + len(y_valid)]
                model = m
            except Exception as exc:
                info["warning"] = f"Prophet failed ({exc}); falling back to naive forecast."
                y_pred_valid = pd.Series([y_train.iloc[-1]] * len(y_valid), index=y_valid.index)
                y_forecast = _naive_forecast(y_train, config.horizon)
        else:
            # Unsupported optional types fallback to naive
            info["warning"] = "Requested model unavailable; falling back to naive forecast."
            y_pred_valid = pd.Series([y_train.iloc[-1]] * len(y_valid), index=y_valid.index)
            y_forecast = _naive_forecast(y_train, config.horizon)
    except Exception as exc:
        info["warning"] = f"Forecasting failed: {exc}. Falling back to naive."
        y_pred_valid = pd.Series([y_train.iloc[-1]] * len(y_valid), index=y_valid.index)
        y_forecast = _naive_forecast(y_train, config.horizon)

    metrics = _forecast_metrics(y_valid, y_pred_valid)
    # Attach backtest metrics if applicable
    try:
        backtest_metrics = _backtest_series(y, config.horizon, config.model_type)
        metrics.update(backtest_metrics)
    except Exception:
        pass

    return ForecastResult(
        model=model,
        train_df=df_ts.iloc[: len(y_train)],
        valid_df=df_ts.iloc[len(y_train):],
        y_train=y_train,
        y_valid=y_valid,
        y_pred_valid=y_pred_valid,
        y_forecast=y_forecast,
        metrics=metrics,
        info=info,
    )


def _forecast_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan")}
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100 if (y_true != 0).any() else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}
