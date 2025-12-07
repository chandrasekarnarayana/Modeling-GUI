import numpy as np
import pandas as pd
import statsmodels.api as sm
from functools import partial
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from scipy.optimize import curve_fit


class _StatsModelWrapper:
    """
    Lightweight wrapper around statsmodels results to add a constant term when
    making predictions. This keeps the public API simple for callers that
    supply raw feature arrays.
    """

    def __init__(self, result, exog_preparer):
        self._result = result
        self._prepare_exog = exog_preparer

    def predict(self, X):
        exog = self._prepare_exog(X)
        return self._result.predict(exog)

    def __getattr__(self, name):
        return getattr(self._result, name)

    def __getstate__(self):
        return {"_result": self._result, "_prepare_exog": self._prepare_exog}

    def __setstate__(self, state):
        self._result = state["_result"]
        self._prepare_exog = state["_prepare_exog"]


def gaussian_curve(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev**2))


def exponential_curve(x, a, b, c):
    return a * np.exp(b * x) + c


class ModelManager:
    def __init__(self):
        self.model = None

    # --- Internal helpers ---

    @staticmethod
    def _prepare_exog(X):
        """Ensure the design matrix is two-dimensional and contains a constant."""
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return sm.add_constant(X_arr, has_constant="add")

    # --- Statistical Models ---

    def ols(self, X, Y):
        """Ordinary Least Squares (OLS) Regression."""
        try:
            X_design = self._prepare_exog(X)
            result = sm.OLS(Y, X_design).fit()
            self.model = _StatsModelWrapper(result, self._prepare_exog)
            return self.model
        except Exception as e:
            raise Exception(f"OLS Model Error: {str(e)}")

    def wls(self, X, Y, weights):
        """Weighted Least Squares (WLS) Regression."""
        try:
            X_design = self._prepare_exog(X)
            result = sm.WLS(Y, X_design, weights=weights).fit()
            self.model = _StatsModelWrapper(result, self._prepare_exog)
            return self.model
        except Exception as e:
            raise Exception(f"WLS Model Error: {str(e)}")

    def gls(self, X, Y, sigma):
        """Generalized Least Squares (GLS) Regression."""
        try:
            X_design = self._prepare_exog(X)
            result = sm.GLS(Y, X_design, sigma=sigma).fit()
            self.model = _StatsModelWrapper(result, self._prepare_exog)
            return self.model
        except Exception as e:
            raise Exception(f"GLS Model Error: {str(e)}")

    def recursive_ls(self, X, Y):
        """Recursive Least Squares (Recursive LS) Regression."""
        try:
            X_design = self._prepare_exog(X)
            result = sm.RecursiveLS(Y, X_design).fit()
            self.model = _StatsModelWrapper(result, self._prepare_exog)
            return self.model
        except Exception as e:
            raise Exception(f"Recursive LS Model Error: {str(e)}")

    def rlm(self, X, Y):
        """Robust Linear Model (RLM) Regression."""
        try:
            X_design = self._prepare_exog(X)
            result = sm.RLM(Y, X_design).fit()
            self.model = _StatsModelWrapper(result, self._prepare_exog)
            return self.model
        except Exception as e:
            raise Exception(f"RLM Model Error: {str(e)}")

    def rolling_ls(self, X, Y, window=5):
        """Rolling Least Squares (RLS) Regression."""
        try:
            X_df = pd.DataFrame(self._prepare_exog(X))
            y_series = pd.Series(np.asarray(Y).reshape(-1), name="target")
            combined = pd.concat([X_df, y_series], axis=1)

            params = []
            indices = []
            for start in range(len(combined) - window + 1):
                window_frame = combined.iloc[start : start + window]
                y_win = window_frame["target"]
                x_win = window_frame.drop(columns="target")
                res = sm.OLS(y_win, x_win).fit()
                params.append(res.params.values)
                indices.append(window_frame.index[-1])

            if not params:
                return pd.DataFrame()

            param_count = len(params[0])
            column_names = [f"beta_{i}" for i in range(param_count)]
            return pd.DataFrame(params, index=indices, columns=column_names)
        except Exception as e:
            raise Exception(f"Rolling LS Model Error: {str(e)}")

    # --- Machine Learning Models ---

    def random_forest(self, X, Y, n_estimators=100, max_depth=None):
        """Random Forest (Classification/Regression)."""
        try:
            if getattr(Y, "dtype", np.array(Y).dtype).kind in "if":
                self.model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth
                )
            self.model.fit(X, Y)
            return self.model
        except Exception as e:
            raise Exception(f"Random Forest Model Error: {str(e)}")

    def gradient_boost(self, X, Y, n_estimators=100, learning_rate=0.1, max_depth=None):
        """Gradient Boosting (Classification/Regression)."""
        try:
            if getattr(Y, "dtype", np.array(Y).dtype).kind in "if":
                self.model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                )
            else:
                self.model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                )
            self.model.fit(X, Y)
            return self.model
        except Exception as e:
            raise Exception(f"Gradient Boosting Model Error: {str(e)}")

    # --- Clustering ---

    def kmeans_clustering(self, X, n_clusters=3):
        """KMeans Clustering with customizable number of clusters."""
        try:
            self.model = KMeans(n_clusters=n_clusters, n_init="auto")
            self.model.fit(X)
            return self.model
        except Exception as e:
            raise Exception(f"KMeans Clustering Error: {str(e)}")

    # --- Advanced Data Fitting ---

    def gaussian_fit(self, X, Y):
        """Gaussian fitting with mean/std plus a callable for predictions."""
        try:
            x_arr = np.asarray(X).flatten()
            y_arr = np.asarray(Y).flatten()

            amp_guess = float(np.max(y_arr)) if y_arr.size else 1.0
            mean_guess = float(np.mean(x_arr)) if x_arr.size else 0.0
            std_guess = float(np.std(x_arr)) if np.std(x_arr) > 0 else 1.0
            params, _ = curve_fit(
                gaussian_curve,
                x_arr,
                y_arr,
                p0=[amp_guess, mean_guess, std_guess],
                maxfev=10000,
            )
            amplitude, mean, std_dev = params
            std_dev = abs(std_dev) if std_dev != 0 else 1e-8

            gaussian_fn = partial(
                gaussian_curve, amplitude=amplitude, mean=mean, std_dev=std_dev
            )
            self.model = gaussian_fn
            return mean, std_dev, gaussian_fn
        except Exception as e:
            raise Exception(f"Gaussian Fitting Error: {str(e)}")

    def gaussian_fitting(self, X, Y):
        """Backward-compatible alias for gaussian_fit."""
        return self.gaussian_fit(X, Y)

    def exponential_fit(self, X, Y):
        """Exponential growth/decay fitting."""
        try:
            x_arr = np.asarray(X).flatten()
            y_arr = np.asarray(Y).flatten()

            a_guess = float(np.max(y_arr)) if y_arr.size else 1.0
            b_guess = 0.1
            c_guess = float(np.min(y_arr)) if y_arr.size else 0.0
            params, _ = curve_fit(
                exponential_curve,
                x_arr,
                y_arr,
                p0=[a_guess, b_guess, c_guess],
                maxfev=10000,
            )
            params = np.asarray(params, dtype=float)
            params[0] = abs(params[0])
            exp_fn = partial(
                exponential_curve, a=params[0], b=params[1], c=params[2]
            )
            self.model = exp_fn
            return tuple(params)
        except Exception as e:
            raise Exception(f"Exponential Fitting Error: {str(e)}")

    def exponential_fitting(self, X, Y):
        """Backward-compatible alias for exponential_fit."""
        return self.exponential_fit(X, Y)

    def get_summary(self):
        """Get the summary of the model if available."""
        if hasattr(self.model, "summary"):
            return self.model.summary()
        raise Exception("Summary is not available for this model.")

    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        if callable(self.model):
            return self.model(X)
        return self.model.predict(X)
