import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

class ModelManager:
    def __init__(self):
        self.model = None

    # --- Statistical Models ---

    def ols(self, X, Y):
        """Ordinary Least Squares (OLS) Regression."""
        try:
            X = sm.add_constant(X)
            self.model = sm.OLS(Y, X).fit()
            return self.model
        except Exception as e:
            raise Exception(f"OLS Model Error: {str(e)}")

    def wls(self, X, Y, weights):
        """Weighted Least Squares (WLS) Regression."""
        try:
            X = sm.add_constant(X)
            self.model = sm.WLS(Y, X, weights=weights).fit()
            return self.model
        except Exception as e:
            raise Exception(f"WLS Model Error: {str(e)}")

    def gls(self, X, Y, sigma):
        """Generalized Least Squares (GLS) Regression."""
        try:
            X = sm.add_constant(X)
            self.model = sm.GLS(Y, X, sigma=sigma).fit()
            return self.model
        except Exception as e:
            raise Exception(f"GLS Model Error: {str(e)}")

    def recursive_ls(self, X, Y):
        """Recursive Least Squares (Recursive LS) Regression."""
        try:
            X = sm.add_constant(X)
            self.model = sm.RecursiveLS(Y, X).fit()
            return self.model
        except Exception as e:
            raise Exception(f"Recursive LS Model Error: {str(e)}")

    def rlm(self, X, Y):
        """Robust Linear Model (RLM) Regression."""
        try:
            X = sm.add_constant(X)
            self.model = sm.RLM(Y, X).fit()
            return self.model
        except Exception as e:
            raise Exception(f"RLM Model Error: {str(e)}")

    def rolling_ls(self, X, Y, window=5):
        """Rolling Least Squares (RLS) Regression."""
        try:
            X = sm.add_constant(X)
            rolling_model = pd.concat([X, Y], axis=1).rolling(window=window).apply(
                lambda x: sm.OLS(x[:, 1], x[:, 0]).fit().params
            )
            return rolling_model
        except Exception as e:
            raise Exception(f"Rolling LS Model Error: {str(e)}")

    # --- Machine Learning Models ---

    def random_forest(self, X, Y, n_estimators=100, max_depth=None):
        """Random Forest (Classification/Regression)."""
        try:
            if Y.dtype.kind in 'if':  # Regression for numerical targets
                self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            else:  # Classification for categorical targets
                self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            self.model.fit(X, Y)
            return self.model
        except Exception as e:
            raise Exception(f"Random Forest Model Error: {str(e)}")

    def gradient_boost(self, X, Y, n_estimators=100, learning_rate=0.1, max_depth=None):
        """Gradient Boosting (Classification/Regression)."""
        try:
            if Y.dtype.kind in 'if':  # Regression for numerical targets
                self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            else:  # Classification for categorical targets
                self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            self.model.fit(X, Y)
            return self.model
        except Exception as e:
            raise Exception(f"Gradient Boosting Model Error: {str(e)}")

    # --- Clustering ---

    def kmeans_clustering(self, X, n_clusters=3):
        """KMeans Clustering with customizable number of clusters."""
        try:
            self.model = KMeans(n_clusters=n_clusters)
            self.model.fit(X)
            return self.model
        except Exception as e:
            raise Exception(f"KMeans Clustering Error: {str(e)}")

    # --- Advanced Data Fitting ---

    def gaussian_fitting(self, X, Y):
        """Gaussian Fitting."""
        try:
            def gaussian(x, a, x0, sigma):
                return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
            params, _ = curve_fit(gaussian, X, Y)
            return params
        except Exception as e:
            raise Exception(f"Gaussian Fitting Error: {str(e)}")

    def exponential_fitting(self, X, Y):
        """Exponential Growth/Decay Fitting."""
        try:
            def exponential(x, a, b, c):
                return a * np.exp(b * x) + c
            params, _ = curve_fit(exponential, X, Y)
            return params
        except Exception as e:
            raise Exception(f"Exponential Fitting Error: {str(e)}")

    def get_summary(self):
        """Get the summary of the model if available."""
        if hasattr(self.model, 'summary'):
            return self.model.summary()
        else:
            raise Exception("Summary is not available for this model.")

    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

