import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import numpy as np

def plot_data(X, Y):
    """
    Plot the input data with respect to its features and target.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Data Plot")
    plt.grid(True)
    fig = plt.gcf()
    return fig

def plot_regression(X, Y, model):
    """
    Plot regression model's fit line along with data points.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, label="Data", color="blue")
    plt.plot(X, model.predict(X), label="Fit", color="red")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Regression Plot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig = plt.gcf()
    return fig

def plot_confusion_matrix(model, X, Y):
    """
    Plot confusion matrix for classification models using ConfusionMatrixDisplay.
    """
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_estimator(model, X, Y, cmap='Blues', xticks_rotation='vertical')
    disp.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    fig = plt.gcf()
    return fig


def plot_confusion_from_predictions(y_true, y_pred, labels=None):
    """
    Plot confusion matrix from true and predicted labels.
    """
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", xticks_rotation="vertical", labels=labels)
    disp.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    fig = plt.gcf()
    return fig


def plot_tree_diagram(model):
    """
    Plot Random Forest or Decision Tree diagram for visualization.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(model.estimators_[0], filled=True)
    plt.title("Random Forest Tree Diagram")
    plt.tight_layout()
    fig = plt.gcf()
    return fig

def plot_curve_fit(X, Y, params, fit_type):
    """
    Plot Gaussian or Exponential curve fitting.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, label="Data", color="blue")

    if fit_type == 'gaussian':
        def gaussian(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
        plt.plot(X, gaussian(X, *params), label=f"Gaussian Fit\nmean={params[1]:.3f}, sigma={params[2]:.3f}", color="red")
    elif fit_type == 'exponential':
        def exponential(x, a, b, c):
            return a * np.exp(b * x) + c
        plt.plot(X, exponential(X, *params), label=f"Exponential Fit\nrate={params[1]:.3f}", color="green")

    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title(f"{fit_type.capitalize()} Curve Fitting Plot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig = plt.gcf()
    return fig


def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """
    Plot residuals versus predicted values.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, color="purple")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    fig = plt.gcf()
    return fig


def plot_feature_importance(importances, feature_names):
    """
    Plot feature importances as a sorted bar chart.
    """
    importances = np.asarray(importances)
    order = np.argsort(importances)[::-1]
    sorted_importances = importances[order]
    sorted_features = np.asarray(feature_names)[order]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances, tick_label=sorted_features)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Feature Importances")
    plt.tight_layout()
    fig = plt.gcf()
    return fig


def plot_forecast(time_index, y_train, y_valid, y_pred_valid, y_forecast):
    """Plot train/validation fit and forward forecast."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(y_train):
        ax.plot(y_train.index, y_train, label="Train", color="#1f77b4")
    if len(y_valid):
        ax.plot(y_valid.index, y_valid, label="Validation", color="#ff7f0e")
        ax.plot(y_valid.index, y_pred_valid, label="Predicted (val)", linestyle="--", color="#2ca02c")
    if len(y_forecast):
        horizon_index = range(len(y_train) + len(y_valid), len(y_train) + len(y_valid) + len(y_forecast))
        ax.plot(horizon_index, y_forecast, label="Forecast", linestyle=":", color="#d62728")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_local_contributions(feature_names, contributions, predicted, baseline=None):
    """Plot local explanation contributions."""
    contribs = np.asarray(contributions)
    order = np.argsort(np.abs(contribs))[::-1]
    feats = np.asarray(feature_names)[order]
    contribs = contribs[order]
    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4" if c >= 0 else "#d62728" for c in contribs]
    plt.barh(range(len(contribs)), contribs, color=colors)
    plt.yticks(range(len(contribs)), feats)
    plt.axvline(0, color="black", linewidth=1)
    title = f"Prediction: {predicted:.3f}"
    if baseline is not None:
        title += f" (baseline {baseline:.3f})"
    plt.title(title)
    plt.tight_layout()
    fig = plt.gcf()
    return fig
