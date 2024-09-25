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
    plt.show()

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
    plt.show()

def plot_confusion_matrix(model, X, Y):
    """
    Plot confusion matrix for classification models using ConfusionMatrixDisplay.
    """
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_estimator(model, X, Y, cmap='Blues', xticks_rotation='vertical')
    disp.ax_.set_title("Confusion Matrix")
    plt.show()


def plot_tree_diagram(model):
    """
    Plot Random Forest or Decision Tree diagram for visualization.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(model.estimators_[0], filled=True)
    plt.title("Random Forest Tree Diagram")
    plt.show()

def plot_curve_fit(X, Y, params, fit_type):
    """
    Plot Gaussian or Exponential curve fitting.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, label="Data", color="blue")

    if fit_type == 'gaussian':
        def gaussian(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
        plt.plot(X, gaussian(X, *params), label="Gaussian Fit", color="red")
    elif fit_type == 'exponential':
        def exponential(x, a, b, c):
            return a * np.exp(b * x) + c
        plt.plot(X, exponential(X, *params), label="Exponential Fit", color="green")

    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title(f"{fit_type.capitalize()} Curve Fitting Plot")
    plt.legend()
    plt.grid(True)
    plt.show()

