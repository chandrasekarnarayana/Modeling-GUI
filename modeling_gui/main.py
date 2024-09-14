import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon
from modeling_gui.models import ModelManager
from modeling_gui.visualization import plot_data, plot_confusion_matrix, plot_tree_diagram, plot_curve_fit
from modeling_gui.dialogs import RandomForestDialog, GradientBoostDialog, KMeansDialog

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modeling GUI")
        self.setWindowIcon(QIcon("icon.png"))
        self.data = None
        self.model_manager = ModelManager()

        # Setup UI elements
        self.setup_ui()

    def setup_ui(self):
        """
        Initialize the UI components.
        """
        # Add buttons, text boxes, etc. (Code omitted for brevity)
        pass

    def load_csv(self):
        """
        Load data from a CSV file and display it.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            self.data = pd.read_csv(file_path)
            # Display data in the UI (e.g., in a table view)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")
            return

        if self.data is None:
            QMessageBox.critical(self, "Error", "Failed to load CSV file.")
            return

    def run_model(self):
        """
        Run the selected model based on user input.
        """
        model_choice = self.model_combo.currentText()
        if model_choice == "OLS":
            self.run_ols()
        elif model_choice == "Rolling Least Squares":
            self.run_rolling_ls()
        elif model_choice == "Random Forest":
            self.run_random_forest()
        elif model_choice == "Gradient Boosting":
            self.run_gradient_boost()
        elif model_choice == "KMeans Clustering":
            self.run_kmeans()
        elif model_choice == "Gaussian Fitting":
            self.run_gaussian_fitting()
        elif model_choice == "Exponential Fitting":
            self.run_exponential_fitting()

    def run_ols(self):
        """
        Run OLS model on the data.
        """
        try:
            X = self.data[['X']]  # Replace with appropriate columns
            Y = self.data['Y']  # Replace with appropriate column
            model = self.model_manager.ols(X, Y)
            self.result_box.setPlainText(self.model_manager.get_summary().as_text())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run OLS: {str(e)}")

    def run_rolling_ls(self):
        """
        Run Rolling Least Squares model on the data.
        """
        try:
            X = self.data[['X']]
            Y = self.data['Y']
            window = 5  # Adjust window as per user input
            rolling_model = self.model_manager.rolling_ls(X, Y, window=window)
            # Display rolling model parameters (code omitted for brevity)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run Rolling LS: {str(e)}")

    def run_random_forest(self):
        """
        Run Random Forest model on the data.
        """
        dialog = RandomForestDialog(self)
        if dialog.exec_() == dialog.Accepted:
            n_estimators = dialog.n_estimators
            max_depth = dialog.max_depth

            try:
                X = self.data[['X']]
                Y = self.data['Y']
                model = self.model_manager.random_forest(X, Y, n_estimators=n_estimators, max_depth=max_depth)
                plot_tree_diagram(model)  # Display Random Forest tree diagram
                self.result_box.setPlainText("Random Forest model trained successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to run Random Forest: {str(e)}")

    def run_gradient_boost(self):
        """
        Run Gradient Boosting model on the data.
        """
        dialog = GradientBoostDialog(self)
        if dialog.exec_() == dialog.Accepted:
            n_estimators = dialog.n_estimators
            learning_rate = dialog.learning_rate
            max_depth = dialog.max_depth

            try:
                X = self.data[['X']]
                Y = self.data['Y']
                model = self.model_manager.gradient_boost(X, Y, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
                plot_confusion_matrix(model, X, Y)  # Display confusion matrix
                self.result_box.setPlainText("Gradient Boost model trained successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to run Gradient Boost: {str(e)}")

    def run_kmeans(self):
        """
        Run KMeans Clustering on the data.
        """
        dialog = KMeansDialog(self)
        if dialog.exec_() == dialog.Accepted:
            n_clusters = dialog.n_clusters

            try:
                X = self.data[['X']]
                model = self.model_manager.kmeans_clustering(X, n_clusters=n_clusters)
                # Display clustering result (code omitted for brevity)
                self.result_box.setPlainText(f"KMeans Clustering with {n_clusters} clusters completed.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to run KMeans Clustering: {str(e)}")

    def run_gaussian_fitting(self):
        """
        Run Gaussian Fitting on the data.
        """
        try:
            X = self.data['X']
            Y = self.data['Y']
            params = self.model_manager.gaussian_fitting(X, Y)
            plot_curve_fit(X, Y, params, 'gaussian')  # Display fitted Gaussian curve
            self.result_box.setPlainText(f"Gaussian Fitting completed with parameters: {params}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run Gaussian Fitting: {str(e)}")

    def run_exponential_fitting(self):
        """
        Run Exponential Fitting on the data.
        """
        try:
            X = self.data['X']
            Y = self.data['Y']
            params = self.model_manager.exponential_fitting(X, Y)
            plot_curve_fit(X, Y, params, 'exponential')  # Display fitted Exponential curve
            self.result_box.setPlainText(f"Exponential Fitting completed with parameters: {params}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run Exponential Fitting: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

