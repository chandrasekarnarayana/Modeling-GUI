import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox, QComboBox, QLabel, QTextEdit, QListWidget, QTableWidget, QTableWidgetItem, QAbstractItemView
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
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Add Load CSV button
        self.load_button = QPushButton("Load CSV")
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)

        # Add model selection combo box
        self.model_combo = QComboBox()
        self.model_combo.addItems(["OLS", "Rolling Least Squares", "Random Forest", "Gradient Boosting", "KMeans Clustering", "Gaussian Fitting", "Exponential Fitting"])
        layout.addWidget(self.model_combo)

        # Add multi-selection widget for selecting X columns
        self.x_list_widget = QListWidget()
        self.x_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(QLabel("Select X Columns:"))
        layout.addWidget(self.x_list_widget)

        # Add combo box for selecting Y column
        self.y_combo = QComboBox()
        layout.addWidget(QLabel("Select Y Column:"))
        layout.addWidget(self.y_combo)

        # Add a table to display the loaded CSV file
        self.csv_preview_table = QTableWidget()
        layout.addWidget(QLabel("CSV Preview:"))
        layout.addWidget(self.csv_preview_table)

        # Add Run Model button
        self.run_button = QPushButton("Run Model")
        self.run_button.clicked.connect(self.run_model)
        layout.addWidget(self.run_button)

        # Add result display box
        self.result_label = QLabel("Results:")
        layout.addWidget(self.result_label)
        self.result_box = QTextEdit()
        layout.addWidget(self.result_box)

        central_widget.setLayout(layout)


    def load_csv(self):
    	"""
    	Load a CSV file and populate the X and Y selection widgets.
    	"""
    	# Open file dialog to load CSV
    	file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
    	if not file_path:
            return

    	try:
            # Load CSV into a pandas DataFrame
            self.data = pd.read_csv(file_path)
        
            # Populate the X (features) list widget and Y (target) combo box
            self.x_list_widget.clear()
            self.y_combo.clear()
        
            # Add all column names to both widgets
            self.x_list_widget.addItems(self.data.columns)
            self.y_combo.addItems(self.data.columns)
        
            # Display the CSV preview in the table widget
            self.csv_preview_table.setRowCount(min(10, len(self.data)))  # Show only first 10 rows
            self.csv_preview_table.setColumnCount(len(self.data.columns))
            self.csv_preview_table.setHorizontalHeaderLabels(self.data.columns)

            for i in range(min(10, len(self.data))):  # Display up to 10 rows
                for j, column in enumerate(self.data.columns):
                    self.csv_preview_table.setItem(i, j, QTableWidgetItem(str(self.data.iloc[i, j])))

    	except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")

    def run_model(self):
        """
        Run the selected model based on user input.
        """
        # Get selected X columns (features)
        selected_x_items = self.x_list_widget.selectedItems()
        x_columns = [item.text() for item in selected_x_items]
    
        # Get selected Y column (target)
        y_column = self.y_combo.currentText()
    
        if not x_columns or not y_column:
            QMessageBox.warning(self, "Selection Error", "Please select at least one X column and one Y column.")
            return

        X = self.data[x_columns]  # Extract X as a DataFrame
        Y = self.data[y_column]   # Extract Y as a Series

        # Get selected model from the dropdown
        model_choice = self.model_combo.currentText()
        if model_choice == "OLS":
            self.run_ols(X, Y)
        elif model_choice == "Rolling Least Squares":
            self.run_rolling_ls(X, Y)
        elif model_choice == "Random Forest":
            self.run_random_forest(X, Y)
        elif model_choice == "Gradient Boosting":
            self.run_gradient_boost(X, Y)
        elif model_choice == "KMeans Clustering":
            self.run_kmeans(X)
        elif model_choice == "Gaussian Fitting":
            self.run_gaussian_fitting(X, Y)
        elif model_choice == "Exponential Fitting":
            self.run_exponential_fitting(X, Y)

    def run_ols(self, X, Y):
        """
        Run OLS model on the data.
        """
        try:
            model = self.model_manager.ols(X, Y)
            self.result_box.setPlainText(self.model_manager.get_summary().as_text())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run OLS: {str(e)}")

    def run_rolling_ls(self, X, Y):
        """
        Run Rolling Least Squares model on the data.
        """
        try:
            window = 5  # Adjust window as per user input
            rolling_model = self.model_manager.rolling_ls(X, Y, window=window)
            # Display rolling model parameters (code omitted for brevity)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run Rolling LS: {str(e)}")

    def run_random_forest(self, X, Y):
        """
        Run Random Forest model on the data.
        """
        dialog = RandomForestDialog(self)
        if dialog.exec_() == dialog.Accepted:
            n_estimators = dialog.n_estimators
            max_depth = dialog.max_depth

            try:
                model = self.model_manager.random_forest(X, Y, n_estimators=n_estimators, max_depth=max_depth)
                plot_tree_diagram(model)  # Display Random Forest tree diagram
                self.result_box.setPlainText("Random Forest model trained successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to run Random Forest: {str(e)}")

    def run_gradient_boost(self, X, Y):
        """
        Run Gradient Boosting model on the data.
        """
        dialog = GradientBoostDialog(self)
        if dialog.exec_() == dialog.Accepted:
            n_estimators = dialog.n_estimators
            learning_rate = dialog.learning_rate
            max_depth = dialog.max_depth

            try:
                model = self.model_manager.gradient_boost(X, Y, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
                plot_confusion_matrix(model, X, Y)  # Display confusion matrix
                self.result_box.setPlainText("Gradient Boost model trained successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to run Gradient Boost: {str(e)}")

    def run_kmeans(self, X):
        """
        Run KMeans Clustering on the data.
        """
        dialog = KMeansDialog(self)
        if dialog.exec_() == dialog.Accepted:
            n_clusters = dialog.n_clusters

            try:
                model = self.model_manager.kmeans_clustering(X, n_clusters=n_clusters)
                # Display clustering result (code omitted for brevity)
                self.result_box.setPlainText(f"KMeans Clustering with {n_clusters} clusters completed.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to run KMeans Clustering: {str(e)}")

    def run_gaussian_fitting(self, X, Y):
        """
        Run Gaussian Fitting on the data.
        """
        try:
            params = self.model_manager.gaussian_fitting(X, Y)
            plot_curve_fit(X, Y, params, 'gaussian')  # Display fitted Gaussian curve
            self.result_box.setPlainText(f"Gaussian Fitting completed with parameters: {params}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run Gaussian Fitting: {str(e)}")

    def run_exponential_fitting(self, X, Y):
        """
        Run Exponential Fitting on the data.
        """
        try:
            params = self.model_manager.exponential_fitting(X, Y)
            plot_curve_fit(X, Y, params, 'exponential')  # Display fitted Exponential curve
            self.result_box.setPlainText(f"Exponential Fitting completed with parameters: {params}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run Exponential Fitting: {str(e)}")


def main():
    """
    Entry point for running the GUI application.
    """
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

