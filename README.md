# Modeling-GUI: A PyQt5-Based GUI for Statistical and Machine Learning Models

**Modeling-GUI** is a Python-based graphical user interface (GUI) that allows users to apply and visualize a variety of statistical and machine learning models without needing to write code. The package integrates models such as Ordinary Least Squares (OLS), Random Forest, Gradient Boosting, KMeans Clustering, Gaussian fitting, and more.

This tool is built using **PyQt5** for the GUI and **matplotlib**, **seaborn**, and other scientific libraries for visualizations. It’s designed for data scientists, analysts, and machine learning enthusiasts who want to quickly prototype models and see immediate results.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Supported Models](#supported-models)
  - [Example Workflow](#example-workflow)
  - [Model Customization](#model-customization)
  - [Visualizations](#visualizations)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [Version](#version)
- [License](#license)

---

## Features

**Modeling-GUI** provides a wide range of models and tools:
- **Statistical Regression Models**:
  - Ordinary Least Squares (OLS)
  - Weighted Least Squares (WLS)
  - Generalized Least Squares (GLS)
  - Recursive Least Squares (Recursive LS)
  - Rolling Least Squares (Rolling LS)
  - Robust Linear Model (RLM)
  
- **Machine Learning Models**:
  - Random Forest (Regression and Classification)
  - Gradient Boosting (Regression and Classification)

- **Clustering**:
  - KMeans Clustering (with customizable number of clusters)

- **Advanced Data Fitting**:
  - Gaussian Fitting
  - Exponential Growth/Decay Fitting

- **Visualizations**:
  - Regression plots
  - Confusion matrices for classification models
  - Random Forest tree diagrams
  - Gaussian and Exponential curve fitting plots

- **Parameter Customization**:
  - Users can specify model parameters such as the number of estimators for Random Forest, learning rate for Gradient Boosting, or the number of clusters for KMeans Clustering through customizable dialogs.

---

## Installation

### Prerequisites

To use **Modeling-GUI**, you need Python 3.6 or higher. Ensure that the required libraries are installed, or let the package manager install them for you.

### Steps
**To install using Pypi package**
   ```bash
pip install modeling-gui
```

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your_username/modeling_gui.git
   cd modeling_gui
   ```

2. **Install dependencies**:

   You can install all the required dependencies listed in `requirements.txt`:
   (The package requires python 3.9)

   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**:

   Run the following command to install the package locally:

   ```bash
   pip install .
   ```

5. **Running the GUI**:

   After installation, you can launch the GUI by running the following command:

   ```bash
   run_modeling_gui
   ```

   Alternatively, if you want to run it directly from the cloned directory:

   ```bash
   python main.py
   ```

---

## Usage

**Modeling-GUI** provides an interactive way to load data, select models, and visualize the results. Here's how to use the application:

### Supported Models

You can choose from a wide array of models, each with specific use cases:

- **OLS (Ordinary Least Squares)**: Standard linear regression model.
- **WLS (Weighted Least Squares)**: Linear regression with weighted observations.
- **GLS (Generalized Least Squares)**: A flexible linear model that accounts for heteroscedasticity.
- **Recursive LS (Recursive Least Squares)**: Online linear regression useful for time series data.
- **Rolling LS (Rolling Least Squares)**: Perform a rolling regression with a moving window.
- **RLM (Robust Linear Model)**: Linear regression that is less sensitive to outliers.
- **Random Forest**: Both classification and regression using an ensemble of decision trees.
- **Gradient Boosting**: Powerful technique for both classification and regression, focusing on reducing prediction errors.
- **KMeans Clustering**: Clustering algorithm for unsupervised learning, with customizable cluster numbers.
- **Gaussian Fitting**: Fits a Gaussian (normal distribution) curve to the data.
- **Exponential Growth/Decay Fitting**: Models exponential growth or decay patterns.

### Example Workflow

1. **Loading Data**:
   - Click the **"Load CSV"** button to load a dataset in CSV format. The dataset should contain both feature columns (X) and a target column (Y).

2. **Selecting Features**:
   - After loading the CSV, select the appropriate feature (X) and target (Y) columns from the dropdown menus.

3. **Choosing a Model**:
   - From the **"Model"** dropdown, select the model you want to run (e.g., OLS, Random Forest, KMeans Clustering).

4. **Running the Model**:
   - Click **"Run Model"** to execute the selected model and visualize the results. For customizable models (Random Forest, Gradient Boosting, KMeans), a dialog box will appear allowing you to set parameters like `n_estimators` or `max_depth`.

5. **Viewing Results**:
   - The results of the model will be displayed in the output box, and visualizations (e.g., regression lines, confusion matrices, fitted curves) will be shown in separate windows.

### Model Customization

For some models, you can customize parameters via dialog boxes. For example:

- **Random Forest**: Customize the number of trees (`n_estimators`) and tree depth (`max_depth`).
- **Gradient Boosting**: Customize the number of boosting rounds (`n_estimators`) and learning rate.
- **KMeans Clustering**: Customize the number of clusters (`n_clusters`).

### Visualizations

**Modeling-GUI** includes a range of high-quality visualizations to help you understand your models:

- **Regression Models**: Scatter plots with regression lines (e.g., for OLS, WLS, GLS, RLM).
- **Classification Models**: Confusion matrices (e.g., for Random Forest classification).
- **Decision Trees**: Visual representations of decision trees in Random Forest models.
- **Gaussian Fitting**: Gaussian (normal distribution) curve fitting plots.
- **Exponential Growth/Decay Fitting**: Plots of exponential growth or decay functions.

---

## Dependencies

The following libraries are required to run **Modeling-GUI**:

- **pandas**: For data manipulation and loading CSV files.
- **matplotlib**: For creating visualizations.
- **seaborn**: For enhanced visualizations (e.g., heatmaps for confusion matrices).
- **statsmodels**: For statistical models like OLS, WLS, GLS, RLM.
- **scikit-learn**: For machine learning models like Random Forest, Gradient Boosting, and KMeans Clustering.
- **pyqt5**: For the graphical user interface.
- **scipy**: For fitting Gaussian and exponential curves.
- **graphviz**: For visualizing Random Forest decision trees.

You can install all these dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome! If you would like to contribute to **Modeling-GUI**, please follow these steps:

1. **Fork the repository** on GitHub.
2. **Create a new feature branch** (`git checkout -b feature/new-feature`).
3. **Make your changes** and commit them (`git commit -m "Add new feature"`).
4. **Push to the branch** (`git push origin feature/new-feature`).
5. **Open a Pull Request** explaining your changes.

Please ensure that your contributions are well-documented and covered by tests where applicable.

---

## Version

**Current Version**: v0.1.0

This is the initial version of **Modeling-GUI**, providing core functionality for regression, classification, clustering, and data fitting models. 
Future updates will include additional models, enhanced visualizations, and further customization options.
Currently, The main focus is establishing a working framework with bug fixes.

---

## Author

Developed by **Chandrasekar SUBRAMANI NARAYANA**.

Feel free to contact me for any questions or suggestions at [chandrasekarnarayana@gmail.com](mailto:chandrasekarnarayana@gmail.com).

---

## License

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.

