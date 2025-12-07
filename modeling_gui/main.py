import sys
import logging
import importlib.resources as pkg_resources
import os
import platform
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import json
import webbrowser

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
    QInputDialog,
    QShortcut,
    QDialog,
    QTableWidget,
    QHeaderView,
)

from modeling_gui import __version__
from modeling_gui.dialogs import GradientBoostDialog, KMeansDialog, RandomForestDialog
from modeling_gui.automl import (
    AutoMLDependencyMissing,
    detect_problem_type,
    run_automl,
)
from modeling_gui.coach import CoachManager, CoachState
from modeling_gui.domain import get_domain_config
from modeling_gui.reporting import build_summary_text, export_report
from modeling_gui.reporting import generate_narrative
from modeling_gui.forecasting import (
    detect_time_column,
    ForecastConfig,
    train_forecast_model,
)
from modeling_gui.preprocessing import (
    infer_column_types,
    parse_date_columns,
    encode_categoricals,
    generate_date_features,
)
from modeling_gui.data_quality import (
    validate_schema,
    detect_missing_issues,
    detect_type_mismatches,
    detect_potential_leakage,
    detect_imbalance,
)
from modeling_gui.drift import compute_snapshot, compare_snapshot
from modeling_gui.metrics import regression_metrics, classification_metrics, choose_primary_metrics
from modeling_gui.logging_view import MemoryLogHandler, LogDialog
from modeling_gui.explain import explain_global, explain_local, compute_partial_dependence
from modeling_gui.troubleshooting import analyze_failure_context
from modeling_gui.main_comparison import comparison_text
from modeling_gui.bundles import export_bundle
from modeling_gui.notebook_export import export_notebook
from modeling_gui.tuning import tune_model
from modeling_gui.model_cards import ModelCard, export_model_card
from modeling_gui.models import ModelManager
from modeling_gui.visualization import (
    plot_confusion_matrix,
    plot_confusion_from_predictions,
    plot_curve_fit,
    plot_feature_importance,
    plot_residuals,
    plot_forecast,
    plot_tree_diagram,
    plot_local_contributions,
)
from modeling_gui.utils import (
    apply_missing_strategy,
    compute_classification_metrics,
    compute_regression_metrics,
    is_regression_target,
    load_csv,
    load_model_bundle,
    save_model_bundle,
    select_numeric_columns,
    split_data,
    standardize_features,
    save_project,
    load_project,
    compute_file_hash,
    filter_run_history,
)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Memory log handler for in-app viewing
memory_handler = MemoryLogHandler()
logger.addHandler(memory_handler)


def _resource_path(*parts: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "resources", *parts)


def _config_path() -> Path:
    """Return the path to the user config file for first-run hints."""
    return Path.home() / ".config" / "modeling_gui" / "config.json"


def _load_user_config() -> dict:
    cfg_path = _config_path()
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning("Could not read config file; using defaults.")
    return {}


def _save_user_config(cfg: dict) -> None:
    cfg_path = _config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        logger.warning("Failed to save user config.")


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modeling GUI")
        try:
            system = platform.system()
            if system == "Windows":
                icon_path = _resource_path("icons", "logo.ico")
            elif system == "Darwin":
                icon_path = _resource_path("icons", "logo.icns")
            else:
                icon_path = _resource_path("icons", "logo.png")
            self.setWindowIcon(QIcon(icon_path))
        except Exception:
            self.setWindowIcon(QIcon())

        self.data = None
        self.model_manager = ModelManager()
        self.last_scaler = None
        self.last_metadata = None
        self.last_feature_importances = None
        self.last_feature_names = []
        self.loaded_metadata = None
        self.basic_mode = True
        self.last_summary_data = None
        self.last_metrics = {}
        self.last_data_path = None
        self.last_fig = None
        self.run_history = []
        self.last_automl_result = None
        self.training_snapshot = None
        self.model_cards = []
        self.last_global_explanation = None
        self.last_local_explanation = None
        self.last_training_features = None
        self.last_training_target = None
        self.drift_issues = []

        self.coach = CoachManager()
        self.domain = get_domain_config("Generic")

        self.setup_ui()
        self.statusBar().showMessage("Ready")
        self.coach_bar.setText(self.coach.update(CoachState.DATA_UNLOADED))
        self._maybe_show_first_run_hint()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Menus
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("Help")
        shortcuts_action = help_menu.addAction("Keyboard Shortcuts…")
        shortcuts_action.triggered.connect(self.show_shortcuts_dialog)
        docs_action = help_menu.addAction("Open Documentation (F1)")
        docs_action.triggered.connect(self.open_docs)
        auto_demo_action = help_menu.addAction("Run Auto Demo")
        auto_demo_action.setToolTip("Automatically load the demo dataset, run Smart Analyze, and cycle key tabs.")
        auto_demo_action.triggered.connect(self.run_auto_demo)

        # Mode toggle
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Basic", "Expert"])
        self.mode_combo.currentTextChanged.connect(self.toggle_mode)
        top_layout.addWidget(self.mode_combo)

        top_layout.addWidget(QLabel("Domain:"))
        self.domain_combo = QComboBox()
        self.domain_combo.addItems(["Generic", "Finance", "Science", "Business"])
        self.domain_combo.setToolTip("Choose the domain closest to your data to tailor hints and metrics.")
        self.domain_combo.currentTextChanged.connect(self.change_domain)
        top_layout.addWidget(self.domain_combo)

        top_layout.addStretch()
        layout.addLayout(top_layout)
        # Shortcuts
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.load_csv)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_project_dialog)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, activated=self.save_model)
        QShortcut(QKeySequence("F5"), self, activated=self.smart_analyze)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.run_model)
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)
        QShortcut(QKeySequence("F1"), self, activated=self.open_docs)

        load_layout = QHBoxLayout()
        self.load_button = QPushButton("Load CSV")
        self.load_button.setToolTip("Load a CSV file from disk.")
        self.load_button.clicked.connect(self.load_csv)
        load_layout.addWidget(self.load_button)

        self.load_demo_button = QPushButton("Load Demo Dataset")
        self.load_demo_button.setToolTip("Load a bundled demo dataset and pre-select defaults.")
        self.load_demo_button.clicked.connect(self.load_demo_dataset)
        load_layout.addWidget(self.load_demo_button)
        layout.addLayout(load_layout)

        layout.addWidget(QLabel("What do you want to predict? (Target column)"))
        self.y_combo = QComboBox()
        self.y_combo.setToolTip("Select the target column (ignored for clustering).")
        layout.addWidget(self.y_combo)

        layout.addWidget(QLabel("Select X Columns (features, optional):"))
        self.x_list_widget = QListWidget()
        self.x_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.x_list_widget.setToolTip("Choose feature columns (numeric columns recommended).")
        layout.addWidget(self.x_list_widget)

        layout.addWidget(QLabel("CSV Preview:"))
        self.csv_preview_table = QTableWidget()
        layout.addWidget(self.csv_preview_table)

        smart_layout = QHBoxLayout()
        self.smart_button = QPushButton("Smart Analyze")
        self.smart_button.setToolTip("Let AutoML pick the best approach and show metrics/plots.")
        self.smart_button.clicked.connect(self.smart_analyze)
        self.smart_button.setEnabled(False)
        smart_layout.addWidget(self.smart_button)
        smart_layout.addStretch()
        layout.addLayout(smart_layout)

        # Advanced (Expert) controls container
        self.advanced_container = QWidget()
        advanced_layout = QVBoxLayout()

        advanced_layout.addWidget(QLabel("Model (Expert mode):"))
        self.model_combo = QComboBox()
        models = [
            "OLS",
            "Rolling Least Squares",
            "Random Forest",
            "Gradient Boosting",
            "KMeans Clustering",
            "Gaussian Fitting",
            "Exponential Fitting",
        ]
        model_tips = {
            "OLS": "Ordinary Least Squares regression",
            "Rolling Least Squares": "Rolling window regression for time series",
            "Random Forest": "Tree-based ensemble for regression/classification",
            "Gradient Boosting": "Boosted trees for regression/classification",
            "KMeans Clustering": "Unsupervised clustering",
            "Gaussian Fitting": "Curve fit to a Gaussian shape",
            "Exponential Fitting": "Curve fit to an exponential shape",
        }
        self.model_combo.addItems(models)
        for idx, name in enumerate(models):
            self.model_combo.setItemData(idx, model_tips[name], Qt.ToolTipRole)
        advanced_layout.addWidget(self.model_combo)

        advanced_layout.addWidget(QLabel("Preprocessing:"))
        preprocess_layout = QHBoxLayout()
        self.standardize_checkbox = QCheckBox("Standardize numeric features")
        self.standardize_checkbox.setToolTip("Scale numeric features to zero mean and unit variance.")
        preprocess_layout.addWidget(self.standardize_checkbox)

        self.missing_strategy_combo = QComboBox()
        self.missing_strategy_combo.addItem("Keep as-is", userData="none")
        self.missing_strategy_combo.addItem("Drop rows with missing", userData="drop")
        self.missing_strategy_combo.addItem("Mean impute (numeric)", userData="mean")
        self.missing_strategy_combo.setToolTip("Choose how to handle missing values before training.")
        preprocess_layout.addWidget(QLabel("Missing values:"))
        preprocess_layout.addWidget(self.missing_strategy_combo)
        advanced_layout.addLayout(preprocess_layout)

        advanced_layout.addWidget(QLabel("Train/Test Split:"))
        split_layout = QHBoxLayout()
        self.train_test_checkbox = QCheckBox("Use train/test split")
        self.train_test_checkbox.setToolTip("Enable to train on a subset and evaluate on held-out data.")
        split_layout.addWidget(self.train_test_checkbox)

        split_layout.addWidget(QLabel("Test size"))
        self.test_size_input = QDoubleSpinBox()
        self.test_size_input.setRange(0.05, 0.9)
        self.test_size_input.setSingleStep(0.05)
        self.test_size_input.setValue(0.2)
        split_layout.addWidget(self.test_size_input)

        split_layout.addWidget(QLabel("Random state"))
        self.random_state_input = QSpinBox()
        self.random_state_input.setRange(0, 10_000)
        self.random_state_input.setValue(42)
        split_layout.addWidget(self.random_state_input)
        advanced_layout.addLayout(split_layout)

        # Forecasting controls
        forecast_layout = QHBoxLayout()
        self.forecast_time_combo = QComboBox()
        self.forecast_time_combo.setToolTip("Select the time column for forecasting.")
        self.forecast_model_combo = QComboBox()
        self.forecast_model_combo.addItems(["naive", "ets", "arima", "prophet"])
        self.forecast_model_combo.setToolTip("Choose a forecasting model. Prophet is included by default.")
        self.forecast_horizon_spin = QSpinBox()
        self.forecast_horizon_spin.setRange(1, 365)
        self.forecast_horizon_spin.setValue(12)
        forecast_layout.addWidget(QLabel("Time column"))
        forecast_layout.addWidget(self.forecast_time_combo)
        forecast_layout.addWidget(QLabel("Forecast model"))
        forecast_layout.addWidget(self.forecast_model_combo)
        forecast_layout.addWidget(QLabel("Horizon"))
        forecast_layout.addWidget(self.forecast_horizon_spin)
        self.forecast_backtest_checkbox = QCheckBox("Run backtest")
        self.forecast_backtest_checkbox.setToolTip("Enable simple rolling-origin backtesting (averaged metrics).")
        forecast_layout.addWidget(self.forecast_backtest_checkbox)
        self.forecast_button = QPushButton("Run forecast")
        self.forecast_button.clicked.connect(self.run_forecast)
        forecast_layout.addWidget(self.forecast_button)
        advanced_layout.addLayout(forecast_layout)

        forecast_opts_layout = QHBoxLayout()
        self.forecast_growth_combo = QComboBox()
        self.forecast_growth_combo.addItems(["linear", "logistic"])
        self.forecast_growth_combo.setToolTip("Prophet growth: linear or logistic.")
        self.forecast_yearly_checkbox = QCheckBox("Yearly seasonality")
        self.forecast_yearly_checkbox.setChecked(True)
        self.forecast_yearly_spin = QSpinBox()
        self.forecast_yearly_spin.setRange(0, 50)
        self.forecast_yearly_spin.setValue(0)
        self.forecast_yearly_spin.setToolTip("Optional override for yearly seasonality (0 means auto/bool).")
        self.forecast_weekly_checkbox = QCheckBox("Weekly seasonality")
        self.forecast_weekly_checkbox.setChecked(True)
        self.forecast_weekly_spin = QSpinBox()
        self.forecast_weekly_spin.setRange(0, 50)
        self.forecast_weekly_spin.setValue(0)
        self.forecast_weekly_spin.setToolTip("Optional override for weekly seasonality (0 means auto/bool).")
        self.forecast_daily_checkbox = QCheckBox("Daily seasonality")
        self.forecast_daily_checkbox.setChecked(False)
        self.forecast_daily_spin = QSpinBox()
        self.forecast_daily_spin.setRange(0, 50)
        self.forecast_daily_spin.setValue(0)
        self.forecast_daily_spin.setToolTip("Optional override for daily seasonality (0 means auto/bool).")
        self.forecast_holidays_checkbox = QCheckBox("Use country holidays")
        self.forecast_holidays_combo = QComboBox()
        self.forecast_holidays_combo.addItems(["US", "UK", "DE", "FR", "IN"])
        self.forecast_holidays_combo.setEnabled(False)
        self.forecast_holidays_checkbox.stateChanged.connect(
            lambda state: self.forecast_holidays_combo.setEnabled(bool(state))
        )
        forecast_opts_layout.addWidget(QLabel("Growth"))
        forecast_opts_layout.addWidget(self.forecast_growth_combo)
        forecast_opts_layout.addWidget(self.forecast_yearly_checkbox)
        forecast_opts_layout.addWidget(self.forecast_yearly_spin)
        forecast_opts_layout.addWidget(self.forecast_weekly_checkbox)
        forecast_opts_layout.addWidget(self.forecast_weekly_spin)
        forecast_opts_layout.addWidget(self.forecast_daily_checkbox)
        forecast_opts_layout.addWidget(self.forecast_daily_spin)
        forecast_opts_layout.addWidget(self.forecast_holidays_checkbox)
        forecast_opts_layout.addWidget(self.forecast_holidays_combo)
        forecast_opts_layout.addStretch()
        advanced_layout.addLayout(forecast_opts_layout)

        action_layout = QHBoxLayout()
        self.run_button = QPushButton("Run selected model")
        self.run_button.clicked.connect(self.run_model)
        action_layout.addWidget(self.run_button)

        self.tune_button = QPushButton("Tune…")
        self.tune_button.setToolTip("Guided hyperparameter tuning for supported models (RF/GB).")
        self.tune_button.clicked.connect(self.tune_current_model)
        action_layout.addWidget(self.tune_button)

        self.predict_button = QPushButton("Predict with Loaded Model")
        self.predict_button.setEnabled(False)
        self.predict_button.setToolTip("Run predictions using a previously loaded model bundle.")
        self.predict_button.clicked.connect(self.predict_with_loaded_model)
        action_layout.addWidget(self.predict_button)
        advanced_layout.addLayout(action_layout)

        save_layout = QHBoxLayout()
        self.save_model_button = QPushButton("Save trained model…")
        self.save_model_button.clicked.connect(self.save_model)
        save_layout.addWidget(self.save_model_button)

        self.load_model_button = QPushButton("Load model…")
        self.load_model_button.clicked.connect(self.load_model)
        save_layout.addWidget(self.load_model_button)

        self.feature_importance_button = QPushButton("Show Feature Importance")
        self.feature_importance_button.setEnabled(False)
        self.feature_importance_button.setToolTip("View feature importances for tree-based models.")
        self.feature_importance_button.clicked.connect(self.show_feature_importance)
        save_layout.addWidget(self.feature_importance_button)
        advanced_layout.addLayout(save_layout)

        self.advanced_container.setLayout(advanced_layout)
        layout.addWidget(self.advanced_container)

        self.problem_label = QLabel("")
        layout.addWidget(self.problem_label)

        layout.addWidget(QLabel("Results:"))
        self.tabs = QTabWidget()
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.details_box = QTextEdit()
        self.details_box.setReadOnly(True)
        self.data_quality_box = QTextEdit()
        self.data_quality_box.setReadOnly(True)
        self.tabs.addTab(self.result_box, "Summary")
        self.tabs.addTab(self.details_box, "What I did")
        self.tabs.addTab(self.data_quality_box, "Data quality")

        # Explainability tab
        self.explain_tab = QWidget()
        explain_layout = QVBoxLayout()
        self.explain_box = QTextEdit()
        self.explain_box.setReadOnly(True)
        explain_layout.addWidget(self.explain_box)
        explain_buttons = QHBoxLayout()
        self.explain_global_btn = QPushButton("Show global importance plot")
        self.explain_global_btn.clicked.connect(self.show_global_explanation_plot)
        explain_buttons.addWidget(self.explain_global_btn)
        self.local_row_spin = QSpinBox()
        self.local_row_spin.setRange(0, 0)
        self.explain_local_btn = QPushButton("Explain selected row")
        self.explain_local_btn.clicked.connect(self.show_local_explanation_plot)
        explain_buttons.addWidget(QLabel("Row index:"))
        explain_buttons.addWidget(self.local_row_spin)
        explain_buttons.addWidget(self.explain_local_btn)
        explain_buttons.addStretch()
        explain_layout.addLayout(explain_buttons)
        self.explain_tab.setLayout(explain_layout)
        self.tabs.addTab(self.explain_tab, "Explain")

        # Model comparison tab
        self.comparison_tab = QWidget()
        comparison_layout = QVBoxLayout()
        self.comparison_box = QTextEdit()
        self.comparison_box.setReadOnly(True)
        comparison_layout.addWidget(self.comparison_box)
        comp_btns = QHBoxLayout()
        self.leaderboard_plot_btn = QPushButton("Show leaderboard chart")
        self.leaderboard_plot_btn.clicked.connect(self.show_leaderboard_plot)
        comp_btns.addWidget(self.leaderboard_plot_btn)
        comp_btns.addStretch()
        comparison_layout.addLayout(comp_btns)
        self.comparison_tab.setLayout(comparison_layout)
        self.tabs.addTab(self.comparison_tab, "Model comparison")

        # Scenario testing tab
        self.scenario_tab = QWidget()
        scenario_layout = QVBoxLayout()
        self.scenario_output = QTextEdit()
        self.scenario_output.setReadOnly(True)
        scenario_controls = QHBoxLayout()
        self.scenario_feature_combo = QComboBox()
        self.scenario_feature_combo.currentTextChanged.connect(self.update_scenario_range)
        scenario_controls.addWidget(QLabel("Feature:"))
        scenario_controls.addWidget(self.scenario_feature_combo)
        self.scenario_value_spin = QDoubleSpinBox()
        self.scenario_value_spin.setRange(-1e9, 1e9)
        self.scenario_value_spin.setDecimals(4)
        scenario_controls.addWidget(QLabel("Value:"))
        scenario_controls.addWidget(self.scenario_value_spin)
        self.scenario_row_spin = QSpinBox()
        self.scenario_row_spin.setRange(0, 0)
        scenario_controls.addWidget(QLabel("Base row:"))
        scenario_controls.addWidget(self.scenario_row_spin)
        self.apply_scenario_btn = QPushButton("Apply scenario")
        self.apply_scenario_btn.clicked.connect(self.apply_scenario_change)
        scenario_controls.addWidget(self.apply_scenario_btn)
        self.partial_plot_btn = QPushButton("Show partial dependence")
        self.partial_plot_btn.clicked.connect(self.show_partial_dependence_plot)
        scenario_controls.addWidget(self.partial_plot_btn)
        scenario_controls.addStretch()
        scenario_layout.addLayout(scenario_controls)
        scenario_layout.addWidget(self.scenario_output)
        self.scenario_tab.setLayout(scenario_layout)
        self.tabs.addTab(self.scenario_tab, "Scenario testing")

        # Data drift tab
        self.drift_tab = QWidget()
        drift_layout = QVBoxLayout()
        self.drift_box = QTextEdit()
        self.drift_box.setReadOnly(True)
        drift_layout.addWidget(self.drift_box)
        self.drift_tab.setLayout(drift_layout)
        self.tabs.addTab(self.drift_tab, "Data Drift")

        # Model cards tab
        self.cards_tab = QWidget()
        cards_layout = QVBoxLayout()
        self.cards_table = QTableWidget(0, 4)
        self.cards_table.setHorizontalHeaderLabels(["Name", "Problem", "Created", "Metrics"])
        cards_layout.addWidget(self.cards_table)
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by problem:"))
        self.run_filter_problem = QComboBox()
        self.run_filter_problem.addItems(["All", "regression", "classification", "forecasting"])
        self.run_filter_problem.currentTextChanged.connect(self.refresh_model_cards_table)
        filter_layout.addWidget(self.run_filter_problem)
        filter_layout.addWidget(QLabel("Model filter:"))
        self.run_filter_model = QComboBox()
        self.run_filter_model.setEditable(True)
        self.run_filter_model.addItems(["All"])
        self.run_filter_model.currentTextChanged.connect(self.refresh_model_cards_table)
        filter_layout.addStretch()
        cards_layout.addLayout(filter_layout)
        self.run_table = QTableWidget(0, 4)
        self.run_table.setHorizontalHeaderLabels(["When", "Model", "Problem", "Primary metric"])
        cards_layout.addWidget(self.run_table)
        cards_btns = QHBoxLayout()
        self.card_activate_btn = QPushButton("Set as active model")
        self.card_activate_btn.clicked.connect(self.activate_selected_card)
        cards_btns.addWidget(self.card_activate_btn)
        self.card_notes_btn = QPushButton("Edit notes")
        self.card_notes_btn.clicked.connect(self.edit_card_notes)
        cards_btns.addWidget(self.card_notes_btn)
        self.card_export_btn = QPushButton("Export model card…")
        self.card_export_btn.clicked.connect(self.export_selected_card)
        cards_btns.addWidget(self.card_export_btn)
        cards_btns.addStretch()
        cards_layout.addLayout(cards_btns)
        self.cards_tab.setLayout(cards_layout)
        self.tabs.addTab(self.cards_tab, "Model cards")

        layout.addWidget(self.tabs)

        export_layout = QHBoxLayout()
        self.export_report_button = QPushButton("Export Report")
        self.export_report_button.setToolTip("Save a text report of what the app did and the results.")
        self.export_report_button.clicked.connect(self.export_report)
        export_layout.addWidget(self.export_report_button)
        self.export_summary_button = QPushButton("Export 'What I did'…")
        self.export_summary_button.setToolTip("Export the 'What I did' summary as Markdown or text.")
        self.export_summary_button.clicked.connect(self.export_summary_dialog)
        export_layout.addWidget(self.export_summary_button)

        self.save_project_button = QPushButton("Save project…")
        self.save_project_button.clicked.connect(self.save_project_dialog)
        export_layout.addWidget(self.save_project_button)

        self.load_project_button = QPushButton("Open project…")
        self.load_project_button.clicked.connect(self.load_project_dialog)
        export_layout.addWidget(self.load_project_button)

        self.batch_predict_button = QPushButton("Batch predict with saved model…")
        self.batch_predict_button.clicked.connect(self.batch_predict_dialog)
        export_layout.addWidget(self.batch_predict_button)

        self.save_plot_button = QPushButton("Save last plot…")
        self.save_plot_button.clicked.connect(self.save_last_plot)
        export_layout.addWidget(self.save_plot_button)

        self.show_log_button = QPushButton("Show log")
        self.show_log_button.clicked.connect(self.show_log_dialog)
        export_layout.addWidget(self.show_log_button)

        self.export_bundle_button = QPushButton("Export analysis bundle…")
        self.export_bundle_button.clicked.connect(self.export_bundle_dialog)
        export_layout.addWidget(self.export_bundle_button)

        self.export_notebook_button = QPushButton("Export notebook…")
        self.export_notebook_button.clicked.connect(self.export_notebook_dialog)
        export_layout.addWidget(self.export_notebook_button)
        export_layout.addStretch()
        layout.addLayout(export_layout)

        # Coach bar
        self.coach_bar = QLabel("")
        self.coach_bar.setStyleSheet("padding:6px; background:#f4f4f4; border-top:1px solid #ccc;")
        layout.addWidget(self.coach_bar)
        central_widget.setLayout(layout)
        self.toggle_mode("Basic")

    def show_shortcuts_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard Shortcuts")
        table = QTableWidget(0, 2, dialog)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        shortcuts = [
            ("Ctrl+O", "Open CSV"),
            ("Ctrl+S", "Save project"),
            ("Ctrl+Shift+S", "Save model"),
            ("F5", "Smart Analyze"),
            ("Ctrl+R", "Run selected model"),
            ("Ctrl+Q", "Quit"),
            ("F1", "Open documentation"),
        ]
        table.setRowCount(len(shortcuts))
        for idx, (key, action) in enumerate(shortcuts):
            table.setItem(idx, 0, QTableWidgetItem(key))
            table.setItem(idx, 1, QTableWidgetItem(action))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout = QVBoxLayout()
        layout.addWidget(table)
        dialog.setLayout(layout)
        dialog.resize(400, 250)
        dialog.exec_()

    def _maybe_show_first_run_hint(self):
        cfg = _load_user_config()
        if cfg.get("first_run_shown"):
            return
        self._first_run_msg = QMessageBox(self)
        self._first_run_msg.setWindowTitle("Welcome to Modeling-GUI")
        self._first_run_msg.setText(
            "Tip: Press F1 for help, Ctrl+R for Smart Analyze.\n"
            "You can also run `modeling-gui-install-shortcut` in a terminal to create a desktop launcher."
        )
        self._first_run_msg.setStandardButtons(QMessageBox.Ok)
        self._first_run_msg.show()
        cfg["first_run_shown"] = True
        _save_user_config(cfg)

    def open_docs(self):
        docs_url = "https://chandrasekarnarayana.github.io/Modeling-GUI/"
        try:
            webbrowser.open(docs_url)
        except Exception:
            logger.warning("Failed to open docs URL: %s", docs_url)

    # --- Data loading helpers ---

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return
        try:
            df = load_csv(file_path)
            self.populate_from_dataframe(df)
            self.statusBar().showMessage(f"Loaded {file_path}", 5000)
            self.smart_button.setEnabled(True)
            self.last_data_path = file_path
            self.data_hash = compute_file_hash(file_path)
            self.coach_bar.setText(
                self.coach.update(CoachState.DATA_LOADED, f"Domain: {self.domain.name}. {self.domain.hints}")
            )
            self.run_data_quality_checks()
        except Exception as e:
            logger.exception("CSV load failed")
            self.show_error("I couldn't read this file. Please check that it's a valid CSV.")

    def load_demo_dataset(self):
        try:
            demo_path = pkg_resources.files("modeling_gui.data").joinpath(
                "demo_quickstart.csv"
            )
            df = load_csv(str(demo_path))
            self.populate_from_dataframe(df)
            self.auto_select_columns(df)
            self.model_combo.setCurrentText("OLS")
            self.statusBar().showMessage("Demo dataset loaded", 5000)
            self.smart_button.setEnabled(True)
            self.last_data_path = str(demo_path)
            self.data_hash = compute_file_hash(str(demo_path))
            self.coach_bar.setText(
                self.coach.update(
                    CoachState.DATA_LOADED,
                    f"Domain: {self.domain.name}. {self.domain.hints}",
                )
            )
            self.run_data_quality_checks()
        except Exception:
            logger.exception("Demo dataset load failed")
            self.show_error("Failed to load the demo dataset. Please reinstall or use your own CSV.")

    def populate_from_dataframe(self, df: pd.DataFrame):
        self.data = df
        self.x_list_widget.clear()
        self.y_combo.clear()
        self.csv_preview_table.clear()
        self.forecast_time_combo.clear()

        self.x_list_widget.addItems(self.data.columns)
        self.y_combo.addItems(self.data.columns)
        time_col = detect_time_column(self.data)
        self.forecast_time_combo.addItems([""] + list(self.data.columns))
        if time_col:
            idx = self.forecast_time_combo.findText(time_col)
            if idx >= 0:
                self.forecast_time_combo.setCurrentIndex(idx)

        row_count = min(10, len(self.data))
        self.csv_preview_table.setRowCount(row_count)
        self.csv_preview_table.setColumnCount(len(self.data.columns))
        self.csv_preview_table.setHorizontalHeaderLabels(self.data.columns)

        for i in range(row_count):
            for j, column in enumerate(self.data.columns):
                self.csv_preview_table.setItem(
                    i, j, QTableWidgetItem(str(self.data.iloc[i, j]))
                )
        self.smart_button.setEnabled(True)
        self.coach_bar.setText(self.coach.update(CoachState.DATA_LOADED))
        self.run_data_quality_checks()
        if len(self.data):
            self.local_row_spin.setMaximum(len(self.data) - 1)
            self.scenario_row_spin.setMaximum(len(self.data) - 1)

    def auto_select_columns(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        if len(df.columns) < 2:
            return
        feature_columns = list(df.columns[:-1])
        target_column = df.columns[-1]

        self.x_list_widget.clearSelection()
        for i in range(self.x_list_widget.count()):
            item = self.x_list_widget.item(i)
            if item.text() in feature_columns:
                item.setSelected(True)
        target_index = self.y_combo.findText(target_column)
        if target_index >= 0:
            self.y_combo.setCurrentIndex(target_index)

    def toggle_mode(self, text):
        self.basic_mode = text == "Basic"
        self.advanced_container.setVisible(not self.basic_mode)
        self.smart_button.setVisible(True)
        self.problem_label.setVisible(True)
        self.coach_bar.setText(self.coach.update(CoachState.DATA_UNLOADED))

    def change_domain(self, name):
        self.domain = get_domain_config(name)
        metrics_hint = ", ".join(self.domain.preferred_metrics)
        state = CoachState.DATA_LOADED if self.data is not None else CoachState.DATA_UNLOADED
        self.coach_bar.setText(
            self.coach.update(state, f"Domain: {self.domain.name}. {self.domain.hints} Focus on: {metrics_hint}.")
        )

    # --- Core modeling flow ---

    def smart_analyze(self):
        """
        AutoML-powered one-click analysis for non-coders.
        """
        if self.data is None:
            self.show_error("Please load a dataset first.")
            return
        self.coach_bar.setText(self.coach.update(CoachState.TARGET_SELECTED))

        y_column = self.y_combo.currentText()
        if not y_column:
            self.show_error("Select what you want to predict (target column).")
            return

        selected_x_items = self.x_list_widget.selectedItems()
        x_columns = [item.text() for item in selected_x_items] or [
            col for col in self.data.columns if col != y_column
        ]
        if not x_columns:
            self.show_error("Please select at least one feature column.")
            return

        df = self.data.copy()
        missing_strategy = self.missing_strategy_combo.currentData() or "none"
        try:
            df = apply_missing_strategy(df, missing_strategy)
        except Exception as e:
            self.show_error(f"Failed to handle missing values: {str(e)}")
            return
        self.last_training_features = df[x_columns].select_dtypes(include="number")
        self.last_training_target = df[y_column]

        detected_type, confidence = detect_problem_type(df, y_column)
        manual_override = None
        if detected_type is None or confidence < 0.6:
            choice, ok = QInputDialog.getItem(
                self,
                "Choose problem type",
                "Select what kind of problem this is:",
                ["Regression", "Classification", "Clustering"],
                0,
                False,
            )
            if not ok:
                QMessageBox.information(
                    self,
                    "Manual selection needed",
                    "I couldn't detect the problem type. Please choose manually.",
                )
                return
            manual_override = choice.lower()
            if manual_override == "clustering":
                self.run_smart_clustering(df, x_columns)
                return
        elif detected_type == "classification":
            manual_override = "classification"
        elif detected_type == "regression":
            manual_override = "regression"

        settings = {
            "test_size": self.test_size_input.value(),
            "random_state": self.random_state_input.value(),
            "time_budget": 30,
        }

        self.set_running(True)
        self.coach_bar.setText(self.coach.update(CoachState.ANALYSIS_RUNNING))
        try:
            result = run_automl(
                df,
                x_columns,
                y_column,
                settings=settings,
                problem_type=manual_override,
            )
            best = result.best_candidate
            self.last_automl_result = result
            self.model_manager.model = best.estimator
            self.last_feature_names = x_columns
            self.handle_feature_importances(best.estimator)
            self.problem_label.setText(f"Detected as: {result.problem_type}")

            metrics_text = self.format_metrics(best.metrics, result.problem_type)
            self.result_box.setPlainText(
                f"Smart Analyze results\nBest model: {best.name}\n\n{metrics_text}"
            )
            self.last_metrics = best.metrics
            self.last_summary_data = self.build_summary_data(
                df, x_columns, y_column, best, settings, missing_strategy, models_compared=len(result.candidate_models)
            )
            self.details_box.setPlainText(
                build_summary_text(self.last_summary_data, best.metrics)
            )
            self.last_automl_settings = settings
            self.explain_box.setPlainText(best.name)
            self.run_history.append(
                {
                    "model_name": best.name,
                    "metrics": self.last_metrics,
                    "domain": self.domain.name,
                    "run_type": "automl",
                    "problem_type": result.problem_type,
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "model_card_id": self.model_cards[-1].id if self.model_cards else "",
                }
            )
            self.model_cards.append(
                ModelCard(
                    name=best.name,
                    problem_type=result.problem_type,
                    domain=self.domain.name,
                    metrics=self.last_metrics,
                    data_summary={"rows": len(df), "cols": len(df.columns)},
                    explainability_summary=self.explain_box.toPlainText(),
                    model_obj=best.estimator,
                )
            )
            self.refresh_comparison_tab()
            self.refresh_model_cards_table()
            self.training_snapshot = compute_snapshot(df, infer_column_types(df))
            self.drift_issues = []
            self.refresh_drift_tab()

            if "residuals" in result.plots:
                y_true, y_pred = result.plots["residuals"]
                self.last_fig = plot_residuals(y_true, y_pred, title="Residuals (test)")
            if "confusion" in result.plots:
                y_true, y_pred = result.plots["confusion"]
                self.last_fig = plot_confusion_from_predictions(y_true, y_pred)
            if "feature_importance" in result.plots:
                importances, features = result.plots["feature_importance"]
                self.last_feature_importances = importances
                self.feature_importance_button.setEnabled(True)
                self.last_feature_names = list(features)
                self.last_fig = plot_feature_importance(importances, features)
            else:
                self.feature_importance_button.setEnabled(False)

            self.last_metadata = {
                "model_choice": "Smart Analyze",
                "x_columns": self.last_feature_names,
                "y_column": y_column,
                "problem_type": result.problem_type,
                "standardized": False,
                "missing_strategy": missing_strategy,
                "train_test_split": True,
                "test_size": settings["test_size"],
                "random_state": settings["random_state"],
                "version": __version__,
            }
            self.loaded_metadata = self.last_metadata
            self.predict_button.setEnabled(True)
            primary = choose_primary_metrics(self.domain.name, result.problem_type)
            primary_msg = ", ".join(primary) if primary else "n/a"
            self.coach_bar.setText(self.coach.update(CoachState.ANALYSIS_DONE, f"Primary metrics ({self.domain.name}): {primary_msg}"))
        except AutoMLDependencyMissing as dep_err:
            self.show_error(str(dep_err))
        except Exception as e:
            logger.exception("Smart Analyze failed")
            self.show_error("Smart Analyze failed. Check that your features are valid for modeling.")
        finally:
            self.set_running(False)

    def tune_current_model(self):
        """Run guided hyperparameter tuning for supported models."""
        if self.data is None:
            self.show_error("Load data before tuning.")
            return
        model_choice = self.model_combo.currentText()
        if model_choice not in ["Random Forest", "Gradient Boosting"]:
            self.show_error("Tuning is available for Random Forest and Gradient Boosting.")
            return
        selected_x_items = self.x_list_widget.selectedItems()
        x_columns = [item.text() for item in selected_x_items]
        y_column = self.y_combo.currentText()
        if not x_columns or not y_column:
            self.show_error("Select X and Y columns before tuning.")
            return

        X_numeric, y_clean = self.prepare_features_and_target(x_columns, y_column, model_choice)
        if X_numeric is None:
            return

        problem_type = "regression" if is_regression_target(y_clean) else "classification"
        base_estimator = (
            RandomForestRegressor(random_state=self.random_state_input.value())
            if model_choice == "Random Forest" and problem_type == "regression"
            else RandomForestClassifier(random_state=self.random_state_input.value())
            if model_choice == "Random Forest"
            else GradientBoostingRegressor(random_state=self.random_state_input.value())
            if problem_type == "regression"
            else GradientBoostingClassifier(random_state=self.random_state_input.value())
        )

        preset, ok = QInputDialog.getItem(
            self, "Tuning preset", "Choose search budget:", ["fast", "balanced", "thorough"], 0, False
        )
        if not ok:
            return
        self.set_running(True)
        try:
            base_estimator.fit(X_numeric, y_clean)
            pre_metrics = (
                regression_metrics(y_clean, base_estimator.predict(X_numeric))
                if problem_type == "regression"
                else classification_metrics(y_clean, base_estimator.predict(X_numeric))
            )
            tuned = tune_model(base_estimator, X_numeric, y_clean, problem_type, preset)
            self.model_manager.model = tuned.estimator
            self.last_feature_names = list(X_numeric.columns)
            self.last_metrics = tuned.metrics
            self.last_training_features = X_numeric
            self.last_training_target = y_clean
            self.handle_feature_importances(tuned.estimator)
            msg_lines = [
                f"Tuning preset: {preset}",
                f"Before: {self.format_metrics(pre_metrics, problem_type)}",
                f"After: {self.format_metrics(tuned.metrics, problem_type)}",
            ]
            self.result_box.setPlainText("\n".join(msg_lines))
            self.run_history.append(
                {
                    "model_name": tuned.name,
                    "metrics": tuned.metrics,
                    "domain": self.domain.name,
                    "run_type": "tuned",
                    "problem_type": problem_type,
                }
            )
            self.refresh_comparison_tab()
            primary = ", ".join(self.domain.primary_metrics)
            self.coach_bar.setText(
                self.coach.update(
                    CoachState.ANALYSIS_DONE,
                    f"Domain: {self.domain.name}. Primary metric focus: {primary}.",
                )
            )
            self.add_model_card(
                name=tuned.name,
                problem_type=problem_type,
                metrics=tuned.metrics,
                explain_summary="Tuned model",
                model_obj=tuned.estimator,
            )
        except Exception:
            logger.exception("Tuning failed")
            self.show_error("Tuning failed. Try a smaller preset or check your data.")
        finally:
            self.set_running(False)
    def run_smart_clustering(self, df, x_columns):
        """
        Simple fallback clustering when AutoML is not applicable.
        """
        try:
            X_numeric, dropped = select_numeric_columns(df, x_columns)
            if X_numeric.empty:
                self.show_error("No numeric features available for clustering.")
                return
            if dropped:
                QMessageBox.information(
                    self,
                    "Non-numeric ignored",
                    f"Non-numeric columns ignored for clustering: {', '.join(dropped)}",
                )
            model = self.model_manager.kmeans_clustering(X_numeric, n_clusters=3)
            self.problem_label.setText("Detected as: clustering")
            self.result_box.setPlainText(
                "Smart clustering complete.\nCluster centers:\n"
                f"{np.asarray(model.cluster_centers_)}"
            )
        except Exception as e:
            self.show_error(f"Clustering failed: {str(e)}")


    def run_model(self):
        if self.data is None:
            self.show_error("Please load a dataset first.")
            return

        selected_x_items = self.x_list_widget.selectedItems()
        x_columns = [item.text() for item in selected_x_items]
        y_column = self.y_combo.currentText()
        model_choice = self.model_combo.currentText()

        if model_choice != "KMeans Clustering" and not x_columns:
            self.show_error("Please select at least one X column.")
            return
        if model_choice not in ["KMeans Clustering"] and not y_column:
            self.show_error("Please select a Y column.")
            return
        if model_choice == "KMeans Clustering":
            self.coach_bar.setText(self.coach.update(CoachState.DATA_LOADED, "Hint: KMeans needs numeric features. Try scaling and different k values."))
        if model_choice == "Rolling Least Squares" and self.train_test_checkbox.isChecked():
            self.show_error("Train/test split is not supported for Rolling Least Squares.")
            return

        self.set_running(True)
        self.coach_bar.setText(self.coach.update(CoachState.ANALYSIS_RUNNING))
        try:
            X_numeric, y_clean = self.prepare_features_and_target(x_columns, y_column, model_choice)
            if X_numeric is None:
                return

            use_split = self.train_test_checkbox.isChecked() and model_choice not in [
                "KMeans Clustering",
                "Rolling Least Squares",
            ]
            test_size = self.test_size_input.value()
            random_state = self.random_state_input.value()

            if use_split:
                X_train, X_test, y_train, y_test = split_data(
                    X_numeric, y_clean, test_size=test_size, random_state=random_state
                )
            else:
                X_train, X_test, y_train, y_test = X_numeric, None, y_clean, None

            # Standardization
            scaler = None
            if self.standardize_checkbox.isChecked():
                X_train, scaler = standardize_features(X_train, X_train.columns)
                if X_test is not None:
                    X_test = X_test.copy()
                    X_test[X_test.columns] = scaler.transform(X_test)

            model, problem_type = self.fit_model(model_choice, X_train, y_train)
            if model is None:
                return

            self.last_training_features = X_numeric
            self.last_training_target = y_clean
            self.last_scaler = scaler
            self.last_feature_names = list(X_train.columns)
            self.last_metadata = {
                "model_choice": model_choice,
                "x_columns": self.last_feature_names,
                "y_column": y_column,
                "problem_type": problem_type,
                "standardized": self.standardize_checkbox.isChecked(),
                "missing_strategy": self.missing_strategy_combo.currentData(),
                "train_test_split": use_split,
                "test_size": test_size,
                "random_state": random_state,
                "version": __version__,
            }
            self.loaded_metadata = self.last_metadata
            self.predict_button.setEnabled(True)
            if problem_type:
                self.problem_label.setText(f"Problem type: {problem_type}")
            self.coach_bar.setText(self.coach.update(CoachState.ANALYSIS_DONE))

            metrics_text = ""
            metrics = {}
            if problem_type == "regression":
                if use_split and X_test is not None:
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    metrics = regression_metrics(y_test, y_test_pred, {})
                    primary = choose_primary_metrics(self.domain.name, problem_type)
                    metrics_text = "\n".join([f"{k}: {metrics.get(k, float('nan')):.4f}" for k in primary])
                    self.last_fig = plot_residuals(y_test, y_test_pred, title="Residuals (test)")
                else:
                    y_pred = model.predict(X_train)
                    metrics = regression_metrics(y_train, y_pred, {})
                    primary = choose_primary_metrics(self.domain.name, problem_type)
                    metrics_text = "\n".join([f"{k}: {metrics.get(k, float('nan')):.4f}" for k in primary])
                    self.last_fig = plot_residuals(y_train, y_pred, title="Residuals")

            elif problem_type == "classification":
                if use_split and X_test is not None:
                    y_test_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                    metrics = classification_metrics(y_test, y_test_pred, y_proba)
                    primary = choose_primary_metrics(self.domain.name, problem_type)
                    metrics_text = "\n".join([f"{k}: {metrics.get(k, float('nan')):.4f}" for k in primary])
                    self.last_fig = plot_confusion_matrix(model, X_test, y_test)
                else:
                    y_pred = model.predict(X_train)
                    y_proba = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
                    metrics = classification_metrics(y_train, y_pred, y_proba)
                    primary = choose_primary_metrics(self.domain.name, problem_type)
                    metrics_text = "\n".join([f"{k}: {metrics.get(k, float('nan')):.4f}" for k in primary])
                    self.last_fig = plot_confusion_matrix(model, X_train, y_train)

            if model_choice == "Random Forest":
                try:
                    plot_tree_diagram(model)
                except Exception:
                    pass

            self.handle_feature_importances(model)
            self.result_box.setPlainText(
                f"Model: {model_choice}\n{metrics_text}".strip() or "Model trained."
            )
            self.last_metrics = metrics if problem_type in ("regression", "classification") else {}
            self.last_summary_data = self.build_summary_data(
                self.data,
                x_columns,
                y_column,
                type("obj", (), {"problem_type": problem_type, "best_model": model, "summary": ""}),
                {"test_size": test_size, "random_state": random_state},
                self.missing_strategy_combo.currentData(),
                used_standardization=self.standardize_checkbox.isChecked(),
            )
            self.details_box.setPlainText(
                build_summary_text(self.last_summary_data, self.last_metrics)
            )
            self.run_history.append(
                {
                    "model_name": model_choice,
                    "metrics": self.last_metrics,
                    "domain": self.domain.name,
                    "run_type": "manual",
                    "problem_type": problem_type,
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "model_card_id": self.model_cards[-1].id if self.model_cards else "",
                }
            )
            self.refresh_comparison_tab()
            if self.data is not None:
                self.training_snapshot = compute_snapshot(self.data, infer_column_types(self.data))
                self.drift_issues = []
                self.refresh_drift_tab()
            self.add_model_card(
                name=model_choice,
                problem_type=problem_type or "",
                metrics=self.last_metrics,
                explain_summary="",
                model_obj=model,
            )
            primary_msg = ", ".join(primary) if primary else "n/a"
            self.coach_bar.setText(self.coach.update(CoachState.ANALYSIS_DONE, f"Primary metrics ({self.domain.name}): {primary_msg}"))
        except Exception as e:
            logger.exception("Model run failed")
            suggestions = analyze_failure_context(
                e, self.data, {"x_columns": x_columns, "y_column": y_column}
            )
            hint = "\n".join([s.short_message for s in suggestions][:3]) or "Check your column selections and data types."
            self.show_error(f"The model could not be trained. {hint}")
        finally:
            self.set_running(False)


    def run_data_quality_checks(self):
        issues = []
        if self.data is None:
            return
        issues.extend(validate_schema(self.data))
        target = self.y_combo.currentText() if self.y_combo.count() else None
        issues.extend(detect_missing_issues(self.data, target))
        issues.extend(detect_type_mismatches(self.data))
        if target:
            leakage = detect_potential_leakage([i.text() for i in self.x_list_widget.selectedItems()], target)
            if leakage:
                issues.append(leakage)
            imbalance_issue = detect_imbalance(self.data[target])
            if imbalance_issue:
                issues.append(imbalance_issue)
        if issues:
            text = "\n".join(
                [
                    f"[{i.severity.upper()}] {i.message} (Suggestion: {i.suggestion or 'n/a'})"
                    for i in issues
                ]
            )
            self.data_quality_box.setPlainText(text)
            self.coach_bar.setText(self.coach.update(CoachState.DATA_LOADED, "Data quality warnings detected."))
        else:
            self.data_quality_box.setPlainText("No major data quality issues detected.")

    def prepare_features_and_target(self, x_columns, y_column, model_choice):
        try:
            relevant_columns = x_columns.copy()
            if model_choice != "KMeans Clustering":
                relevant_columns.append(y_column)
            df_selected = self.data[relevant_columns]
            missing_strategy = self.missing_strategy_combo.currentData()
            df_selected = apply_missing_strategy(df_selected, missing_strategy)

            if df_selected.empty:
                self.show_error("Dataset is empty after applying missing value strategy.")
                return None, None

            y_clean = None
            if model_choice != "KMeans Clustering":
                y_clean = df_selected[y_column]

            X_numeric, dropped = select_numeric_columns(df_selected, x_columns)
            if dropped:
                QMessageBox.warning(
                    self,
                    "Non-numeric columns dropped",
                    f"The following columns were skipped because they are non-numeric: {', '.join(dropped)}",
                )
            if X_numeric.empty:
                self.show_error("No numeric feature columns remain after filtering.")
                return None, None

            if y_clean is not None and not is_regression_target(y_clean):
                # Leave classification targets as-is; regression targets must be numeric.
                pass
            elif y_clean is not None:
                y_clean = pd.to_numeric(y_clean, errors="coerce")
                if y_clean.isna().any():
                    y_clean = y_clean.dropna()
                    X_numeric = X_numeric.loc[y_clean.index]
            return X_numeric, y_clean
        except Exception as e:
            self.show_error(f"Failed to prepare data: {str(e)}")
            return None, None

    def fit_model(self, model_choice, X, y):
        if model_choice == "OLS":
            return self.model_manager.ols(X, y), "regression"
        if model_choice == "Rolling Least Squares":
            return self.model_manager.rolling_ls(X, y), "rolling"
        if model_choice == "Random Forest":
            dialog = RandomForestDialog(self)
            if dialog.exec_() != dialog.Accepted:
                return None, None
            model = self.model_manager.random_forest(
                X, y, n_estimators=dialog.n_estimators, max_depth=dialog.max_depth
            )
            problem_type = "regression" if is_regression_target(y) else "classification"
            return model, problem_type
        if model_choice == "Gradient Boosting":
            dialog = GradientBoostDialog(self)
            if dialog.exec_() != dialog.Accepted:
                return None, None
            model = self.model_manager.gradient_boost(
                X,
                y,
                n_estimators=dialog.n_estimators,
                learning_rate=dialog.learning_rate,
                max_depth=dialog.max_depth,
            )
            problem_type = "regression" if is_regression_target(y) else "classification"
            return model, problem_type
        if model_choice == "KMeans Clustering":
            dialog = KMeansDialog(self)
            if dialog.exec_() != dialog.Accepted:
                return None, None
            return self.model_manager.kmeans_clustering(
                X, n_clusters=dialog.n_clusters
            ), "clustering"
        if model_choice == "Gaussian Fitting":
            if X.shape[1] > 1:
                QMessageBox.warning(
                    self,
                    "Too many features",
                    "Gaussian fitting uses only the first selected feature.",
                )
                X = X.iloc[:, [0]]
            params = self.model_manager.gaussian_fit(X, y)
            plot_curve_fit(X, y, params, "gaussian")
            return self.model_manager.model, "regression"
        if model_choice == "Exponential Fitting":
            if X.shape[1] > 1:
                QMessageBox.warning(
                    self,
                    "Too many features",
                    "Exponential fitting uses only the first selected feature.",
                )
                X = X.iloc[:, [0]]
            params = self.model_manager.exponential_fit(X, y)
            plot_curve_fit(X, y, params, "exponential")
            return self.model_manager.model, "regression"
        return None, None

    # --- Persistence & post-processing ---

    def add_model_card(self, name: str, problem_type: str, metrics: dict, explain_summary: str, model_obj=None):
        """Append a model card entry and refresh the UI."""
        card = ModelCard(
            name=name,
            problem_type=problem_type,
            domain=self.domain.name,
            metrics=metrics,
            data_summary={"rows": len(self.data) if self.data is not None else 0, "cols": len(self.data.columns) if self.data is not None else 0},
            explainability_summary=explain_summary,
            model_obj=model_obj,
        )
        self.model_cards.append(card)
        self.refresh_model_cards_table()

    def handle_feature_importances(self, model):
        if hasattr(model, "feature_importances_"):
            self.last_feature_importances = getattr(model, "feature_importances_")
            self.feature_importance_button.setEnabled(True)
            self.update_scenario_controls()
        else:
            self.last_feature_importances = None
            self.feature_importance_button.setEnabled(False)
            self.update_scenario_controls()

    def show_feature_importance(self):
        if self.last_feature_importances is None:
            self.show_error("No feature importances available for the last model.")
            return
        feature_names = self.last_feature_names or [
            f"feature_{i}" for i in range(len(self.last_feature_importances))
        ]
        plot_feature_importance(self.last_feature_importances, feature_names)

    def compute_global_explanation(self):
        """Compute and store a global explanation for the active model."""
        if self.model_manager.model is None or self.last_feature_names is None:
            self.show_error("Train a model before running explanations.")
            return None
        X = self.last_training_features
        if X is None and self.data is not None:
            X = self.data[self.last_feature_names].select_dtypes(include="number")
        if X is None or X.empty:
            self.show_error("No feature data available for explanations.")
            return None
        self.local_row_spin.setMaximum(max(len(X) - 1, 0))
        exp = explain_global(self.model_manager.model, X, self.last_feature_names)
        self.last_global_explanation = exp
        self.explain_box.setPlainText(exp.summary_text)
        self.update_scenario_controls()
        return exp

    def show_global_explanation_plot(self):
        exp = self.last_global_explanation or self.compute_global_explanation()
        if not exp:
            return
        self.last_fig = plot_feature_importance(exp.importance_values, exp.feature_names)

    def show_local_explanation_plot(self):
        if self.model_manager.model is None:
            self.show_error("Train a model before running local explanations.")
            return
        X = self.last_training_features
        if X is None and self.data is not None and self.last_feature_names:
            X = self.data[self.last_feature_names].select_dtypes(include="number")
        if X is None or X.empty:
            self.show_error("No feature data available for explanations.")
            return
        row_idx = min(self.local_row_spin.value(), len(X) - 1)
        exp = explain_local(self.model_manager.model, X, self.last_feature_names, index=row_idx)
        self.last_local_explanation = exp
        self.last_fig = plot_local_contributions(
            exp.feature_names,
            exp.contributions,
            exp.predicted_value,
            exp.base_value,
        )
        self.explain_box.append(
            f"\nRow {row_idx} prediction: {exp.predicted_value:.4f} (method: {exp.method})"
        )

    def update_scenario_controls(self):
        """Refresh scenario feature choices based on latest model info."""
        if not self.last_feature_names:
            return
        features = list(self.last_feature_names)
        if self.last_feature_importances is not None and len(self.last_feature_importances) == len(features):
            order = np.argsort(self.last_feature_importances)[::-1]
            features = list(np.asarray(features)[order])
        top = features[:5] if len(features) > 5 else features
        self.scenario_feature_combo.blockSignals(True)
        self.scenario_feature_combo.clear()
        self.scenario_feature_combo.addItems(top)
        self.scenario_feature_combo.blockSignals(False)
        self.update_scenario_range()

    def update_scenario_range(self):
        feat = self.scenario_feature_combo.currentText()
        if not feat or self.data is None or feat not in self.data.columns:
            return
        series = pd.to_numeric(self.data[feat], errors="coerce")
        series = series.dropna()
        if series.empty:
            return
        min_val, max_val = float(series.min()), float(series.max())
        self.scenario_value_spin.setRange(min_val, max_val)
        mid = float(series.median())
        self.scenario_value_spin.setValue(mid)

    def apply_scenario_change(self):
        """Apply scenario tweaks and compute a new prediction."""
        if self.model_manager.model is None or self.data is None or not self.last_feature_names:
            self.show_error("Train a model first to run scenario testing.")
            return
        feat = self.scenario_feature_combo.currentText()
        if not feat:
            self.show_error("Select a feature to adjust.")
            return
        try:
            base_idx = min(self.scenario_row_spin.value(), len(self.data) - 1)
            base_row = self.data.iloc[base_idx][self.last_feature_names].copy()
            base_row = pd.to_numeric(base_row, errors="coerce")
            base_row = base_row.fillna(base_row.median())
            baseline_pred = float(self.model_manager.predict([base_row.values])[0])
            new_row = base_row.copy()
            new_row.loc[feat] = self.scenario_value_spin.value()
            new_pred = float(self.model_manager.predict([new_row.values])[0])
            delta = new_pred - baseline_pred
            self.scenario_output.setPlainText(
                f"Scenario for row {base_idx}:\n"
                f"{feat}: {base_row.loc[feat]:.4f} -> {new_row.loc[feat]:.4f}\n"
                f"Prediction: {baseline_pred:.4f} -> {new_pred:.4f} (Δ {delta:+.4f})"
            )
        except Exception:
            logger.exception("Scenario testing failed")
            self.show_error("Scenario testing failed for this model/feature.")

    def show_partial_dependence_plot(self):
        """Display a simple partial dependence plot for the selected feature."""
        if self.model_manager.model is None or self.last_training_features is None:
            self.show_error("Train a model before plotting partial dependence.")
            return
        feat = self.scenario_feature_combo.currentText()
        if not feat:
            self.show_error("Select a feature to plot.")
            return
        try:
            grid, preds = compute_partial_dependence(self.model_manager.model, self.last_training_features, feat)
            if grid.size == 0:
                self.show_error("Could not compute partial dependence for this feature.")
                return
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(grid, preds, marker="o")
            ax.set_title(f"Partial dependence of {feat}")
            ax.set_xlabel(feat)
            ax.set_ylabel("Prediction")
            ax.grid(True)
            fig.tight_layout()
            fig.show()
            self.last_fig = fig
        except Exception:
            logger.exception("Partial dependence failed")
            self.show_error("Partial dependence could not be computed for this model.")

    def run_forecast(self):
        """Run forecasting workflow with optional ARIMA/Prophet backends."""
        if self.data is None:
            self.show_error("Load a dataset before forecasting.")
            return
        date_col = self.forecast_time_combo.currentText() or detect_time_column(self.data)
        target_col = self.y_combo.currentText()
        if not date_col or date_col not in self.data.columns:
            self.show_error("Select a valid time column for forecasting.")
            return
        if not target_col:
            self.show_error("Select a target column to forecast.")
            return
        model_type = self.forecast_model_combo.currentText()
        growth = self.forecast_growth_combo.currentText()
        yearly = self.forecast_yearly_spin.value() if self.forecast_yearly_spin.value() > 0 else self.forecast_yearly_checkbox.isChecked()
        weekly = self.forecast_weekly_spin.value() if self.forecast_weekly_spin.value() > 0 else self.forecast_weekly_checkbox.isChecked()
        daily = self.forecast_daily_spin.value() if self.forecast_daily_spin.value() > 0 else self.forecast_daily_checkbox.isChecked()
        holidays = self.forecast_holidays_combo.currentText() if self.forecast_holidays_checkbox.isChecked() else None
        config = ForecastConfig(
            date_col=date_col,
            target_col=target_col,
            horizon=self.forecast_horizon_spin.value(),
            model_type=model_type,
            growth=growth,  # type: ignore[arg-type]
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
            holidays_country=holidays,
        )
        try:
            result = train_forecast_model(self.data[[date_col, target_col]], config)
            self.last_fig = plot_forecast(
                self.data[date_col],
                result.y_train,
                result.y_valid,
                result.y_pred_valid,
                result.y_forecast,
            )
            metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in result.metrics.items()])
            extra = f"\nNote: {result.info.get('warning')}" if result.info.get("warning") else ""
            self.result_box.setPlainText(f"Forecast model: {model_type}\n{metrics_text}{extra}")
            primary = ", ".join(self.domain.primary_metrics)
            self.coach_bar.setText(
                self.coach.update(
                    CoachState.ANALYSIS_DONE,
                    f"Domain: {self.domain.name}. Primary metric focus: {primary}.",
                )
            )
        except Exception:
            logger.exception("Forecasting failed")
            self.show_error("Forecasting failed. Check that your date/target columns are valid.")

    def save_model(self):
        if self.model_manager.model is None:
            self.show_error("Train a model before saving.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save model", "", "Joblib Files (*.joblib);;All Files (*)"
        )
        if not file_path:
            return
        metadata = self.last_metadata or {}
        metadata["feature_names"] = self.last_feature_names
        metadata["scaler"] = self.last_scaler
        try:
            save_model_bundle(file_path, self.model_manager.model, metadata)
            self.statusBar().showMessage(f"Model saved to {file_path}", 5000)
        except Exception as e:
            self.show_error(f"Failed to save model: {str(e)}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load model", "", "Joblib Files (*.joblib);;All Files (*)"
        )
        if not file_path:
            return
        try:
            bundle = load_model_bundle(file_path)
            self.model_manager.model = bundle.get("model")
            self.loaded_metadata = bundle.get("metadata", {})
            self.last_scaler = self.loaded_metadata.get("scaler")
            self.last_feature_names = self.loaded_metadata.get("feature_names", [])
            self.handle_feature_importances(self.model_manager.model)
            self.predict_button.setEnabled(True)

            meta_lines = [f"{k}: {v}" for k, v in self.loaded_metadata.items() if k != "scaler"]
            self.result_box.setPlainText(
                "Loaded model bundle:\n" + "\n".join(meta_lines)
            )
            self.statusBar().showMessage(f"Loaded model from {file_path}", 5000)
        except Exception as e:
            self.show_error(f"Failed to load model: {str(e)}")

    def predict_with_loaded_model(self):
        if self.model_manager.model is None or self.loaded_metadata is None:
            self.show_error("Load a model bundle first.")
            return
        if self.data is None:
            self.show_error("Load a CSV to run predictions.")
            return
        feature_names = self.loaded_metadata.get("feature_names", [])
        missing_strategy = self.loaded_metadata.get("missing_strategy", "none")

        try:
            X = self.data[feature_names]
            X = apply_missing_strategy(X, missing_strategy)
            if self.last_scaler is not None:
                X = X.copy()
                X[X.columns] = self.last_scaler.transform(X)
            preds = self.model_manager.predict(X)
            preview = ", ".join(map(lambda v: str(v)[:12], np.asarray(preds)[:5]))
            self.result_box.setPlainText(
                f"Predictions (first 5): {preview}\nTotal predictions: {len(preds)}"
            )
        except Exception as e:
            self.show_error(f"Failed to run predictions: {str(e)}")

    # --- UI helpers ---

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.statusBar().showMessage(message, 5000)
        self.coach_bar.setText(self.coach.update(CoachState.ERROR))
        logger.error(message, exc_info=True)

    def format_metrics(self, metrics, problem_type):
        """
        Nicely format metrics for display.
        """
        if not metrics:
            return ""
        lines = []
        for k, v in metrics.items():
            try:
                lines.append(f"{k}: {float(v):.4f}")
            except Exception:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def build_summary_data(
        self,
        df,
        x_cols,
        y_col,
        result_obj,
        settings,
        missing_strategy,
        used_standardization=False,
    ):
        """
        Build a structured summary dictionary for transparency and export.
        """
        data_summary = {
            "rows": len(df),
            "cols": len(df.columns),
            "types": {
                "numeric": len(df.select_dtypes(include="number").columns),
                "categorical": len(df.select_dtypes(exclude="number").columns),
            },
        }
        preprocessing = {
            "missing": missing_strategy,
            "scaling": "standardize" if used_standardization else "none",
            "categoricals": "Auto-handled by AutoML" if hasattr(result_obj, "best_model") else "n/a",
        }
        modeling = {
            "problem_type": getattr(result_obj, "problem_type", "n/a"),
            "algorithms": [getattr(result_obj, "best_model", None).__class__.__name__]
            if hasattr(result_obj, "best_model")
            else [],
            "best_model": getattr(result_obj, "best_model", None).__class__.__name__
            if hasattr(result_obj, "best_model")
            else getattr(result_obj, "summary", ""),
            "cv": f"{settings.get('n_splits', 5)}-fold" if settings else "n/a",
        }
        evaluation = {
            "split": f"test_size={settings.get('test_size', 'n/a')}" if settings else "n/a",
            "primary": {},
            "secondary": {},
        }
        return {
            "data": data_summary,
            "preprocessing": preprocessing,
            "modeling": modeling,
            "evaluation": evaluation,
            "x_columns": x_cols,
            "y_column": y_col,
            "domain": self.domain.name,
        }

    def set_running(self, running: bool):
        if running:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Running…")
        else:
            QApplication.restoreOverrideCursor()
            self.statusBar().showMessage("Ready", 3000)

    # --- Reporting & Project helpers ---

    def export_report(self):
        """
        Export a simple text/markdown report of the last run.
        """
        if not self.last_summary_data:
            self.show_error("Run Smart Analyze or a model first to export a report.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export report", "", "Markdown Files (*.md);;Text Files (*.txt)"
        )
        if not path:
            return
        try:
            report_text = build_summary_text(self.last_summary_data, self.last_metrics)
            saved_path = export_report(path, report_text)
            self.statusBar().showMessage(f"Report saved to {saved_path}", 5000)
        except Exception as e:
            logger.exception("Report export failed")
            self.show_error("Report export failed. Please choose another location or check file permissions.")

    def export_summary_dialog(self):  # pragma: no cover - GUI
        if not self.last_summary_data:
            self.show_error("Run Smart Analyze or a model first to export a summary.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export 'What I did'", "", "Markdown Files (*.md);;Text Files (*.txt)"
        )
        if not path:
            return
        try:
            summary_text = build_summary_text(self.last_summary_data, self.last_metrics or {})
            saved_path = export_report(path, summary_text)
            self.statusBar().showMessage(f"'What I did' summary saved to {saved_path}", 5000)
        except Exception:
            logger.exception("Summary export failed")
            self.show_error("Summary export failed. Please choose another location or check file permissions.")

    def save_last_plot(self):  # pragma: no cover - GUI
        if self.last_fig is None:
            self.show_error("No plot available to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot", "", "PNG Files (*.png);;PDF Files (*.pdf)"
        )
        if not path:
            return
        try:
            self.last_fig.savefig(path, bbox_inches="tight")
            self.statusBar().showMessage(f"Plot saved to {path}", 5000)
        except Exception:
            logger.exception("Plot save failed")
            self.show_error("Could not save plot. Please try another location.")

    def save_project_dialog(self):
        if not self.last_summary_data:
            self.show_error("Run Smart Analyze or a model first to save a project.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", "", "Modeling GUI Project (*.mgui)"
        )
        if not path:
            return
        project_data = {
            "version": __version__,
            "data_path": self.last_data_path,
            "data_hash": getattr(self, "data_hash", None),
            "domain": self.domain.name,
            "x_columns": self.last_feature_names,
            "y_column": self.last_metadata.get("y_column") if self.last_metadata else None,
            "automl_settings": getattr(self, "last_automl_settings", {}),
            "summary": self.last_summary_data,
            "metrics": self.last_metrics,
            "model": self.model_manager.model,
            "training_snapshot": self.training_snapshot and self.training_snapshot.__dict__ if self.training_snapshot else None,
            "dataframe_for_snapshot": self.data if self.data is not None else None,
            "model_cards": [dict({k: v for k, v in asdict(c).items() if k != "model_obj"}) for c in self.model_cards],
            "run_history": self.run_history,
        }
        try:
            saved = save_project(path, project_data)
            self.statusBar().showMessage(f"Project saved to {saved}", 5000)
        except Exception as e:
            logger.exception("Project save failed")
            self.show_error("Project could not be saved. Please try a different path or filename.")

    def load_project_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "", "Modeling GUI Project (*.mgui)"
        )
        if not path:
            return
        try:
            data = load_project(path)
            stored_version = data.get("version")
            if stored_version and stored_version != __version__:
                QMessageBox.warning(
                    self,
                    "Version mismatch",
                    f"Project created with version {stored_version}; current is {__version__}. "
                    "Model reload may fail.",
                )
            domain_name = data.get("domain", "Generic")
            self.domain_combo.setCurrentText(domain_name)
            snap = data.get("training_snapshot")
            if isinstance(snap, dict):
                self.training_snapshot = SimpleNamespace(**snap)
            else:
                self.training_snapshot = snap

            data_path = data.get("data_path")
            loaded_df = None
            if data_path and Path(data_path).exists():
                loaded_df = load_csv(data_path)
            else:
                choice = QMessageBox.question(
                    self,
                    "Data file missing",
                    "Original data file not found. Would you like to locate it?",
                )
                if choice == QMessageBox.Yes:
                    new_path, _ = QFileDialog.getOpenFileName(
                        self, "Locate data file", "", "CSV Files (*.csv)"
                    )
                    if new_path:
                        loaded_df = load_csv(new_path)
                        self.last_data_path = new_path
                        self.data_hash = compute_file_hash(new_path)
            if loaded_df is not None:
                self.populate_from_dataframe(loaded_df)
                self.y_combo.setCurrentText(data.get("y_column", ""))
                for i in range(self.x_list_widget.count()):
                    item = self.x_list_widget.item(i)
                    item.setSelected(item.text() in data.get("x_columns", []))
                self.problem_label.setText(f"Problem type: {data.get('summary', {}).get('modeling', {}).get('problem_type', '')}")

            model_loaded = data.get("model") is not None
            if model_loaded:
                self.model_manager.model = data["model"]
                self.predict_button.setEnabled(True)
            else:
                QMessageBox.information(
                    self,
                    "Model unavailable",
                    "Model could not be reloaded. You may need to rerun Smart Analyze with the current data.",
                )

            self.last_summary_data = data.get("summary")
            self.last_metrics = data.get("metrics", {})
            summary_text = build_summary_text(self.last_summary_data, self.last_metrics) if self.last_summary_data else ""
            self.details_box.setPlainText(summary_text)
            self.result_box.setPlainText(summary_text)
            self.coach_bar.setText(self.coach.update(CoachState.ANALYSIS_DONE))
            self.model_cards = []
            for card_dict in data.get("model_cards", []):
                try:
                    self.model_cards.append(ModelCard(**card_dict))
                except Exception:
                    continue
            self.refresh_model_cards_table()
            self.run_history = data.get("run_history", [])
            self.refresh_comparison_tab()
        except Exception as e:
            logger.exception("Project load failed")
            self.show_error("Project could not be opened. It might be from a different version or corrupted.")

    def batch_predict_dialog(self):
        """Batch prediction using a saved model on a new CSV."""
        if self.model_manager.model is None:
            self.show_error("Load or train a model first.")
            return
        csv_path, _ = QFileDialog.getOpenFileName(self, "Select CSV for batch prediction", "", "CSV Files (*.csv)")
        if not csv_path:
            return
        try:
            new_df = load_csv(csv_path)
            missing_strategy = self.missing_strategy_combo.currentData() if self.missing_strategy_combo.currentData() else "none"
            new_df = apply_missing_strategy(new_df, missing_strategy)
            X = new_df[self.last_feature_names]
            if self.last_scaler is not None:
                X = X.copy()
                X[X.columns] = self.last_scaler.transform(X)
            preds = self.model_manager.predict(X)
            output = new_df.copy()
            output["prediction"] = preds
            output["model_id"] = self.last_metadata.get("model_choice") if self.last_metadata else "model"
            output["predicted_at"] = pd.Timestamp.utcnow()
            if self.training_snapshot:
                snap_obj = self.training_snapshot
                col_types = (
                    getattr(snap_obj, "col_types", infer_column_types(new_df))
                    if not isinstance(snap_obj, dict)
                    else snap_obj.get("col_types", infer_column_types(new_df))
                )
                snap_struct = snap_obj
                if isinstance(snap_struct, dict):
                    snap_struct = SimpleNamespace(**snap_struct)
                drift_issues = compare_snapshot(
                    snap_struct,
                    new_df,
                    col_types,
                )
                self.drift_issues = drift_issues
                if any(i.severity == "warning" for i in drift_issues):
                    self.coach_bar.setText(
                        self.coach.update(
                            CoachState.ERROR,
                            "New data looks different from training data; predictions may be unreliable.",
                        )
                    )
                self.refresh_drift_tab()
            save_path, _ = QFileDialog.getSaveFileName(self, "Save predictions", "", "CSV Files (*.csv)")
            if save_path:
                output.to_csv(save_path, index=False)
                self.statusBar().showMessage(f"Predictions saved to {save_path}", 5000)
        except Exception:
            logger.exception("Batch prediction failed")
            self.show_error("Batch prediction failed. Ensure the CSV has the required columns.")

    def show_log_dialog(self):  # pragma: no cover - GUI
        dlg = LogDialog(memory_handler, self)
        dlg.exec_()

    def export_bundle_dialog(self):  # pragma: no cover - GUI
        if not self.last_summary_data:
            self.show_error("Save or run a project before exporting a bundle.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export analysis bundle", "", "Zip Files (*.zip)")
        if not path:
            return
        project_data = self.last_summary_data.copy() if self.last_summary_data else {}
        try:
            export_bundle(project_data, path)
            self.statusBar().showMessage(f"Bundle exported to {path}", 5000)
        except Exception:
            logger.exception("Bundle export failed")
            self.show_error("Failed to export bundle.")

    def export_notebook_dialog(self):  # pragma: no cover - GUI
        if not self.last_summary_data:
            self.show_error("Run an analysis before exporting a notebook.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export notebook", "", "Notebook Files (*.ipynb)")
        if not path:
            return
        project = {
            "data_path": self.last_data_path,
            "x_columns": self.last_feature_names,
        }
        try:
            export_notebook(path, project)
            self.statusBar().showMessage(f"Notebook exported to {path}", 5000)
        except Exception:
            logger.exception("Notebook export failed")
            self.show_error("Failed to export notebook.")

    def refresh_comparison_tab(self):
        """Refresh the model comparison tab content."""
        text_parts = [comparison_text(self.run_history)]
        if self.last_automl_result:
            lines = ["AutoML leaderboard:"]
            for cand in self.last_automl_result.candidate_models:
                best_flag = " (best)" if cand.is_best else ""
                lines.append(f"- {cand.name}{best_flag}: {cand.metrics}")
            text_parts.append("\n".join(lines))
        self.comparison_box.setPlainText("\n\n".join(text_parts))

    def refresh_drift_tab(self):
        """Show drift issues in the Data Drift tab."""
        if not self.drift_issues:
            self.drift_box.setPlainText("No drift detected.")
            return
        lines = []
        for issue in self.drift_issues:
            lines.append(f"[{issue.severity}] {issue.message} ({issue.suggestion or 'n/a'})")
        self.drift_box.setPlainText("\n".join(lines))

    def show_leaderboard_plot(self):  # pragma: no cover - GUI plotting
        entries = []
        if self.last_automl_result:
            primary = choose_primary_metrics(self.domain.name, getattr(self.last_automl_result, "problem_type", "regression"))[0]
            for cand in self.last_automl_result.candidate_models:
                val = cand.metrics.get(primary)
                if val is not None:
                    entries.append((cand.name, val, cand.is_best))
        for run in self.run_history:
            metrics = run.get("metrics") or {}
            if not metrics:
                continue
            primary_list = choose_primary_metrics(self.domain.name, run.get("problem_type", "regression"))
            metric_name = primary_list[0] if primary_list else next(iter(metrics))
            val = metrics.get(metric_name)
            if val is not None:
                entries.append((run.get("model_name"), val, False))
        if not entries:
            self.show_error("No leaderboard data available yet.")
            return
        import matplotlib.pyplot as plt

        names = [e[0] for e in entries]
        vals = [e[1] for e in entries]
        colors = ["#2ca02c" if e[2] else "#1f77b4" for e in entries]
        fig, ax = plt.subplots(figsize=(8, 5))
        positions = range(len(names))
        ax.bar(positions, vals, color=colors)
        ax.set_ylabel("Primary metric")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title("Model leaderboard")
        fig.tight_layout()
        fig.show()
        self.last_fig = fig

    def refresh_model_cards_table(self):
        """Refresh the model cards table UI."""
        if not hasattr(self, "cards_table"):
            return
        self.cards_table.setRowCount(len(self.model_cards))
        for row, card in enumerate(self.model_cards):
            self.cards_table.setItem(row, 0, QTableWidgetItem(card.name))
            self.cards_table.setItem(row, 1, QTableWidgetItem(card.problem_type))
            self.cards_table.setItem(row, 2, QTableWidgetItem(card.created_at))
            metrics_txt = ", ".join([f"{k}:{v:.3f}" for k, v in card.metrics.items()])
            self.cards_table.setItem(row, 3, QTableWidgetItem(metrics_txt))
        self.cards_table.resizeColumnsToContents()
        # Run history table
        if hasattr(self, "run_table"):
            # populate model filter options
            model_names = sorted({run.get("model_name", "") for run in self.run_history if run.get("model_name")})
            self.run_filter_model.blockSignals(True)
            current_model = self.run_filter_model.currentText() if self.run_filter_model.count() else "All"
            self.run_filter_model.clear()
            self.run_filter_model.addItem("All")
            for name in model_names:
                self.run_filter_model.addItem(name)
            idx_match = self.run_filter_model.findText(current_model)
            if idx_match >= 0:
                self.run_filter_model.setCurrentIndex(idx_match)
            self.run_filter_model.blockSignals(False)

            filtered_runs = filter_run_history(
                self.run_history,
                problem=self.run_filter_problem.currentText(),
                model_filter=self.run_filter_model.currentText(),
            )
            self.run_table.setRowCount(len(filtered_runs))
            for idx, run in enumerate(filtered_runs):
                self.run_table.setItem(idx, 0, QTableWidgetItem(str(run.get("timestamp", ""))))
                self.run_table.setItem(idx, 1, QTableWidgetItem(run.get("model_name", "")))
                self.run_table.setItem(idx, 2, QTableWidgetItem(run.get("problem_type", "")))
                metrics = run.get("metrics", {}) or {}
                primary = choose_primary_metrics(run.get("domain", "Generic"), run.get("problem_type", ""))
                primary_name = primary[0] if primary else next(iter(metrics), "")
                primary_val = metrics.get(primary_name, "")
                self.run_table.setItem(idx, 3, QTableWidgetItem(f"{primary_name}: {primary_val}"))
            self.run_table.resizeColumnsToContents()

    def _selected_card(self):
        row = self.cards_table.currentRow()
        if row < 0 or row >= len(self.model_cards):
            return None
        return self.model_cards[row]

    def activate_selected_card(self):  # pragma: no cover - GUI
        card = self._selected_card()
        if not card:
            self.show_error("Select a model card to activate.")
            return
        if card.model_obj is None:
            self.show_error("This card has no model attached. Re-run analysis.")
            return
        self.model_manager.model = card.model_obj
        self.predict_button.setEnabled(True)
        self.statusBar().showMessage(f"Activated model: {card.name}", 5000)

    def export_selected_card(self):  # pragma: no cover - GUI
        card = self._selected_card()
        if not card:
            self.show_error("Select a model card to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export model card", "", "Markdown Files (*.md);;JSON Files (*.json)"
        )
        if not path:
            return
        try:
            export_model_card(card, path)
            self.statusBar().showMessage(f"Model card saved to {path}", 5000)
        except Exception:
            logger.exception("Model card export failed")
            self.show_error("Could not export model card. Please try another location.")

    def edit_card_notes(self):  # pragma: no cover - GUI
        card = self._selected_card()
        if not card:
            self.show_error("Select a model card first.")
            return
        text, ok = QInputDialog.getMultiLineText(self, "Edit notes", "Notes:", card.notes)
        if ok:
            card.notes = text
            self.statusBar().showMessage("Notes updated", 3000)



def main():
    app = QApplication(sys.argv)
    system = platform.system()
    if system == "Windows":
        icon_path = _resource_path("icons", "logo.ico")
    elif system == "Darwin":
        icon_path = _resource_path("icons", "logo.icns")
    else:
        icon_path = _resource_path("icons", "logo.png")
    app.setWindowIcon(QIcon(icon_path))

    window = MainApp()
    window.setWindowIcon(QIcon(icon_path))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
