# Feature Inventory

This document inventories Modeling-GUI features and their status across code, GUI, docs, examples, and tests. Legend: Yes / No / Partial. Notes highlight remaining TODOs or caveats.

## A. Statistical Regression Models

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| OLS | Yes | Yes | Yes (README/docs) | Demo CSV implicit | Yes (test_models.py) | Consider explicit notebook example. |
| WLS | Yes | Yes | Yes (docs/user-guide) | Yes (regression_advanced_ls.md) | Yes (test_advanced_ls.py) | OK. |
| GLS | Yes | Yes | Yes (docs/user-guide) | Yes (regression_advanced_ls.md) | Yes (test_advanced_ls.py) | OK. |
| Recursive LS | Yes | Yes | Yes (docs/user-guide) | Yes (regression_advanced_ls.md) | Yes (test_advanced_ls.py) | OK. |
| Rolling LS | Yes | Yes | Yes (docs/user-guide) | Yes (regression_advanced_ls.md) | Yes (test_advanced_ls.py) | Window doc in example. |
| RLM | Yes | Yes | Yes (docs/user-guide) | Yes (regression_advanced_ls.md) | Yes (test_advanced_ls.py) | OK. |

## B. Machine Learning Models

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| RandomForestRegressor | Yes | Yes | Yes | Yes (feature_importance_basic) | Yes (test_models.py, test_tuning_and_leaderboard.py) | OK. |
| RandomForestClassifier | Yes | Yes | Yes | Yes (classification_binary_basic.md; classification_business_roc_pr.md) | Yes (test_tuning_and_leaderboard.py) | Business ROC/PR example available. |
| GradientBoostingRegressor | Yes | Yes | Yes | Yes (regression_gradient_boosting_basic.md) | Yes (test_gradient_boosting.py) | OK. |
| GradientBoostingClassifier | Yes | Yes | Yes | Yes (classification_multiclass_basic.md; classification_business_roc_pr.md) | Yes (test_gradient_boosting.py) | Business ROC/PR example available. |
| LogisticRegression/SVM/etc. | No | No | No | No | No | Not implemented. |

## C. Clustering

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| KMeans | Yes | Yes | Yes (user-guide classical models) | Yes (examples/clustering_kmeans_basic.md) | Yes (test_clustering.py) | Coach hint added; consider silhouette/elbow note in GUI. |

## D. Time-Series / Forecasting

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Naive forecast | Yes | Yes | Yes | Yes (examples/forecasting_basic.md) | Yes (test_forecasting.py) | OK. |
| ETS/ExponentialSmoothing | Yes | Yes | Yes | Yes (examples/forecasting_basic.md) | Yes (test_forecasting.py) | Requires statsmodels. |
| ARIMA | Yes | Yes (selectable) | Yes (docs/forecasting) | Yes (examples/forecasting_basic.md) | Yes (test_forecasting.py) | Warnings on tiny series are acceptable. |
| Prophet | Yes | Yes | Yes (docs/forecasting) | Yes (examples/forecasting_basic.md) | Yes (test_forecasting_advanced.py) | Holidays limited to preset country list; no custom regressors yet. |
| Time-aware splits/backtesting | Yes (rolling backtest) | Yes (checkbox) | Yes (docs/forecasting) | Partial (advanced note in forecasting_basic.md) | Yes (test_forecasting_advanced.py) | Multi-fold backtest simple; could visualize more. |

## E. Advanced Fitting

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Gaussian fit | Yes | Yes | Yes (user-guide curve fitting) | Yes (examples/curve_fitting_gaussian_exponential.md) | Yes (test_curve_fitting.py) | Minor optimize warnings acceptable. |
| Exponential fit | Yes | Yes | Yes (user-guide curve fitting) | Yes (examples/curve_fitting_gaussian_exponential.md) | Yes (test_curve_fitting.py) | Minor optimize warnings acceptable. |

## F. Preprocessing & Encoding

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Missing value handling | Yes | Yes | Yes | Implicit | Yes (test_preprocessing.py) | OK. |
| Numeric vs categorical inference | Yes | Partial | Yes | Yes (examples/preprocessing_dates_and_categoricals.md) | Yes (test_preprocessing.py) | UI exposure is light; backend defaulted. |
| One-hot encoding | Yes (default) | Yes (Expert dropdown) | Yes | Yes (examples/preprocessing_dates_and_categoricals.md) | Yes (test_preprocessing_encoding.py) | Default strategy in Basic mode. |
| Target encoding | Yes (optional) | Yes (Expert dropdown) | Yes | Yes (examples/preprocessing_dates_and_categoricals.md) | Yes (test_preprocessing_encoding.py) | Advanced; unseen categories use global mean. |
| Date parsing + derived features | Yes | Partial | Yes (forecasting/user-guide) | Yes (examples/preprocessing_dates_and_categoricals.md) | Yes (test_preprocessing.py) | Add clearer GUI toggle description. |
| Scaling/standardization | Yes | Yes | Yes | Implicit | Yes | OK. |

## G. Metrics & Evaluation

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Regression metrics (R², RMSE, MAE, MAPE) | Yes | Yes | Yes (user-guide) | Implicit | Yes (test_metrics.py) | OK. |
| Classification metrics (accuracy, balanced acc, F1) | Yes | Yes | Yes | Implicit | Yes (test_metrics.py) | OK. |
| ROC AUC / PR AUC | Yes (when proba) | Yes (metrics table) | Yes (user-guide) | Yes (classification_business_roc_pr.md) | Yes (test_metrics.py, test_metrics_ui_integration.py) | Business example highlights ROC/PR. |
| Domain-specific highlighting | Yes | Yes | Yes (user-guide) | Yes (recipes.md; classification_business_roc_pr.md; forecasting_basic.md) | Yes (test_metrics_ui_integration.py) | Domain presets drive primary metrics and coach text. |

## H. Visualizations

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Regression plots | Yes | Yes | Yes (visualizations example) | Yes (visualizations_regression_and_residuals.md) | Yes (test_visualizations.py) | OK. |
| Residual plots | Yes | Yes | Yes (visualizations example) | Yes (visualizations_regression_and_residuals.md) | Yes (test_visualizations.py) | OK. |
| Confusion matrices | Yes | Yes | Yes | Yes (visualizations_confusion_and_trees.md) | Yes (test_visualizations.py) | OK. |
| Tree diagrams | Yes | Yes | Partial | Yes (visualizations_confusion_and_trees.md) | Yes (test_visualizations.py) | Graphviz must be installed. |
| Feature importance plots | Yes | Yes | Yes (explainability/docs) | Yes (examples/feature_importance_basic.md) | No | Add plot test (optional). |
| Gaussian/Exponential fits | Yes | Yes | Yes | Yes (examples/curve_fitting_gaussian_exponential.md) | Yes (test_curve_fitting.py) | Add screenshot in docs. |
| Forecast plots | Yes | Yes | Yes | Yes (examples/forecasting_basic.md) | Yes (test_visualizations.py) | OK. |
| Partial dependence | Yes | Yes (Scenario tab button) | Yes (docs/explainability) | Yes (examples/scenario_testing_basic.md) | Yes (test_explainability.py) | OK. |

## I. AutoML / Smart Analyze

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Problem type detection | Yes | Yes | Yes | Yes (automl_smart_analyze_quickstart.md) | Partial (test_automl.py covers) | Doc + example now present. |
| AutoML backend (FLAML optional + fallbacks) | Yes (with fallback models) | Yes | Yes | Yes (automl_smart_analyze_quickstart.md) | Yes (test_automl.py/test_automl_optional.py) | OK. |
| Leaderboard of candidate models | Yes | Yes (tab + chart) | Yes | Yes (automl_smart_analyze_quickstart.md) | Yes (test_tuning_and_leaderboard.py) | Screenshot referenced in docs. |
| Hyperparameter tuning presets | Yes | Yes (Tune… button) | Yes | Yes (automl_smart_analyze_quickstart.md) | Yes (test_tuning_and_leaderboard.py) | GUI screenshot referenced. |

## J. Explainability

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Global explanations (SHAP or fallback) | Yes | Yes (Explain tab) | Yes | Yes (examples/explainability_basic.md) | Yes (test_explain.py, test_explainability.py) | SHAP optional. |
| Local explanations | Yes | Yes (row selector) | Yes | Yes (examples/explainability_basic.md) | Yes | OK. |
| Partial dependence | Yes | Yes (Scenario tab) | Yes | Yes (examples/scenario_testing_basic.md) | Yes | OK. |

## K. Scenario / What-if Testing

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Sliders/inputs for key features | Yes | Yes (Scenario tab) | Yes | Yes (examples/scenario_testing_basic.md) | Yes (test_scenario.py) | OK. |
| Live recomputation of predictions | Yes | Yes | Yes | Yes | Yes | OK. |

## L. Data Quality & Troubleshooting

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Missing target checks | Yes | Yes (Data quality tab) | Yes | Yes (data_quality_and_troubleshooting.md) | Partial (test_data_quality.py) | OK. |
| Non-numeric feature checks | Yes | Yes | Yes | No | Yes | OK. |
| Leakage detection | Yes | Yes | Yes (troubleshooting.md) | Yes (data_quality_and_troubleshooting.md) | Yes (test_data_quality.py) | Heuristics improved (names + correlation). |
| Class imbalance warnings | Yes (heuristic) | Yes | Yes (troubleshooting.md) | Yes (data_quality_and_troubleshooting.md) | Yes (test_data_quality.py) | Threshold based on max/min ratio. |
| Exception-based suggestions | Yes | Yes | Yes (troubleshooting.md) | Yes (data_quality_and_troubleshooting.md) | Partial | Friendly mappings added; expand coverage in future. |

## M. Drift / Monitoring

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Training snapshot | Yes | Yes (projects) | Yes | Partial (batch prediction example) | Yes (test_drift.py) | OK. |
| Drift comparison on new data | Yes | Yes (Data Drift tab + coach bar) | Yes | Partial (examples/batch_prediction_basic.md) | Yes | Could add UI screenshot. |

## N. Persistence & Projects

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Model save/load | Yes | Yes | Yes | Partial | Yes (test_persistence.py) | OK. |
| Project .mgui save/load | Yes | Yes | Yes (projects docs) | Partial | Yes (test_version_history_and_notebook.py) | Ensure legacy projects migrate. |
| Model cards | Yes | Yes (Model cards tab) | Yes | Yes (model_cards_and_history.md) | Yes (test_model_cards.py) | Export Markdown/JSON supported. |
| Version history | Yes (run_history persisted) | Yes (history table) | Yes (projects-and-reports.md) | Yes (model_cards_and_history.md) | Yes (test_version_history_and_notebook.py) | Filtering by problem/model supported. |

## O. Reports & Exports

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| “What I did” summary | Yes | Yes | Yes | Yes (reports_and_bundles.md) | Partial | Export button available. |
| Narrative report | Yes | Yes (Export report) | Yes | Yes (reports_and_bundles.md) | Partial | Uses narrative builder; add more tests. |
| Analysis bundles (zip) | Yes | Yes | Yes | Yes (reports_and_bundles.md) | Yes (test_bundles.py) | OK. |
| Notebook export | Yes (richer cells) | Yes | Yes | Yes (reports_and_bundles.md) | Yes (test_version_history_and_notebook.py) | Exports runnable notebook with data/preprocessing/training/plots. |
| Batch prediction & CSV export | Yes | Yes | Yes | Yes (examples/batch_prediction_basic.md) | Yes (test_batch_prediction.py) | OK; add drift screenshot. |

## P. GUI & UX

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Basic vs Expert mode | Yes | Yes | Yes | Partial | No | OK. |
| Domain presets | Yes | Yes | Yes | Yes (recipes.md + linked examples) | No | Coach bar and metrics highlighting aligned with domain. |
| Coach/status bar | Yes | Yes | Yes | Partial | No | OK. |
| Keyboard shortcuts | Yes | Yes (Help → Keyboard Shortcuts…) | Yes (shortcuts-and-installation.md) | Partial | Yes (test_gui_smoke.py) | Cheat-sheet dialog added; consider in-app quick ref link. |
| Icons & shortcut installer | Yes | Yes | Yes (shortcuts-and-installation.md) | No | No | First-run tip added; optional GUI link. |

## Q. Docs, Examples, Tests

| Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Gaps |
|---------|--------------------|----------------|------------|-------------------|-------|--------------|
| Docs site (MkDocs) | Yes | N/A | Yes (populated pages) | N/A | N/A | Keep screenshots current. |
| Examples (beyond demo) | Yes | N/A | Yes (examples/*.md) | Yes | N/A | Add notebook export sample/screenshot. |
| Tests | Broad coverage | N/A | N/A | N/A | Yes (pytest suite) | UI/visual regression still light. |

## R. Roadmap & Strategic Features

The following table maps high-level roadmap items to current status.

| Roadmap Group | Feature | Implemented (code) | Exposed in GUI | Documented | Example available | Tests | Notes / Status |
|---------------|---------|--------------------|----------------|------------|-------------------|-------|----------------|
| Quick Win 1.1 | Smart Analyze + GUI core polish | Yes | Yes | Yes | Yes (quickstart/demo) | Partial | Core flow solid; coach hints still evolving (see I, P). |
| Quick Win 1.2 | GUI Enhancements (Basic/Expert clarity) | Yes | Yes | Yes | Partial | No | Mode toggle/tooltips in place; more examples/screenshots welcome (see P). |
| Quick Win 1.3 | Hyperparameter Tuning (basic) | Yes | Yes | Yes | Yes (automl_smart_analyze_quickstart.md) | Yes (test_tuning_and_leaderboard.py) | RF/GB presets with dialog (see I). |
| Quick Win 1.4 | Leaderboard Visual Upgrade | Partial | Yes | Yes | Yes | Partial | Chart exists; screenshots in docs; polish still desired (see I, H). |
| Quick Win 1.5 | Explainability – Global view | Yes | Yes | Yes | Yes | Yes | SHAP optional; fallback importances (see J). |
| Quick Win 1.6 | Scenario Testing – MVP sliders | Yes | Yes | Yes | Yes | Yes | Fully wired (see K). |
| Quick Win 1.7 | Forecasting UI (minimal but usable) | Yes | Yes | Yes | Yes | Yes | Prophet/ARIMA/backtest available (see D). |
| Quick Win 1.8 | Model Cards – Basic version | Yes | Yes | Yes | Yes (model_cards_and_history.md) | Yes | Export available; filtering added (see N). |
| Quick Win 1.9 | Drift Detection – Basic version | Yes | Yes | Yes | Partial | Yes | Simple heuristics; richer views desirable (see M). |
| Quick Win 1.10 | Documentation Essentials | Yes | N/A | Yes | N/A | N/A | Docs site + README; keep screenshots current (see Q). |
| Quick Win 1.11 | Auto Demo Mode + GIF recorder helper | No | No | No | No | No | Not started; future nicety. |
| Medium 2.1 | Batch Prediction + Export | Yes | Yes | Yes | Yes | Yes | Available via batch predict dialog (see O). |
| Medium 2.2 | Explainability – Local view (per-row) | Yes | Yes | Yes | Yes | Yes | Row selector + plots (see J). |
| Medium 2.3 | Partial Dependence Plots | Yes | Yes | Yes | Yes | Yes | Via Scenario tab (see H, J, K). |
| Medium 2.4 | Advanced Forecasting (Auto-ARIMA, Prophet, backtesting) | Yes | Yes | Yes | Yes | Yes | Prophet core; ARIMA/ETS; backtest metrics (see D). |
| Medium 2.5 | Improved Preprocessing | Partial | Partial | Yes | Yes | Yes | Target encoding advanced; UI toggle clarity pending (see F). |
| Medium 2.6 | Model Persistence / Bundles / Reproducibility | Yes | Yes | Yes | Yes (model_cards_and_history.md) | Yes | Projects, bundles, notebook export; more examples welcome (see N, O). |
| Medium 2.7 | Plugin-like Architecture (Lite) | No | No | No | No | No | Planned; not started. |
| Medium 2.8 | Multiple-file Data Loading (Optional) | No | No | No | No | No | Planned; not started. |
| Medium 2.9 | More ML Models (LogReg, SVM, XGBoost/LightGBM) | No | No | No | No | No | Early planning; current set limited (see B). |
| Long 3.1 | Interactive Workflow Canvas (Drag-and-Drop) | No | No | No | No | No | Long-term (post v1.0). |
| Long 3.2 | Database Connectors | No | No | No | No | No | Long-term. |
| Long 3.3 | Cloud Execution / Remote Runtime | No | No | No | No | No | Long-term. |
| Long 3.4 | Model Monitoring Dashboard | No | No | No | No | No | Long-term. |
| Long 3.5 | Plugin Marketplace | No | No | No | No | No | Long-term. |
| Long 3.6 | Collaborative Editing | No | No | No | No | No | Long-term. |

## Summary of Key Gaps

- **Implementation gaps**: Troubleshooting heuristics could be expanded further; medium items not started: plugin-lite (2.7), multi-file loading (2.8), additional ML models (2.9). Long-term (3.1–3.6) intentionally unstarted.
- **GUI wiring / UX gaps**: Drift/model-card views could surface richer summaries/filters; feature-importance/tree export affordances could be surfaced further.
- **Documentation gaps**: Screenshots for leaderboard/backtesting/scenario testing/drift/model cards; optional deeper SHAP/visual export notes.
- **Example gaps**: Optional feature-importance export/tree diagram example; richer notebook export screenshot.
- **Test gaps**: GUI/integration/visual regression tests still light; SHAP path only lightly exercised when installed.
