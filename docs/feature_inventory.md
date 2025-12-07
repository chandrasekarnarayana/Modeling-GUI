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
| Feature importance plots | Yes | Yes | Yes (explainability/docs) | Yes (examples/feature_importance_basic.md) | Yes (test_explainability.py) | OK; export/screenshot polish still useful. |
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

Shipped work is captured in sections A–Q. This section tracks what is ahead to keep pushing toward a best-in-class no-code modeling experience.

### Near-term polish (next releases)

| Focus | Goal | Status | Notes |
|-------|------|--------|-------|
| Leaderboard & comparison UX | Cleaner comparisons, exportable charts, clearer coach hints | In progress | Builds on current leaderboard; tighten visuals and screenshots. |
| Preprocessing guidance | Surface type inference, missing/imbalance guardrails, clearer Expert toggles | Planned/partial | Backend inference exists; needs guided UI copy and wizarding. |
| Drift/report visuals | Richer drift and report screenshots/exports | Planned | Drift heuristics exist; add export/share affordances. |
| Demo & templates | Auto demo mode, guided tours, starter templates | Planned | Successor to Quick Win 1.11; keeps onboarding truly no-code. |

### Capability expansion (medium horizon)

| Focus | Goal | Status | Notes |
|-------|------|--------|-------|
| Additional models | Logistic Regression, SVM, XGBoost/LightGBM defaults | Planned | Expands classification breadth without overwhelming beginners. |
| Data loading breadth | Multi-file/partitioned CSV ingest and optional DB connectors | Planned | Start with multi-CSV merge plus profiling. |
| Plugin-like extensibility | Lite hooks for custom models, metrics, plots | Planned | Config-driven hooks first; marketplace later. |
| Batch/automation | Headless/CLI scoring and scheduled batch runs | Planned | Complements GUI with repeatable pipelines. |

### Big bets (longer-term)

| Focus | Goal | Status | Notes |
|-------|------|--------|-------|
| Workflow canvas | Drag-and-drop pipeline editing and saved recipes | Not started | Post-1.0 UX overhaul. |
| Cloud/remote runtime | Run heavy jobs remotely; keep GUI as control plane | Not started | Requires auth plus compute backends. |
| Monitoring dashboard | Production drift/quality/alerting views | Not started | Builds on existing drift checks. |
| Collaboration & sharing | Project sharing, comments, template gallery, plugin marketplace | Not started | Multi-user story. |

## Summary of Key Gaps

- **Implementation gaps**: More model coverage (LogReg/SVM/XGBoost/LightGBM), multi-file ingest/DB connectors, plugin hooks, headless/batch automation, and longer-term canvas/remote/monitoring/collaboration work.
- **GUI/UX gaps**: Leaderboard and drift/report polish, clearer preprocessing guidance, guided demos/templates, richer export/share affordances.
- **Docs & tests gaps**: Smart Analyze/ROC–PR/forecast/notebook screenshots refreshed; still add drift/model-card visuals and expand GUI/visual regression coverage alongside upcoming features.
- **Media**: Demo video ready at `docs/assets/demo_promo.mp4`; include in release assets and site links.
