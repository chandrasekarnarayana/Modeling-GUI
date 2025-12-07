# Development Checklist

## 1. Core UX & AutoML

### Smart Analyze + Basic/Expert
- [ ] `SmartAnalyzeController` or orchestration class.
  - [ ] Detect problem type from target (regression/classification; later: clustering/forecasting).
  - [ ] Call `automl.run_automl(df, x_cols, y_col, settings)`.
  - [ ] Handle fallback when AutoML extras are not installed.
- [ ] `Basic` mode in GUI:
  - [ ] Hide advanced model controls.
  - [ ] Show: data load, domain preset, target, **Smart Analyze**, results.
- [ ] `Expert` mode:
  - [ ] Expose existing model list (OLS, RF, GB, etc.).
  - [ ] Parameter dialogs for RF/GB/KMeans.
  - [ ] Preprocessing toggles (standardization, missing-value strategy).
  - [ ] Manual “Run Selected Model” action.

### AutoML backend (e.g. FLAML)
- [ ] Module `modeling_gui/automl.py`:
  - [ ] `detect_problem_type(df, y_col) -> str`.
  - [ ] `run_automl(df, x_cols, y_col, settings) -> AutoMLResult`.
  - [ ] Internally handle missing values and encodings.
  - [ ] Provide metrics + artifacts: best model, leaderboard, predicted values, residuals/confusion matrix data.
- [ ] Optional dependency handling:
  - [ ] Extras `modeling-gui[automl]`.
  - [ ] Graceful error messages if AutoML is unavailable.

## 2. Coach bar & domain presets

### Coach / status bar
- [ ] `modeling_gui/coach.py` with states: `DATA_UNLOADED`, `DATA_LOADED`, `TARGET_SELECTED`, `ANALYSIS_RUNNING`, `ANALYSIS_DONE`, `ERROR`.
- [ ] `CoachManager.update(state, extra=None)` returns user text.
- [ ] GUI integration updates coach bar on key events with friendly messages.

### Domain presets
- [ ] `modeling_gui/domain.py` with `DomainPreset` dataclass: name, default metrics, plots, wording hints.
- [ ] Presets for `Generic`, `Finance`, `Science`, `Business`.
- [ ] GUI domain selector; Smart Analyze uses preset for metrics, plots, hints.

## 3. Visualization & metrics

### Visualizations
- [ ] Residual plot for regression.
- [ ] Feature importance plots.
- [ ] Confusion matrix from precomputed data; optional ROC/PR in Expert mode.
- [ ] Plot APIs take arrays, not models.

### Metrics & split
- [ ] Train/test split helper (reproducible `random_state`).
- [ ] Regression: R², RMSE, MAE.
- [ ] Classification: accuracy, precision/recall/F1 (macro/micro).
- [ ] GUI controls: use_split, test_size.

## 4. Model & project persistence

### Model save/load
- [ ] `modeling_gui/persistence.py`: `save_model`, `load_model`, version warnings.

### Project files (`.mgui`)
- [ ] Schema: data reference, domain, target/X, preprocessing, AutoML settings, best model, metrics, “What I did”.
- [ ] Functions: `save_project`, `load_project`.
- [ ] GUI menu items with missing-data recovery.

## 5. “What I did” & reporting
- [ ] Structured summary (e.g., `AnalysisSummary`).
- [ ] GUI tab/panel rendering formatted text.
- [ ] `reporting.py` export (Markdown/text); “Export report…” button.

## 6. Error handling & robustness
- [ ] Wrap risky operations (file load, AutoML, training, plotting, save/load).
- [ ] Friendly dialogs + coach bar on errors.
- [ ] Logging to stderr or file with trace details.
