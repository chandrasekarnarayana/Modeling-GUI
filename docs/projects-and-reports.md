# Projects & Reports

## .mgui Projects

- Save/load project files capturing:
  - Data path/hash
  - Domain preset
  - Target/feature selection
  - Preprocessing and AutoML settings
  - Trained model + metadata
  - Training snapshot (for drift)
  - Run history and summary (“What I did”)

## Analysis Bundles

- Export a zip bundle (project, reports, models, plots) via **Export analysis bundle…**.
- Contents typically include:
  - `.mgui` project
  - Serialized model(s)
  - Latest report text/Markdown
  - Saved plots (if any)
- Use this to hand off a reproducible snapshot to collaborators.

## Reports

- Export text/Markdown reports with summaries, metrics, and notes.
- Narrative report generation (plain language) outlines goal, data, preprocessing, best model, metrics, and next steps.
- See `examples/reports_and_bundles.md` for an end-to-end export walkthrough.

## Notebook Export

- Create a minimal reproducible notebook (.ipynb) to rerun training/prediction (requires nbformat).
- Includes cells for data loading, preprocessing steps, model training, metrics, and basic plots.
- Export via **Export notebook…** in the main toolbar.
- See `examples/reports_and_bundles.md` and screenshot `docs/screenshots/notebook_export_cells.png` for expected structure.

## Usage

1. Train or Smart Analyze.
2. Save project (.mgui) and optionally export bundle/report/notebook.
3. Reload project later to continue, run batch predictions, or compare models.

See walkthroughs:
- `examples/batch_prediction_basic.md` (reusing saved models + drift warnings)
- `examples/forecasting_basic.md` (adding forecasts to projects)
- `examples/feature_importance_basic.md` (reports with explainability).

## Model cards & history

- View model cards in the **Model cards** tab:
  - Lists model name, problem, created time, and metrics.
  - Filtering: filter run history by problem type or model name.
- Export a model card:
  - Select a model card, click **Export model card…**, choose Markdown or JSON.
  - Includes metadata, metrics, data summary, and notes.
- Version history:
  - Run history table shows past runs; use filters to focus on regression/classification/forecasting or specific models.
  - Activate a past model from the table to reuse it.
