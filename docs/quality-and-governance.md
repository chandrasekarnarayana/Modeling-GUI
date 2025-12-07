# Quality & Governance

## Data Quality Panel

- Runs checks on load/target selection:
  - Missing target values
  - Non-numeric features
  - Potential leakage (target in features)
  - Class imbalance
  - Type mismatches (numeric stored as text)
- Issues are listed with severity and suggestions; coach bar summarizes high-level warnings.

## Troubleshooting

- When model training fails, suggestions map common errors (missing target, non-numeric data, shape issues) to friendly guidance.

## Drift Monitoring

- Training snapshot stored in projects (.mgui): per-feature stats and row count.
- During batch prediction, new data is compared to the snapshot; drift warnings appear in coach bar and logs when moderate/strong drift is detected.
- The **Data Drift** tab lists per-feature drift issues (none/moderate/high) with short suggestions.

## Model Cards (lightweight)

- Each trained model can be associated with a model card (ID, name, date, domain, metrics, notes, explainability summary).
- Cards can be viewed/edited in Expert mode via the **Model cards / History** tab:
  - Select a card → view metrics and creation time.
  - Edit notes inline.
  - Set a card’s model as the active model for predictions/explainability.
- Export is planned/experimental; if visible, use the tab controls to export.

### How to use in the GUI

1. Train a model (Smart Analyze, manual run, or tuning).
2. Open **Model cards** tab → select the new card → add notes as needed.
3. For drift, run **Batch predict** with a saved model; the **Data Drift** tab updates automatically and coach bar will warn on strong drift.
4. Save a project to persist snapshots and cards (.mgui).

## PIN (guardrails)

- Optional simple PIN can gate Expert tweaks (if enabled) to avoid unintended advanced changes.

See also: `examples/batch_prediction_basic.md` for drift warnings, `examples/feature_importance_basic.md` for explainability/model card context.
