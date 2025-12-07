# Troubleshooting Guide

Common issues and how to fix them in Modeling-GUI.

## My model won’t run
- Check that the target column exists and matches your selection.
- If you see “could not convert string to float”: some features/target contain text. Enable categorical encoding or clean the data.
- Missing values: use preprocessing to drop or impute.
- Target in features: remove the target column from X to avoid leakage.

## Data leakage warnings
- The Data Quality tab flags:
  - Target present in features.
  - Columns named like `id`, `target`, `label`.
  - Features almost identical to the target (very high correlation).
- Fix: remove or transform the flagged columns before training.

## Class imbalance warnings
- Triggered when one class dominates (>10x another).
- Options:
  - Use class weights (model option if available).
  - Resample (over/under) your data offline.
  - Adjust decision threshold (business context: balance false positives vs false negatives).

## Where to see warnings
- Data Quality tab shows issues and suggestions.
- Coach bar gives a short hint when warnings are present.
- See `examples/data_quality_and_troubleshooting.md` for a guided walkthrough.
