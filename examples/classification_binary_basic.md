# Binary Classification Quickstart (Churn-Style)

This walkthrough shows how to run a simple binary classification (e.g., churn vs. not) in Modeling-GUI using the bundled demo dataset.

## Steps

1) **Load data**
   - Click **Load Demo Dataset** (uses `modeling_gui/data/demo_quickstart.csv`).

2) **Pick what to predict**
   - Set **Target column** to `target_class` (0/1).
   - Leave all other columns as features by default.

3) **Run Smart Analyze**
   - Click **Smart Analyze** (Basic mode) or **Run selected model** (Expert).
   - The app detects classification and trains candidate models.

4) **Review metrics**
   - Check **accuracy**, **balanced accuracy**, **macro F1**.
   - If probabilities are available, also see **ROC AUC** and **PR AUC** (highlighted per domain).

5) **Inspect plots**
   - Open the confusion matrix to see where errors occur.
   - Use the feature-importance / explainability tab to see top drivers.

6) **Threshold thinking (business context)**
   - Higher recall may matter more for churn; consider the confusion matrix and PR AUC.
   - If needed, switch to Expert mode and adjust models/thresholds manually.

7) **Save & reuse**
   - Save a project or model for batch prediction on new customers.
