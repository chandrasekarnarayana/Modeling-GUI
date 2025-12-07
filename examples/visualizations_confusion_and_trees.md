# Confusion Matrices & Tree Diagrams

This walkthrough shows how to view confusion matrices and tree diagrams for classification models in Modeling-GUI.

## Steps

1) **Load data**
   - Use a binary or multiclass dataset (demo CSV has `target_class` for binary).

2) **Run a classification model**
   - Basic mode: **Smart Analyze** will pick a classifier.
   - Expert mode: choose **Random Forest** or **Gradient Boosting (classification)** and click **Run selected model**.

3) **Confusion matrix**
   - After the run, open the metrics/plots area to view the confusion matrix.
   - Shows counts of true vs predicted classes; off-diagonal cells highlight misclassifications.
   - Use **Save plot…** to export as PNG/PDF.

4) **Tree diagram (for tree-based models)**
   - Select **Show tree diagram** (if available) for Random Forest/GB to inspect a representative tree.
   - Useful for understanding splits and feature thresholds.
   - Requires Graphviz installed (see docs).

5) **Interpretation**
   - Confusion matrix: examine false positives/negatives; for business use, pair with ROC/PR metrics (see `examples/classification_business_roc_pr.md`).
   - Tree diagram: use to explain decision paths and key features; not all ensembles expose a single tree, so diagrams are illustrative.
