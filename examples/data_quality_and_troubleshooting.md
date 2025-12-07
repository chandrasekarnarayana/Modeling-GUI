# Data Quality & Troubleshooting Walkthrough

This example shows how Modeling-GUI surfaces leakage and imbalance warnings and how to address them.

## Steps

1) **Load data with potential issues**
   - Use a CSV that includes:
     - An ID/label-like column (e.g., `customer_id` or a duplicate of the target).
     - A highly imbalanced target (e.g., 95% class 0, 5% class 1).

2) **Select target and features**
   - Pick the binary target (e.g., `churn_flag`).
   - Leave all columns selected initially to see warnings.

3) **Run data quality checks**
   - The Data Quality tab will flag:
     - Leakage: target-like or highly correlated columns.
     - Imbalance: large max/min class ratio.
     - Missing target values if present.
   - The coach bar will show a brief warning.

4) **Fix the issues**
   - Remove ID/target-like columns from X to avoid leakage.
   - Consider class weights or resampling for imbalance.
   - Enable missing-value handling in preprocessing if needed.

5) **Re-run**
   - After cleaning selections, rerun Smart Analyze or a chosen model.
   - Verify that warnings reduce or disappear in the Data Quality tab.
