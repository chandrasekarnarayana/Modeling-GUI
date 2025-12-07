# Explainability (global + local)

Goal: see which features drive predictions, globally and for a specific row.

Steps:
1) Load CSV → choose target → **Smart Analyze**.  
2) Open **Explain** tab.  
3) Click **Show global importance plot** to view SHAP/native importances and summary text.  
4) Set a row index and click **Explain selected row** to see local contributions (waterfall-style bar chart).  
5) Save a report or screenshot if needed.

Tips:
- Install `shap` for richer attributions; otherwise, feature_importances_ / coefficients are used.
- Use the same tab to copy the summary into reports.
- For classification with probabilities, check the Summary tab for **ROC AUC** and **PR AUC** alongside accuracy/F1; confusion matrix + feature importance help interpret why classes are predicted.***
