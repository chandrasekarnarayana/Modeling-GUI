# Feature importance

Goal: view and interpret which inputs matter most.

Steps:
1) Train a model (tree-based or linear).  
2) Click **Show Feature Importance** or open the **Explain** tab and click **Show global importance plot**.  
3) Interpret large vs small bars; top 3 drivers are summarized in text.  
4) Use this to decide which features to focus on or drop; capture a screenshot for stakeholders.

Notes:
- Tree models expose `feature_importances_`; linear models use coefficient magnitudes; SHAP is used when installed.  
- See also `examples/explainability_basic.md` for linking to local explanations.
