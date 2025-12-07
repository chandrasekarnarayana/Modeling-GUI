# Multiclass Classification Quickstart

This walkthrough shows how to run a simple 3-class classification in Modeling-GUI. You can adapt any CSV with a categorical target that has three or more classes.

## Steps

1) **Load data**
   - Open your CSV with a multiclass target (e.g., `class_A/B/C`). For a quick test, extend the demo CSV by mapping `target_class` into three buckets (0/1/2) before loading.

2) **Pick what to predict**
   - Set **Target column** to your multiclass column (e.g., `label`).
   - Keep all other columns as features unless you want to filter them.

3) **Run Smart Analyze**
   - Click **Smart Analyze** (Basic) or **Run selected model** (Expert).
   - Candidate models run; the best model is highlighted in the leaderboard.

4) **Review metrics**
   - Focus on **macro F1** (treats all classes equally) and **micro F1** (weighted by support).
   - Balanced accuracy is also useful for uneven class sizes.

5) **Inspect plots**
   - Confusion matrix shows per-class mistakes.
   - Feature importance / explainability tab shows top drivers across classes.

6) **Interpretation tips**
   - If one class is rare, macro F1 will reveal poor performance even when accuracy looks high.
   - Try Gradient Boosting or Random Forest in Expert mode if default isn’t satisfactory.

7) **Save & compare**
   - Save the run to compare against other models in the leaderboard.
