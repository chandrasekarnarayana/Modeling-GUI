# Business Classification with Gradient Boosting (ROC/PR Focus)

Use this example to run a business-style binary classification (e.g., churn/default) and focus on ROC/PR metrics.

## Steps

1) **Load data**
   - Use the demo CSV’s binary target (`target_class`) or load your own churn/default dataset.

2) **Select target and features**
   - Set **Target column** to the binary flag (0/1 or Yes/No).
   - Keep relevant numeric/categorical features selected.

3) **Choose model**
   - In Expert mode, pick **Gradient Boosting** (classification).
   - Alternatively, click **Smart Analyze** and let AutoML pick; confirm Gradient Boosting if selected.

4) **Run and review metrics**
   - Metrics panel will show **accuracy**, **balanced accuracy**, **macro F1**, and—when probabilities are available—**ROC AUC** and **PR AUC**.
   - Open the **confusion matrix** to see false positives/negatives.

5) **Interpretation for business decisions**
   - **ROC AUC**: good for overall ranking of risk; higher is better.
   - **PR AUC**: valuable for imbalanced problems; focus on precision/recall trade-off.
   - Use the confusion matrix to balance false positives (unnecessary retention spend) vs false negatives (lost customers).

6) **Explainability & what-if**
   - Open the **Explain** tab for global/local drivers.
   - Use **Scenario testing** to adjust key features and see how predicted churn/default changes.

7) **Save & compare**
   - Save the run in the leaderboard, export reports, or save a project for later batch prediction on new customers.
