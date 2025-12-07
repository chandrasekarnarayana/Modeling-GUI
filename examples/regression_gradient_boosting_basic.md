# Gradient Boosting Regression (Basic)

This walkthrough shows how to train a Gradient Boosting regressor in Modeling-GUI using a simple numeric dataset (the bundled demo CSV or any small regression CSV).

## Steps

1) **Load data**
   - Click **Load Demo Dataset** or load your own CSV with numeric features and a numeric target.

2) **Pick target and features**
   - Set **Target column** to your numeric target (e.g., `target` in the demo).
   - Leave all numeric feature columns selected (or pick a subset).

3) **Switch to Expert mode**
   - In the **Model (Expert mode)** dropdown, choose **Gradient Boosting**.

4) **Adjust key parameters**
   - Open the Gradient Boosting dialog if available, or set basic params in the GUI:
     - `n_estimators` (e.g., 100–200)
     - `learning_rate` (e.g., 0.05–0.1)
     - `max_depth` (small depth like 3 works well for tabular data)

5) **Run the model**
   - Click **Run selected model**.
   - Check metrics: **R², RMSE, MAE, MAPE**.
   - Open regression/residual plots to inspect fit quality.

6) **Compare vs Random Forest**
   - Gradient Boosting often performs better on smaller/tabular datasets with moderate complexity.
   - Random Forest can be more robust to noise but may need more trees for similar accuracy.

7) **Save or tune**
   - Use the **Tune…** button for a quick preset search (fast/balanced/thorough).
   - Save the model or project if you want to reuse or compare in the leaderboard.
