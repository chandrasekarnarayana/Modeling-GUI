# Smart Analyze (AutoML) Quickstart

This walkthrough shows how to run Smart Analyze/AutoML end-to-end in Modeling-GUI and read the results.

## Steps

1) **Launch**
   - Run `run_modeling_gui` (or `python -m modeling_gui`).

2) **Load data**
   - Click **Load Demo Dataset** (uses `modeling_gui/data/demo_quickstart.csv`) or load your own CSV.

3) **Choose what to predict**
   - Set **Target column** to `target` (regression) or `target_class` (classification).
   - Leave features preselected (or adjust if needed).

4) **Smart Analyze**
   - Click **Smart Analyze**.
   - The app detects problem type (regression vs classification; time-series when a date column is present) and runs AutoML (FLAML + fallbacks).

5) **Inspect results**
   - **Summary tab**: primary metrics for the detected problem type.
   - **Leaderboard tab**: candidate models, metrics, train time; you can set a different model as active.
   - **What I did**: human-readable summary of preprocessing, models tried, and metrics.

6) **Screenshots**
   - Main run: `docs/screenshots/smart_analyze_run.png`
   - Leaderboard: `docs/screenshots/smart_analyze_leaderboard.png`
   - Tuning dialog: `docs/screenshots/smart_analyze_tuning.png`

7) **Refine in Expert mode**
   - Switch to **Expert** mode to:
     - Inspect/tweak the model AutoML chose.
     - Use **Tune…** (Fast/Balanced/Thorough) on supported models (RF/GB).
     - Re-run and compare in the leaderboard.

8) **Save & export**
   - Save a project to keep metrics and runs.
   - Export a report or save plots for sharing.
