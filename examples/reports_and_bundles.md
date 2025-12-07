# Reports, Bundles, and Notebook Export

This walkthrough shows how to export reporting artifacts after running Smart Analyze or a manual model.

## Steps

1) **Run an analysis**
   - Load a CSV (or demo), pick a target, click **Smart Analyze** (or run a model manually).

2) **Export “What I did”**
   - Click **Export 'What I did'…** to save a Markdown/Text summary:
     - Data summary, preprocessing steps, models tried, metrics, and notes.

3) **Export narrative report**
   - Click **Export Report** to generate a more verbose Markdown/HTML report with headings (Data, Methods, Results).

4) **Export analysis bundle**
   - Click **Export analysis bundle…** to write a zip containing:
     - `.mgui` project
     - Saved model(s)
     - Reports
     - Plots (if saved)

5) **Export notebook**
   - Click **Export notebook…** to generate a runnable `.ipynb` with:
     - Data loading
     - Preprocessing
     - Model training
     - Metrics and basic plots
   - See screenshot `docs/screenshots/notebook_export_cells.png` for expected structure.

6) **What’s inside**
   - “What I did”: concise, human-readable summary of the current run.
   - Narrative report: longer-form summary with headings.
   - Bundle: everything for reproducibility/sharing.
   - Notebook: code scaffold to rerun training/evaluation.
