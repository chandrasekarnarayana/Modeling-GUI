# Modeling-GUI Long-form Demo (6–8 minutes)

Target: recruiters, collaborators, and GitHub visitors. Tone: calm, clear, confident. Emphasize **no-code**, faster than manual programming, **explainability**, and **reproducible workflows**. Use the prepared dataset from `docs/scripts/demo_setup.py`.

## Chapter-by-chapter storyboard

For each chapter: timestamp, on-screen actions, narration (read verbatim or lightly adapted), and camera/mouse guidance.

### 0:00–0:20 – Hook
- **On-screen**: Title card fades to Modeling-GUI main window (dataset preloaded). Small overlay: “Modeling-GUI – No-code modelling for real-world data”.
- **Mouse/zoom**: Static; subtle zoom-in on the window.
- **Narration**: “Most people have data. Very few have the time or confidence to code full machine learning pipelines. Modeling-GUI is my attempt to fix that.”

### 0:20–1:00 – Intro & concept
- **On-screen**: Webcam PIP bottom-left (optional). GUI dimmed in background. Overlay: “Chandrasekar Subramani Narayana — Creator” and “Concept & user perspective: Arunaachalam Subramani Narayana”.
- **Mouse/zoom**: Keep cursor still; slow pan across the window.
- **Narration**: “I’m Chandrasekar, a biophysics researcher and AI practitioner. This tool was conceptualised with my brother Arunaachalam, who wanted serious modelling without writing code. Modeling-GUI keeps the power of Python libraries, but presents them in a no-code, faster-than-manual interface.”

### 1:00–2:00 – Quick UI tour
- **On-screen**: Main window empty project. Highlight menu bar, left project/data panel, central work area, bottom logs.
- **Mouse/zoom**: Glide across menu titles; hover left panel; circle central area; hover logs.
- **Narration**: “Across the top is a classic menu bar. On the left, the project and datasets. The center is where tables, plots, and models appear. Logs live at the bottom so you can see what happens under the hood. Tabs keep the workflow simple: Data, EDA, Models, Results.”

### 2:00–3:30 – Demo 1: load data + simple EDA
- **On-screen**: Click **Open Dataset**, pick `demo_data/california_housing.csv` (prepared by script), table appears. Switch to summary stats; show columns.
- **Mouse/zoom**: Slow scroll rows; drag horizontal scrollbar; click Summary tab.
- **Narration**: “Let’s load a concrete dataset — California housing. It opens straight into a table, so I can scan income, house age, rooms, and the target. With one click I get summary stats and missing-value checks.”

### 3:30–4:00 – Quick plot
- **On-screen**: Go to **EDA/Plots** tab. Choose Scatter, X=`MedInc`, Y=`MedHouseVal`. Generate plot.
- **Mouse/zoom**: Hover axes; move cursor through point cloud.
- **Narration**: “Here’s a quick scatter of median income versus house value. No code — just pick axes and click generate.”

### 4:00–4:50 – Model 1: Linear Regression
- **On-screen**: Switch to **Models** tab → **New Model** → choose **Regression → Linear Regression**. Target=`MedHouseVal`; Features=`MedInc`, `HouseAge`, `AveRooms`, `AveOccup`. Click Run. Metrics appear.
- **Mouse/zoom**: Smooth clicks; hover metrics table.
- **Narration**: “To fit a model, I select linear regression, pick the target, choose a few predictors, and hit Run. Modeling-GUI wires up the estimator for me and returns R-squared and RMSE in seconds.”

### 4:50–5:30 – Interpretation plots
- **On-screen**: Click Residual Plot; then Predicted vs Actual.
- **Mouse/zoom**: Trace along axes; briefly point to outliers or diagonal.
- **Narration**: “Residuals show whether we over- or under-predict. Predicted-versus-actual gives a fast visual check on fit quality. All of this is generated from the configuration panel, not from hand-written scripts.”

### 5:30–6:30 – Model 2: Tree-based / AutoML
- **On-screen**: **New Model** → **Random Forest Regression** (or AutoML). Target=`MedHouseVal`; same features. Optionally tweak `n_estimators` to 100. Run. Show improved metrics and feature importance plot.
- **Mouse/zoom**: Hover model list to show both runs; mouse over feature-importance bars.
- **Narration**: “Now a more flexible tree-based model. Same dataset, same target, minimal tweaks. Performance improves, and I immediately get a feature-importance view to explain which variables drive the predictions.”

### 6:30–6:50 – Export
- **On-screen**: Click **Export** → select Predictions CSV, Model file, Report → save to `results/demo_run/`.
- **Mouse/zoom**: Click checkboxes slowly; browse to folder; confirm export.
- **Narration**: “When I’m happy, I export predictions, the trained model, and a report into a reproducible folder that teammates can use without touching notebooks.”

### 6:50–7:20 – No-code vs code split view
- **On-screen**: OBS Scene 3 with split view. Left: GUI showing Random Forest config. Right: VS Code tab with equivalent Python script (pandas load, RandomForestRegressor, metrics).
- **Mouse/zoom**: Slide cursor from left to right once; rest at center.
- **Narration**: “Here’s what the same workflow looks like in raw Python: imports, data loading, cleaning, estimator setup, metrics, plotting. Modeling-GUI doesn’t replace Python — it packages these steps so non-coders can drive them, while you keep full control of the stack.”

### 7:20–7:40 – Closing & CTA
- **On-screen**: Back to GUI. Open **Help → About** showing credits for Chandrasekar and Arunaachalam. Close dialog; slight zoom on app chrome; fade out to GitHub URL.
- **Mouse/zoom**: Hover the credit line briefly; move to top-right; fade.
- **Narration**: “Modeling-GUI is about serious modelling without forcing everyone to become a programmer. It’s built with a user perspective, inspired by many conversations with my brother Arunaachalam. If this resonates, explore the repository, open issues, or reach out to collaborate. Thanks for watching.”

## Narration script (continuous)
Keep sentences synchronized with the actions above; you can read this verbatim:

> “Most people have data. Very few have the time or confidence to code full machine learning pipelines. Modeling-GUI is my attempt to fix that.  
> I’m Chandrasekar, a biophysics researcher and AI practitioner. This tool was conceptualised with my brother Arunaachalam, who wanted serious modelling without writing code. Modeling-GUI keeps the power of Python libraries, but presents them in a no-code, faster-than-manual interface.  
> Across the top is a classic menu bar. On the left, the project and datasets. The center is where tables, plots, and models appear. Logs live at the bottom so you can see what happens under the hood. Tabs keep the workflow simple: Data, EDA, Models, Results.  
> Let’s load a concrete dataset — California housing. It opens straight into a table, so I can scan income, house age, rooms, and the target. With one click I get summary stats and missing-value checks.  
> Here’s a quick scatter of median income versus house value. No code — just pick axes and click generate.  
> To fit a model, I select linear regression, pick the target, choose a few predictors, and hit Run. Modeling-GUI wires up the estimator for me and returns R-squared and RMSE in seconds.  
> Residuals show whether we over- or under-predict. Predicted-versus-actual gives a fast visual check on fit quality. All of this is generated from the configuration panel, not from hand-written scripts.  
> Now a more flexible tree-based model. Same dataset, same target, minimal tweaks. Performance improves, and I immediately get a feature-importance view to explain which variables drive the predictions.  
> When I’m happy, I export predictions, the trained model, and a report into a reproducible folder that teammates can use without touching notebooks.  
> Here’s what the same workflow looks like in raw Python: imports, data loading, cleaning, estimator setup, metrics, plotting. Modeling-GUI doesn’t replace Python — it packages these steps so non-coders can drive them, while you keep full control of the stack.  
> Modeling-GUI is about serious modelling without forcing everyone to become a programmer. It’s built with a user perspective, inspired by many conversations with my brother Arunaachalam. If this resonates, explore the repository, open issues, or reach out to collaborate. Thanks for watching.”

## Automation helper
- Run `python docs/scripts/demo_setup.py` to fetch the demo dataset and optionally launch Modeling-GUI with it preloaded.
- Use OBS scenes:  
  - Scene 1 (intro): display + webcam + title.  
  - Scene 2 (GUI focus): window capture of Modeling-GUI, small webcam.  
  - Scene 3 (split view): GUI left, VS Code right (for no-code vs code).

## Recording checklist
- Resolution 1920×1080, 30 fps; UI scaling so text is readable.  
- Turn on Do Not Disturb; close noisy apps.  
- Run `python docs/scripts/demo_setup.py` and keep `demo_data/california_housing.csv` handy.  
- Launch Modeling-GUI once before recording to warm caches; close and relaunch when rolling.  
- Record in segments: 0:00–3:30, 3:30–6:30, 6:30–7:40; redo only the segment if a mistake happens.  
- Keep mouse movement smooth and slow; pause briefly before speaking each line.  
- After recording: trim pauses, add low-volume background music if desired, and replace `docs/assets/demo_promo.mp4` with the final export before publishing the release and README links.
