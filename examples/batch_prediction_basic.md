# Batch prediction (saved model)

Goal: apply a trained/saved model to a new CSV and export predictions.

Steps:
1) Train or load a model/project (.mgui).  
2) Click **Batch predict with saved model…** and choose a new CSV with matching feature columns.  
3) (Optional) Handle missing values per the selected strategy; scaling is applied if saved.  
4) Export predictions to CSV when prompted; output includes original columns + `prediction` + `model_id` + timestamp.  
5) If a training snapshot exists, drift issues will show in the **Data Drift** tab and coach bar.

Tips:
- Ensure column names match the saved model’s feature list.
- Use Save Project before batch prediction to keep snapshots for drift comparisons.***
