# Clustering with KMeans in Modeling-GUI

Use KMeans to segment customers or group similar records without labels. This walkthrough uses a simple 2–4 feature CSV (demo or synthetic).

## Steps

1) Launch `run_modeling_gui` and load a CSV with numeric columns (e.g., spending, visits, age).  
2) In Expert mode, select your numeric feature columns.  
3) Choose **KMeans Clustering** from the model list.  
4) Click the KMeans dialog button to set **k** (e.g., 3–5).  
5) Run clustering. The app will fit KMeans and show cluster centers.  
6) (Optional) Plot clusters externally: save predictions/labels and scatter two features color-coded by cluster.

## Tips

- **Choosing k**: start small (3–5). Elbow/silhouette methods can help; try a few values.  
- **Interpret centroids**: each centroid is the mean of points in that cluster; compare feature averages to understand segments.  
- **Limitations**: scale sensitive (standardization helps), assumes roughly spherical clusters, not guaranteed to find global optimum.  
- **Data types**: KMeans requires numeric features; non-numeric columns are dropped with a warning.
