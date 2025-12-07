# Curve fitting (Gaussian & Exponential) in Modeling-GUI

Fit peaks (Gaussian) or growth/decay (Exponential) without code.

## Scenario

- **Gaussian**: spectroscopy peak fitting or bell-shaped signal.
- **Exponential**: decay (half-life) or growth curves (e.g., population, sales adoption).

## Steps

1) Launch `run_modeling_gui` and load a CSV with numeric `x` and `y` columns (or the demo dataset).  
2) In Expert mode, select `x` as feature and `y` as target.  

### Gaussian fit

1) Choose **Gaussian Fitting** from the model list.  
2) Click **Run selected model**.  
3) The plot shows data points and the fitted Gaussian curve; note the peak center (mean) and width (sigma).  
4) Save the plot (use **Save last plot…**) if you want to share results.

### Exponential fit

1) Choose **Exponential Fitting**.  
2) Run; the fitted curve overlays the data.  
3) Interpret the growth/decay rate (b) and baseline/scale parameters.  

## Tips

- Use one feature at a time for these fits (the app takes the first selected feature).
- Outliers can distort fits; clean or filter data first.
- Use **Save last plot…** to export the curve overlay as PNG/PDF.
