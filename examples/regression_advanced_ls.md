# Advanced Linear Models in Modeling-GUI

This walkthrough shows when and how to use the advanced linear models beyond OLS, using either the bundled demo CSV (`modeling_gui/data/demo_quickstart.csv`) or a small CSV with numeric features/target and an optional weights column.

## 1. Setup and loading data

1. Launch the app (`run_modeling_gui`).
2. Click **Load CSV** (or **Load Demo Dataset**).
3. Choose your target column and select the numeric feature columns in Expert mode.

## 2. Weighted Least Squares (WLS)

- **When**: Heteroscedastic errors (variance differs across observations) and you have a weights column.
- **How**:
  1. Include the weights column in X selection.
  2. In Expert mode, choose **WLS** from the model list.
  3. Run the model; the app uses the selected weights column.
- **Interpretation**: Coefficients prioritize rows with higher weights; compare to OLS for differences.

## 3. Generalized Least Squares (GLS)

- **When**: Known correlation/variance structure in errors.
- **How**:
  1. Select your features/target.
  2. Choose **GLS** in the model list.
  3. Run the model (uses a simple sigma structure internally).
- **Interpretation**: Accounts for correlated errors; expect different standard errors vs OLS.

## 4. Recursive Least Squares

- **When**: Data arrives over time and you want online/iterative updates.
- **How**:
  1. Select features/target.
  2. Choose **Recursive LS**.
  3. Run; parameters update as if data streamed in.
- **Interpretation**: Similar to OLS on full data, but suited for streaming/online contexts.

## 5. Rolling Least Squares

- **When**: Relationships change over time; use a moving window.
- **How**:
  1. Ensure data is time-ordered.
  2. Select features/target.
  3. Choose **Rolling Least Squares** (uses a default window).
  4. Run; outputs parameters per window.
- **Interpretation**: Parameters per window show how coefficients evolve. No train/test split here; treat it as descriptive over time.

## 6. Robust Linear Model (RLM)

- **When**: Outliers may distort OLS.
- **How**:
  1. Select features/target.
  2. Choose **RLM**.
  3. Run; uses a robust M-estimator under the hood.
- **Interpretation**: Coefficients should move less with outliers than OLS; compare residual patterns.
