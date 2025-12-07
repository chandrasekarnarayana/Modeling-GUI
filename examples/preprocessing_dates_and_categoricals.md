# Preprocessing dates and categoricals

This walkthrough shows how Modeling-GUI handles date parsing, date-derived features, and categorical encoding (one-hot vs target encoding).

## Steps

1) Load a CSV with a date column (e.g., `date`) and a categorical column (e.g., `category`), plus numeric features.  
2) In Expert mode, select your target and feature columns.  
3) Enable date parsing/feature extraction if available (date features such as year, month, weekday).  
4) Encoding:
   - Default: **One-hot** (safe, works for any task).  
   - Advanced: **Target encoding** (requires a target; maps categories to mean target, unseen categories → global mean).
5) Run the model (Smart Analyze or manual) and inspect the transformed features in reports/plots.

## Notes

- One-hot expands categorical columns into binary indicators; can increase dimensionality.
- Target encoding is supervised; use when categories are many/high-cardinality and you want a compact numeric representation.
- Unseen categories map to the global mean target to avoid errors on new data.
- Date features help models use seasonality/temporal patterns; ensure the date column is parsed correctly.
