## Results Summary

This document summarizes key outputs and performance from the notebooks.

### Dataset and Merge (from NASA_Exoplanet_Data_Analysis.ipynb)
- Unified dataset size: ~21,271 rows (after cleaning/alignment).
- Training subset (confirmed/false-positive): ~12,194 rows.
- Candidates: ~9,077 rows.
- Artifacts: `unified_exoplanets.csv`, `unified_exoplanets.parquet`, `unified_exoplanets_train.csv`, `unified_schema.yaml`, `rename_maps.json`, `merge_report.md`.

### XGBoost Classification (from XGBoost_Exoplanet_Classification_CLEAN.ipynb)
- Star-based split to prevent leakage.
- Test ROC AUC: ~0.996 (indicative; verify by rerunning in your environment).
- Final combined CSV: `unified_exoplanet_dataset_with_predictions.csv` (training + predicted candidates with probabilities).
- Candidate predictions distribution (indicative): ~53% class 0, ~47% class 1.

### Ensemble (from Ensemble_Exoplanet_Classification_SIMPLE.ipynb)
- Individual test accuracies (indicative):
  - XGBoost: ~0.905
  - LightGBM: ~0.903
  - CatBoost: ~0.902
- Soft Voting Accuracy: ~0.909 (improvement vs best individual model ~+0.004).
- Additional analyses: model agreement (2–1 cases), probability distributions, efficiency metrics, overfitting checks.
- Saved models: `xgb_exoplanet_model_soft.pkl`, `lgb_exoplanet_model_soft.pkl`, `cat_exoplanet_model_soft.pkl`, `ensemble_soft_voting.pkl`, and `feature_columns_soft.json`.

### Notes
- Some figures above are representative of a prior run and may change if you rerun cells.
- Always prefer star-based splits for evaluation to avoid train/test contamination.


