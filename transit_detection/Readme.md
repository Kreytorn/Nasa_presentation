## Exoplanet Classification Project

This repository contains three Jupyter notebooks that build a complete pipeline for exoplanet data preparation and classification using modern gradient boosting models and an ensemble:

1) NASA_Exoplanet_Data_Analysis.ipynb
- Merges and aligns NASA exoplanet datasets (Kepler, TESS, K2) using a schema-first approach.
- Performs aggressive feature cleaning, KNN imputation, and exports unified datasets and artifacts (CSV/Parquet, schema YAML, rename maps JSON, merge report).
- Produces training and candidate splits: confirmed/false-positive labels for supervised training and candidates for prediction.

2) XGBoost_Exoplanet_Classification_CLEAN.ipynb
- Loads the unified training/candidate data and performs data quality checks and cleaning.
- Uses a star-based split to prevent data leakage, trains an XGBoost model, evaluates with ROC AUC, and predicts candidate labels and probabilities.
- Exports a final CSV that combines training data with predicted candidates and summary statistics/visualizations.

3) Ensemble_Exoplanet_Classification_SIMPLE.ipynb
- Trains XGBoost, LightGBM, and CatBoost models, evaluates each, and combines them with soft-voting (probability averaging).
- Compares individual vs ensemble accuracy, analyzes agreement between models, and exports trained models and feature metadata.

### Key Practices
- Star-based splitting to avoid data leakage between train/test.
- Schema-first dataset alignment across Kepler/TESS/K2.
- Probability outputs for downstream UI/thresholding.

### Inputs and Artifacts
- Input CSVs are expected under `Nasa-Exoplanet/` (cleaned Kepler/TESS/K2 files).
- Unified/derived artifacts are written in the project root (e.g., `unified_exoplanets.csv`, `unified_exoplanet_dataset_with_predictions.csv`, model `.pkl` files, and schema/report files).

### Quick Start
1. Open the notebooks in order:
   - `NASA_Exoplanet_Data_Analysis.ipynb`
   - `XGBoost_Exoplanet_Classification_CLEAN.ipynb`
   - `Ensemble_Exoplanet_Classification_SIMPLE.ipynb`
2. Run each notebook top-to-bottom.
3. Review `unified_exoplanet_dataset_with_predictions.csv` and the exported models/reports.

### Environment
- Python 3.10+
- Core libs: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, seaborn, matplotlib, joblib, yaml

### Outputs to Look At
- `unified_exoplanets.csv` / `unified_exoplanets.parquet` (merged dataset)
- `unified_exoplanets_train.csv` (training subset)
- `unified_exoplanet_dataset_with_predictions.csv` (training + predicted candidates)
- `ensemble_soft_voting.pkl` and individual model `.pkl` files
- `merge_report.md`, `unified_schema.yaml`, `rename_maps.json`


