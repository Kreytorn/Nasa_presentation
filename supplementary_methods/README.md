# Radial Velocity Exoplanet Detection

This repository contains a full pipeline for **exoplanet detection using radial velocity (RV) data** from the NASA Exoplanet Archive and HARPS (rvbank). It automates everything from data download and parsing to feature extraction and model training. The workflow builds Lomb–Scargle periodogram features for each star and trains a LightGBM classifier to identify planet-hosting systems.

There are four main code blocks:
1. **Setup** – mounts Google Drive and defines directories.  
2. **Pipeline** – downloads, cleans, extracts features, and trains the model.  
3. **Sanity Checks** – summarizes dataset and feature distributions.  
4. **Inference Helpers** – provides a standalone script for making predictions on new RV time series.

All results (features, trained model, predictions) are automatically saved to a `/processed` folder inside your Drive.
