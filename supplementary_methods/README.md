# Radial Velocity Exoplanet Detection

This repository contains a full pipeline for **exoplanet detection using radial velocity (RV) data** from the NASA Exoplanet Archive and HARPS (rvbank). It automates everything from data download and parsing to feature extraction and model training. The workflow builds Lomb–Scargle periodogram features for each star and trains a LightGBM classifier to identify planet-hosting systems.

There are four main code blocks:
1. **Setup** – mounts Google Drive and defines directories.  
2. **Pipeline** – downloads, cleans, extracts features, and trains the model.  
3. **Sanity Checks** – summarizes dataset and feature distributions.  
4. **Inference Helpers** – provides a standalone script for making predictions on new RV time series.

All results (features, trained model, predictions) are automatically saved to a `/processed` folder inside your Drive.

# Microlensing Planet Detection

Detects planets using gravitational microlensing - when a star with a planet passes in front of another star, the planet creates a brief extra brightness spike.

## Quick Start

Run the notebook cells in order:
1. Downloads data from Roman Space Telescope 2018 challenge (293 events)
2. Processes lightcurves into uniform format
3. Trains neural network (15 minutes)
4. Ready for predictions

## Using the Model

Predict from any lightcurve file or array:
```python
# Single prediction
prob = predict_single_event("lightcurve.csv")

# Multiple files
probs, df = predict_many_events(["event1.csv", "event2.npy"])
```
Accepts CSV, NPY, or NumPy arrays. Auto-handles different lengths and channels.
