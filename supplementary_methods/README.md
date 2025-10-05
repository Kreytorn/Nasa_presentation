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

# Astrometry Exoplanet Detection

Identifies planet-hosting stars using Gaia DR3 astrometric wobbles - tiny position shifts caused by orbiting companions.

## Quick Start

Run notebook cells in order:
1. Mount Drive
2. Process Gaia data with adaptive labels
3. Cross-match with NASA confirmed hosts (16 positives found)
4. Train RandomForest classifier
5. Load inference functions

## Using the Model

Score a star from orbital parameters:
```python
star = {
    "period": 318.6,              # days
    "eccentricity": 0.26,
    "inclination": 70.0,          # degrees
    "parallax_over_error": 1075,
    "ruwe": 0.84,
    "astrometric_chi2_al": 250.0,
    "astrometric_excess_noise": 0.0,
    "visibility_periods_used": 12,
    "phot_g_mean_mag": 5.59,
    "arg_periastron": 120.0
}

score = score_astrometry(star)  # returns 0-1 probability
