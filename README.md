# ExoHunter: Multi-Signal Planetary Detection System
NASA Space Apps Challenge 2025 | Team inzva
Challenge: A World Away: Hunting for Exoplanets with AI

---

## Overview
Multi-method exoplanet detection system combining transit photometry, astrometry, radial velocity, and microlensing signals. Each detection method is trained independently, then fused for robust planetary candidate identification.

---

## Repository Structure
- exoplanet-detection-ai/
  - transit_detection/: Primary method - Kepler/TESS light curves
    - notebooks/: Training pipeline
    - models/: Saved CNN models
    - results/: Performance metrics and plots
  - supplementary_methods/: Alternative detection techniques
    - radial_velocity.ipynb: Stellar wobble (spectroscopy)
    - microlensing.ipynb: Gravitational lensing events
    - astrometry.ipynb: Position shifts (Gaia DR3)
    - direct_imaging.ipynb: Visual detection
  - docs/: Detailed documentation
    - main_method.md: Transit detection deep-dive
    - supplementary_methods.md: Other techniques explained
    - datasets.md: Data sources and preprocessing
    - results.md: Performance analysis
  - demo/: Interactive demonstrations
  - data/: Dataset info and download scripts

---

## Quick Start

### Prerequisites
pip install -r requirements.txt

### Run Transit Detection (Primary Method)
cd transit_detection/notebooks
Open and run the training notebook in order.

### Run Supplementary Methods
cd supplementary_methods
Each notebook is standalone—run individually.

---

## Methods Overview

### Transit Detection (Primary)
Detects periodic brightness dips in stellar light curves caused by planets crossing in front of their host stars. Uses a CNN on Kepler/TESS data.
Performance: ROC-AUC 0.98, Recall 0.94
Location: transit_detection/

### Astrometry
Identifies planet-hosting stars via tiny position wobbles using Gaia DR3 orbital solutions. Cross-matched with NASA Exoplanet Archive.
Performance: ROC-AUC 0.99, PR-AUC 0.86 (139k stars analyzed)
Location: supplementary_methods/astrometry.ipynb

### Microlensing
Detects short-duration brightness spikes from gravitational lensing. Trained on Roman Space Telescope 2018 synthetic campaign.
Performance: 93.5% accuracy on synthetic planet signals (293 dual-filter events)
Location: supplementary_methods/microlensing.ipynb

### Radial Velocity
Analyzes stellar velocity oscillations caused by orbiting planets using spectroscopic data.
Location: supplementary_methods/radial_velocity.ipynb

### Direct Imaging
Visual detection of planets in resolved systems (under development).
Location: supplementary_methods/direct_imaging.ipynb

---

## Data Sources
- NASA Exoplanet Archive — Confirmed planets & host stars
- Kepler/TESS — Transit light curves
- Gaia DR3 — Astrometric orbits (Non-Single-Star catalog)
- Roman Space Telescope — Microlensing simulations

See data/README.md for download instructions.

---

## Documentation
Detailed guides in docs/:
- main_method.md — Transit detection architecture & training
- supplementary_methods.md — Alternative detection pipelines
- datasets.md — Data preprocessing & sources
- results.md — Comprehensive performance analysis

---

## Results Summary
Method | Dataset Size | Metric   | Score
------ | -------------|----------|------
Transit | 5,087 curves | ROC-AUC | 0.98
Astrometry | 139,649 stars | ROC-AUC | 0.99
Microlensing | 293 events | Accuracy | 93.5%

---

## Team
The Barrel — NASA Space Apps Challenge 2025

---

## License
See the LICENSE file for details.

---

## Citation
If you use this work, please reference:
ExoHunter: Multi-Signal Planetary Detection System
NASA Space Apps Challenge 2025, Team inzva
https://github.com/Kreytorn/Nasa_presentation


