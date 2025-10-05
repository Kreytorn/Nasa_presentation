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
Detects periodic brightness dips in stellar light curves caused by planets crossing in front of their host stars. Uses a CNN/gradient-boosting ensemble on Kepler/TESS data.
Performance: Accuracy 92.0% (ROC-AUC 0.956, PR-AUC 0.932)
Location: transit_detection/

### Astrometry
Identifies planet-hosting stars via tiny position wobbles using Gaia DR3 orbital solutions. Cross-matched with NASA Exoplanet Archive.
Performance: ROC-AUC 0.994, PR-AUC 0.861
Location: supplementary_methods/astrometry.ipynb

### Microlensing
Detects short-duration brightness spikes from gravitational lensing. Trained on Roman Space Telescope 2018 synthetic campaign.
Performance: 93.5% accuracy on synthetic planet signals (293 dual-filter events)
Location: supplementary_methods/microlensing.ipynb

### Radial Velocity
Analyzes stellar velocity oscillations caused by orbiting planets using spectroscopic data.
Performance: ROC-AUC 0.870
Location: supplementary_methods/radial_velocity.ipynb

### Direct Imaging
Visual detection of planets in resolved systems (high-contrast imaging).
Performance: 92% accuracy (CNN + Transformer)
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
| Method         | Dataset Size        | Primary Metric | Score   |
|----------------|---------------------|----------------|---------|
| Transit        | 21,000 light curves | Accuracy       | 92.0%   |
| Astrometry     | 139,649 stars       | ROC-AUC        | 0.994   |
| Microlensing*  | 293 events          | Accuracy       | 93.5%   |
| Radial Velocity| 1,200+ stars        | ROC-AUC        | 0.870   |
| Direct Imaging | ~5,000 patches      | Accuracy       | 92%     |

*Microlensing results use synthetic labels from the Roman 2018 challenge.

---

## Team
Team inzva — NASA Space Apps Challenge 2025

---

## License
See the LICENSE file for details.

---

## Citation
If you use this work, please reference:
ExoHunter: Multi-Signal Planetary Detection System
NASA Space Apps Challenge 2025, Team inzva
https://github.com/Kreytorn/Nasa_presentation
