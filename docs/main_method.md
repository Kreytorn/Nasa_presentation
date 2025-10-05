# Main Method: Transit Photometry Detection

## Overview

Transit photometry is the primary exoplanet detection method used in this project. When a planet passes in front of its host star from our viewpoint, it causes a small, periodic dip in the star's brightness. Our machine learning system learns to identify these characteristic patterns in light curves.

## Why Transit Detection?

Transit photometry has discovered the majority of confirmed exoplanets and offers several advantages:

- **Proven track record**: Kepler, TESS, and K2 missions have confirmed thousands of planets using this method
- **Direct measurement**: Provides planet radius, orbital period, and transit timing
- **High accuracy**: Clear signal when properly aligned
- **Scalable**: Can process thousands of light curves automatically

## Datasets Used

### Kepler Objects of Interest (KOI)
- **Source**: NASA Kepler Mission
- **Coverage**: 150,000+ stars monitored continuously for 4 years
- **Confirmed planets**: 2,700+
- **Key features**: Deep, high-cadence photometry, well-characterized stellar parameters

### TESS Objects of Interest (TOI)
- **Source**: NASA Transiting Exoplanet Survey Satellite
- **Coverage**: 200,000+ bright stars across the entire sky
- **Observation mode**: 2-minute and 30-minute cadence
- **Advantage**: All-sky survey covering diverse stellar populations

### K2 Planets and Candidates
- **Source**: Kepler Extended Mission (K2)
- **Coverage**: 500,000+ targets across multiple sky fields
- **Diversity**: Young stars, eclipsing binaries, diverse campaigns
- **Unique contribution**: Different stellar types and environments

### Combined Training Set
From these missions, we compiled **21,000 labeled light curves** containing:
- Confirmed transiting planets
- Vetted false positives (eclipsing binaries, instrumental artifacts)
- Non-transiting stars

## Data Preprocessing

### 1. Dataset Unification
Each mission produces data in different formats with varying cadences and quality metrics. We standardized:
- Time stamps (converted to consistent Julian date format)
- Flux normalization (relative brightness centered at 1.0)
- Error propagation (uncertainty estimates)
- Metadata harmonization (stellar parameters, observation conditions)

### 2. Missing Value Imputation
Cross-dataset integration resulted in missing features for some targets. We applied **K-Nearest Neighbors (KNN) imputation** with k=5 neighbors:
- Preserves statistical relationships between features
- More robust than mean/median filling
- Accounts for correlations in stellar parameters

### 3. Feature Engineering
Extracted transit-specific features from raw light curves:

**Transit Morphology:**
- Transit depth (fractional brightness decrease)
- Transit duration (ingress to egress time)
- Transit shape (U-shaped vs V-shaped)
- Ingress/egress slopes

**Periodicity:**
- Orbital period (using Lomb-Scargle periodogram)
- Transit timing variations
- Period stability across multiple transits

**Signal Quality:**
- Signal-to-noise ratio (SNR)
- Box Least Squares (BLS) detection statistic
- Phase-folded light curve coherence

**Stellar Context:**
- Host star effective temperature
- Stellar radius and mass
- Photometric magnitude
- Distance (parallax-based)

## Model Architecture: Ensemble Approach

Rather than relying on a single model, we trained three complementary tree-based algorithms:

### XGBoost (Extreme Gradient Boosting)
- Gradient boosting with L1/L2 regularization
- Handles complex feature interactions
- Built-in feature importance ranking
- Strong performance on imbalanced datasets

### LightGBM (Light Gradient Boosting Machine)
- Efficient histogram-based algorithm
- Faster training on large datasets
- Leaf-wise tree growth (deeper, more accurate trees)
- Native handling of categorical features

### CatBoost (Categorical Boosting)
- Specialized for categorical feature encoding
- Ordered boosting reduces overfitting
- Robust to hyperparameter choices
- Symmetric tree structure

### Ensemble Strategy: Soft Voting
Final predictions combine all three models using probability averaging:
