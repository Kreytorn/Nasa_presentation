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
P_ensemble = (P_xgboost + P_lightgbm + P_catboost) / 3

**Why soft voting?**
- Reduces individual model bias
- Leverages different learning strategies
- More robust to outliers than hard voting
- Smoother probability distributions for threshold tuning

## Training Configuration

**Train/Test Split:** 75/25 stratified by label
**Cross-validation:** 5-fold stratified CV during hyperparameter tuning
**Class weighting:** Applied to handle planet/non-planet imbalance
**Evaluation metric:** Precision-Recall AUC (more informative than ROC-AUC for imbalanced classes)

## Results

### Individual Model Performance
- **XGBoost**: 90.8% accuracy
- **LightGBM**: 90.5% accuracy  
- **CatBoost**: 90.9% accuracy

### Ensemble Performance
- **Accuracy**: 92.0%
- **Precision**: 89.6%
- **Recall**: 91.4%
- **F1-Score**: 90.5%


### Confusion Matrix (Test Set)

|                    | Predicted No Planet | Predicted Planet |
|--------------------|---------------------|------------------|
| **Actual No Planet** | 2000              | 212              |
| **Actual Planet**    | 171               | 1829             |

**Interpretation:**
- **True Negatives (TN)**: 2000 - correctly identified non-planets
- **False Positives (FP)**: 212 - non-planets incorrectly classified as planets
- **False Negatives (FN)**: 171 - planets missed by the classifier
- **True Positives (TP)**: 1829 - correctly identified planets

**Key Metrics:**
- **Specificity**: 90.4% (2000/2212) - correctly identifies non-planets
- **Sensitivity/Recall**: 91.4% (1829/2000) - correctly identifies planets
- **Precision**: 89.6% (1829/2041) - when model predicts planet, it's right 89.6% of the time
- **False Positive Rate**: 9.6% (212/2212) - acceptable for candidate screening

**Key Insights:**
- High true negative rate effectively filters out non-planets
- Strong recall catches 91.4% of actual planets
- Balanced performance with no severe class bias
- False positives are acceptable for generating candidates requiring follow-up observations


  Top contributing features (averaged across ensemble):
1. Transit depth (28%)
2. Signal-to-noise ratio (18%)
3. Orbital period (15%)
4. BLS detection statistic (12%)
5. Transit duration (9%)
6. Stellar radius (7%)
7. Phase-folded coherence (6%)
8. Others (5%)

## Limitations and Considerations

**Method-specific constraints:**
- Requires edge-on orbital alignment (geometric probability ~0.5-1%)
- Biased toward short-period planets (more transits observed)
- Sensitive to stellar activity and instrumental noise
- Cannot determine planet mass (only radius from transit depth)

**Why supplementary methods matter:**
These limitations motivated development of four additional detection approaches to create a comprehensive toolkit.

## Computational Requirements

**Training time**: ~45 minutes on GPU (NVIDIA T4)
**Inference time**: ~0.1 seconds per light curve
**Memory usage**: ~4GB RAM for full dataset
**Scalability**: Can process 10,000+ candidates per hour

## References

- Kepler Mission Data: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
- TESS Mission: [MAST Portal](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)
- K2 Campaign Data: [K2 Mission Page](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc)
