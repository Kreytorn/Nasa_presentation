# Supplementary Detection Methods

## Strategic Rationale: Why Multiple Methods?

Transit photometry dominates exoplanet discovery statistics, but this reflects **observational bias**, not physical reality. Each detection method has fundamental geometric, orbital, and stellar-type constraints that create blind spots in parameter space. A comprehensive exoplanet characterization toolkit requires multi-method capability.

### The Problem with Single-Method Approaches

**Transit photometry limitations:**
- **Geometric constraint**: Requires orbital inclination i > 80 degrees (edge-on alignment). For randomly oriented orbits, transit probability is proportional to stellar radius divided by orbital separation. A Jupiter analog (a = 5 AU) around a solar-type star has less than 0.1% transit probability.
- **Period bias**: Short-period planets transit frequently, enabling confirmation within mission lifetimes. Long-period planets (P > 1 year) require multi-year baselines and may transit only once.
- **Host star bias**: Optimized for quiet, main-sequence stars. Active stars, giants, and binaries produce false positives.
- **Information deficit**: Measures planet-to-star radius ratio, but cannot determine planet mass independently.

**Why this matters:**
Our Solar System's architecture would be invisible to Kepler/TESS. Jupiter (P = 11.86 years, i approximately 1.3 degrees) would never transit. Saturn, Uranus, Neptune are non-transiting from the ecliptic plane. Transit surveys inherently miss the majority of planetary systems.

### Our Contribution: End-to-End Multi-Method Framework

Rather than building a single-purpose transit classifier, we developed **parallel AI pipelines for five detection methods**, each with:
- Domain-specific preprocessing (PCA-ADI for imaging, periodogram extraction for RV, etc.)
- Method-appropriate architectures (CNNs for images, tree models for tabular, RNNs for sequences)
- Rigorous evaluation with class-imbalanced metrics (PR-AUC, F1-optimized thresholds)

This demonstrates **technical versatility** and **scientific completeness**: the system mirrors professional workflows where complementary methods validate and characterize planetary systems.

---

## 1. Radial Velocity Spectroscopy

### Physical Principle

Orbital mechanics dictates that star and planet orbit their common barycenter. For a planet of mass Mp orbiting a star of mass M★ at semi-major axis a:

**Semi-amplitude of stellar wobble:**

K = (2πG / P)^(1/3) × (Mp sin i) / (M★ + Mp)^(2/3) × 1/√(1 - e²)

For Jupiter around the Sun: K ≈ 12.5 m/s over P = 11.86 years.

We measure K via **Doppler shift** in stellar absorption lines. As the star moves toward us, lines blueshift; as it recedes, they redshift. High-resolution spectroscopy (R > 100,000) resolves these shifts.

### Why RV Complements Transit

| Property | Transit | Radial Velocity |
|----------|---------|-----------------|
| **Geometric requirement** | i ≈ 90° (edge-on) | None (works at any inclination) |
| **Period sensitivity** | Biased toward P < 50 days | Efficient for P = 1–10 years |
| **Mass measurement** | No (only radius) | Yes (Mp sin i) |
| **False positive rate** | ~10% (eclipsing binaries) | Low (spectroscopic confirmation) |

**Key advantage**: RV provides **planet mass**, enabling density calculation ρp = Mp/Vp when combined with transit radius. This distinguishes rocky planets from gas giants.

### Dataset & Preprocessing

**Primary sources:**
- **HARPS** (High Accuracy Radial velocity Planet Searcher): ESO 3.6m telescope, La Silla Observatory. Wavelength range 378–691 nm, velocity precision ~1 m/s.
- **HIRES** (High Resolution Echelle Spectrometer): Keck I 10m telescope. Precision ~2–3 m/s.

**Data structure:**
Time-series arrays per star with columns:
- BJD (Barycentric Julian Date): time stamps
- RV (m/s): line-of-sight velocity
- RV_err (m/s): measurement uncertainty
- instrument: HARPS/HIRES (for cross-calibration)

**Preprocessing pipeline:**
1. **Outlier rejection**: Median Absolute Deviation (MAD) filtering, removing points with |z| > 7σ
2. **Detrending**: Subtract polynomial fit for long-term instrumental drift and stellar activity
3. **Normalization**: Per-star z-score standardization
4. **Feature extraction**: Lomb-Scargle periodogram analysis

**Extracted features** (per star):
- **Periodogram peaks**: Top 3 frequency components, amplitudes, false alarm probabilities
- **Statistical moments**: Velocity mean, standard deviation, skewness, kurtosis
- **Semi-amplitude proxy**: Half the peak-to-peak velocity range
- **Phase coherence**: Variance of phase-folded data at dominant period
- **Stellar context**: M★, [Fe/H] (metallicity), Teff (temperature)

### Model Architecture: LightGBM

**Rationale:**
- Tree-based models excel at capturing non-linear feature interactions (e.g., period-amplitude-eccentricity correlations)
- Gradient boosting with leaf-wise growth handles complex decision boundaries
- Native support for missing values and categorical features (instrument type)
- Fast training on tabular data (~1000 features × 800 stars)

**Configuration:**

n_estimators = 500
max_depth = 8
learning_rate = 0.05
num_leaves = 63

**Training protocol:**
- Stratified train/test split by star (prevent data leakage from same system)
- Class balancing: undersample non-planet stars to match confirmed planet count
- Evaluation: ROC-AUC, Average Precision (PR-AUC)

### Performance

**Test set evaluation:**
- **AUC**: 0.87 (strong discriminative ability)
- **AP (Average Precision)**: 0.82 (robust to class imbalance)
- **Best F1 threshold**: τ = 0.42
- **Precision at τ**: 0.79 (79% of predicted planets are true positives)
- **Recall at τ**: 0.84 (catches 84% of actual planets)

**Feature importance (top contributors):**
1. Periodogram peak amplitude (32%)
2. Semi-amplitude estimate (24%)
3. Period stability (18%)
4. Velocity dispersion (12%)
5. Stellar mass (8%)

### Technical Innovation

Unlike image-based deep learning, RV detection requires **domain-specific signal processing**. Our pipeline:
- Implements robust periodogram analysis with proper false alarm probability correction
- Handles multi-instrument data with systematic offset calibration
- Extracts physically motivated features rather than raw time-series (more sample-efficient)

---

## 2. Gravitational Microlensing

### Physical Principle

General relativity predicts that massive objects bend spacetime, deflecting light rays passing nearby. When a foreground star ("lens") passes in front of a background star, the lens's gravity acts as a natural telescope, magnifying the background star's light.

**Einstein radius**: The characteristic angular scale of gravitational lensing

**Magnification**: A(t) = (u² + 2) / (u × √(u² + 4))

Where u(t) is the normalized lens-source separation as a function of time.

**Planetary signature:**
If the lens star has a planet, the planet creates a **second Einstein ring** with much smaller radius. When the source crosses this ring, magnification exhibits a short-duration spike (hours to days) superimposed on the main lensing event (weeks).

### Why Microlensing Complements Other Methods

| Property | Transit/RV | Microlensing |
|----------|------------|--------------|
| **Orbital distance** | Biased toward a < 1 AU | Efficient for a > 1 AU (cold planets) |
| **Galactic reach** | Limited to ~1 kpc (nearby stars) | Probes Galactic bulge (~8 kpc) |
| **Repeatable?** | Yes (periodic signal) | No (one-time event) |
| **Host star** | Must be bright | Can be faint or invisible |

**Critical niche**: Microlensing finds **cold Jupiters and super-Earths** at 2–10 AU, inaccessible to transits and challenging for RV (require decade-long baselines).

### Dataset & Preprocessing

**Primary source:**
- **Roman Space Telescope (WFIRST) 2018 Microlensing Challenge**: Synthetic Galactic bulge survey simulation
- **293 dual-filter lensing events**: W149 (1.5 μm) and Z087 (0.9 μm)

**Data structure:**
Text files per event/filter with columns:
- BJD: time stamps
- mag: apparent magnitude
- err: photometric uncertainty

**Preprocessing pipeline:**

1. **Pair events by ID**: Match W149 and Z087 files for same event

2. **Robust outlier rejection**:
   - Compute Median Absolute Deviation (MAD) in magnitude space
   - Clip points with |z| > 7σ
   - Removes cosmic ray hits and bad pixels

3. **Magnitude to Flux conversion**:
   F proportional to 10^(-0.4 × mag)
   (Flux is linearly additive, magnitude is logarithmic)

4. **Time window alignment**:
   - For dual-filter events: use overlapping time span
   - For single-filter: use full span

5. **Uniform resampling**:
   - Linear interpolation to fixed grid of **512 time points**
   - Forward/backward fill edge NaNs
   - Zero-fill remaining gaps

6. **Per-channel normalization**:
   z = (x - μ) / σ
   (Zero mean, unit variance per filter)

**Output tensors:**
- X_2ch.npy: Shape (293, 512, 2) for dual-filter events
- y.npy: Labels (0/1 for non-planet/planet)
- manifest.csv: Metadata per event (time spans, filter counts, peak estimates)

**Label generation challenge:**
The Roman 2018 dataset lacks ground-truth planet labels. To demonstrate model capability, we **injected synthetic planetary anomalies**:
- Gaussian bumps with random center t0, width σ in [3, 12] points, amplitude A in [1.0, 2.5] flux units
- Injected into ~35% of curves
- Slight amplitude mismatch between channels (realistic chromatic effects)

### Model Architecture: Conv1D + GRU Hybrid

**Rationale:**
Microlensing signals have **dual timescale structure**:
- **Long-term**: Main lensing event (weeks, smooth magnification profile)
- **Short-term**: Planetary spike (hours to days, sharp peak)

**Architecture:**

Input: (batch, 512, 2) - time × channels

Conv1D stack:
  Conv1D(2 to 32, kernel=7) + ReLU + MaxPool(2)
  Conv1D(32 to 64, kernel=5) + ReLU + MaxPool(2)
  Conv1D(64 to 128, kernel=3) + ReLU
  Result: (batch, 128, 128) - spatial features

Bidirectional GRU:
  GRU(128 to 64, bidirectional=True)
  Result: (batch, 128, 128) - temporal dependencies

Global pooling:
  Mean over time - (batch, 128)

Dense head:
  Linear(128 to 1) + Sigmoid
  Result: probability of planet

**Why this design?**
- **Conv1D**: Detects local morphology (bump shape, slope changes) with translation invariance
- **MaxPool**: Downsamples while preserving salient features, reduces computation
- **Bidirectional GRU**: Captures long-range temporal context (how baseline evolves before/after bump)
- **Dual-channel input**: Learns wavelength-dependent signatures (chromatic microlensing effects)

### Training Protocol

**Loss function:** Binary Cross-Entropy with Logits
**Class weighting:** pos_weight = n_negatives / n_positives (handles imbalance)
**Optimizer:** AdamW (lr = 3e-4, weight decay = 1e-4)
**Epochs:** 15 with early stopping (patience = 3)
**Split:** Stratified 90/10 train/test

### Performance (Synthetic Labels)

**Test set (31 events, 11 positives):**
- **PR-AUC**: 1.000 (perfect ranking of positives)
- **Confusion matrix**:
  TN = 20, FP = 0
  FN = 2, TP = 9
- **Precision**: 1.00 (no false alarms)
- **Recall**: 0.82 (missed 2/11 planets)
- **Accuracy**: 93.5%

**Interpretation:**
These metrics reflect **synthetic anomalies**, not real planet detections. The model successfully learns to identify Gaussian bump patterns, demonstrating architectural soundness. Real-world validation requires labeled OGLE/MOA planet events.

### Technical Innovation

**Multi-channel time-series deep learning:**
- Preserves filter information as separate channels (like RGB in images)
- Conv1D acts as learned feature detector (replaces hand-crafted periodogram analysis)
- GRU captures non-Markovian dependencies (magnification history informs current state)

**Production-ready inference:**
Exported standalone functions predict_single_event() and predict_many_events() that:
- Accept CSV/NPY/TXT inputs
- Auto-detect channels (1 or 2)
- Auto-resample to 512 points
- Return calibrated probabilities

---

## 3. Astrometry (Positional Wobble)

### Physical Principle

Both RV and astrometry measure stellar wobble induced by planetary companions, but in **orthogonal directions**:
- **RV**: Line-of-sight velocity (Doppler shift)
- **Astrometry**: Sky-plane position (angular displacement)

**Angular wobble amplitude:**
α = (Mp/M★) × (a/D)

Where a is orbital semi-major axis, D is distance to system.

For Jupiter around the Sun viewed from 10 pc:
α ≈ 500 microarcseconds (μas)

**Gaia precision**: ~20–30 μas for bright stars over 5-year baseline, enabling marginal Jupiter detection but insufficient for Earth-mass planets at current sensitivity.

### Why Astrometry Complements RV

| Property | Radial Velocity | Astrometry |
|----------|-----------------|------------|
| **Measures** | Line-of-sight velocity | Sky-plane position |
| **Inclination** | Mp sin i (degenerate) | Mp (breaks degeneracy) |
| **Period sensitivity** | Efficient P < 10 years | Best for P > 1 year (long baseline) |
| **Stellar activity** | Affected (line distortion) | Immune (geometric) |

**Key advantage**: Astrometry measures **true planet mass Mp** (not Mp sin i), enabling direct density calculation without assuming inclination.

### Dataset & Preprocessing

**Primary source:**
- **Gaia DR3 Non-Single-Star (NSS) Two-Body Orbit Catalog**: ESA Gaia mission
- **139,649 orbital solutions** after quality cuts

**Quality filters applied in ADQL query:**

ruwe < 1.4
parallax_over_error > 10
visibility_periods_used >= 8

**Joined tables:**
- gaiadr3.nss_two_body_orbit: Orbital parameters
- gaiadr3.gaia_source: Photometry and quality metrics

**Features extracted** (23 total):

**Orbital dynamics:**
- period (days): Orbital period
- eccentricity: Orbit shape (0 = circular, ~1 = highly elliptical)
- inclination (deg): Orbit tilt relative to sky plane
- arg_periastron_sin, arg_periastron_cos: Angle-safe encoding of ω (prevents discontinuity at 0°/360°)

**Astrometric quality:**
- parallax_over_error: Distance measurement SNR
- ruwe: Renormalized Unit Weight Error (goodness-of-fit)
- astrometric_chi2_al: Along-scan χ² statistic
- astrometric_excess_noise: Unexplained positional scatter
- visibility_periods_used: Number of observation epochs

**Photometry:**
- phot_g_mean_mag: G-band apparent magnitude (brightness)

**Excluded from features (label leakage risk):**
- mass_ratio: Directly correlates with planet vs. star companion
- Any derived mass estimates

### Labeling Strategy

**Challenge**: Gaia NSS catalog contains **stellar binaries and planetary systems** mixed together. Most companions are low-mass stars, not planets.

**Our approach**:
Cross-match with **NASA Exoplanet Archive** confirmed planet catalog:
1. **Direct match**: Gaia source_id in NASA.gaia_id
2. **Positional fallback**: 2.0 arcsecond cone search (RA/Dec match)

**Result:**
- **Positives**: 16 confirmed exoplanet host stars
- **Negatives**: 139,633 other orbital solutions (mostly binaries)

**Extreme class imbalance** (0.01% positive rate) requires balanced training set

**Balanced subset construction:**
- Keep all 16 positives
- Sample 800 negatives (50:1 ratio)
- **Total training set**: 816 rows
- Stratified 75/25 split, test set: 204 rows (4 positives, 200 negatives)

### Model Architecture: Random Forest

**Rationale:**
- **Interpretable**: Feature importance reveals which orbital parameters distinguish planets from binaries
- **Robust to outliers**: Ensemble of trees votes, reducing overfitting
- **No feature scaling required**: Tree splits are threshold-based, invariant to monotonic transforms
- **Handles mixed data types**: Numeric (period, eccentricity) and derived (sin/cos angles)

**Configuration:**

n_estimators = 400
max_depth = 12
class_weight = 'balanced_subsample'
random_state = 42

### Performance

**Test set (204 samples: 200 negatives, 4 positives):**

**Confusion matrix at F1-optimal threshold (τ = 0.557):**

               Predicted
            No Planet  Planet
Actual No      200       0
Actual Planet    1       3

**Metrics:**
- **ROC-AUC**: 0.994 (near-perfect ranking)
- **PR-AUC**: 0.861 (realistic given 1:50 imbalance)
- **Accuracy**: 99.5%
- **Precision**: 1.00 (no false positives)
- **Recall**: 0.75 (missed 1 of 4 planets)
- **Specificity**: 1.00 (perfect negative identification)

**Feature importance (top 5):**
1. period (28%): Planets have shorter periods than wide binaries
2. eccentricity (19%): Planets tend toward circular orbits
3. parallax_over_error (16%): Nearby systems better characterized
4. inclination (12%): Orbital geometry encodes dynamical stability
5. phot_g_mean_mag (9%): Brightness correlates with detection ease

### Technical Innovation

**Trigonometric feature engineering:**
Raw angle arg_periastron (ω) has discontinuity at 0°/360°. We expand:

ω → [sin(ω), cos(ω)]

This creates continuous representation on unit circle, preserving angle relationships for tree splits.

**Cross-archive integration:**
Automated TAP query to NASA Exoplanet Archive, handles missing Gaia IDs with positional matching, validates via Astropy coordinate transformations.

---

## 4. Direct Imaging (High-Contrast Coronagraphy)

### Physical Principle

All previous methods detect planets **indirectly** via their gravitational or occultation effects. Direct imaging photographs the planet itself, capturing **photons emitted or reflected by the planet**.

**Challenge**: Contrast ratio
At visible wavelengths, Jupiter reflects ~10^-9 of the Sun's light. For young planets (age < 100 Myr), thermal emission in near-infrared provides ~10^-6 contrast.

**Solution**: Extreme Adaptive Optics (ExAO) + Coronagraph
- **ExAO**: Deformable mirror corrects atmospheric turbulence at 1 kHz
- **Coronagraph**: Physical mask blocks starlight, creating "dark hole"
- **Post-processing**: Subtract residual speckle patterns via reference star observations or angular differential imaging (ADI)

### Why Direct Imaging is Unique

| Property | Other Methods | Direct Imaging |
|----------|---------------|----------------|
| **Information** | Indirect (gravity, occultation) | Direct (planet photons) |
| **Spectroscopy** | No | Yes (atmospheric composition) |
| **Orbit constraint** | Various | Wide separation (>5 AU for ground-based) |
| **Target age** | Any | Young (<1 Gyr, self-luminous planets) |

**Key advantage**: **Direct atmospheric characterization**. Spectroscopy of planet light reveals H2O, CH4, CO absorption features, providing temperature, chemistry, cloud structure.

### Dataset & Preprocessing

**Primary source:**
- **VLT/SPHERE** (Spectro-Polarimetric High-contrast Exoplanet REsearch): ESO Very Large Telescope
- **IRDIS** (Infrared Dual-band Imager and Spectrograph): Near-IR imaging

**Data structure:**
FITS image cubes: (n_frames, height, width)
- n_frames: 50–200 exposures per target
- height × width: Typically 1024 × 1024 pixels

**Preprocessing: PCA-ADI (Principal Component Analysis + Angular Differential Imaging)**

**Step 1: Frame alignment**
- Centroid detection via 2D Gaussian fit
- Sub-pixel registration to common center
- Rotate frames to fixed north-up orientation

**Step 2: PCA speckle subtraction**

Reshape cube: (n_frames, height, width) to (n_frames, height×width)
X = cube.reshape(n_frames, -1)

Compute principal components:
U, S, Vt = np.linalg.svd(X, full_matrices=False)

Reconstruct stellar PSF from first K components:
PSF_model = U[:, :K] @ diag(S[:K]) @ Vt[:K, :]

Subtract model:
residual = X - PSF_model

Reshape back: (n_frames, height×width) to (n_frames, height, width)
residual = residual.reshape(n_frames, height, width)

**Why this works:**
- Star (dominant) captured by first K=10 principal components
- Planet (faint, rotating) appears in residual after subtraction
- Each component removes stellar speckle layer-by-layer

**GPU-accelerated implementation:**
Custom pca_adi_gpu_exact() using PyTorch for ~10× speedup on large cubes

**Step 3: Frame stacking**
- Median combination of residuals
- Outlier rejection via sigma-clipping
- Signal-to-noise (SNR) map generation

**Step 4: Synthetic planet injection (for training)**
- Add point sources with known positions/fluxes
- Apply realistic PSF convolution
- Calibrate SNR distribution

### Model Architecture: CNN + Transformer Hybrid

**Baseline: ExoplanetCNN**

Input: (batch, 1, H, W) - processed ADI frame

Conv blocks:
  Conv2D(1 to 32, 3×3) + ReLU + MaxPool(2)
  Conv2D(32 to 64, 3×3) + ReLU + MaxPool(2)
  Conv2D(64 to 128, 3×3) + ReLU + MaxPool(2)
  Result: (batch, 128, H/8, W/8)

Flatten + Dense:
  Linear(128×(H/8)×(W/8) to 256)
  Dropout(0.3)
  Linear(256 to 1) + Sigmoid

**Advanced: CNNTransformerExoplanet**

CNN feature extraction (as above)
  Result: (batch, 128, H/8, W/8)

Spatial attention:
  Reshape to (batch, (H/8)×(W/8), 128) - sequence
  
  Transformer encoder:
    Multi-head attention (8 heads)
    Position encoding (learnable)
    Attends to planet-like spatial patterns
  
Dense head:
  Global average pooling - (batch, 128)
  Linear(128 to 1) + Sigmoid

**Why transformer?**
- **Long-range dependencies**: Planet may appear anywhere in field of view
- **Attention mechanism**: Learns to focus on faint point sources amid speckle noise
- **Rotation invariance**: Attention is permutation-equivariant, handles ADI artifacts

### Training Protocol

**Dataset construction:**
- **Positives**: Synthetic planet injections at SNR in [3, 10]
- **Negatives**: Empty regions, stellar residuals
- **Augmentation**: Random rotations, flips, Gaussian noise

**Loss:** Binary Cross-Entropy with Logits
**Optimizer:** AdamW (lr = 1e-4, weight decay = 1e-5)
**Regularization:** Dropout (0.3), early stopping
**Batch size:** 16 (memory-constrained by image size)

**Evaluation:**
- PR-AUC (precision-recall, more informative than ROC for rare targets)
- False positive rate at fixed recall (critical for candidate vetting)
- Detection limit vs. SNR curve

### Performance

**Validation set:**
- **Precision**: >0.90 at SNR ≥ 5
- **Recall**: >0.85 for injected planets
- **False positive rate**: <5% in empty regions

**Ablation study:**
- CNN alone: 87% accuracy
- CNN + Transformer: 92% accuracy (+5% from spatial attention)

### Technical Innovation

**GPU-accelerated PCA:**
Standard VIP package uses CPU for SVD decomposition. Our PyTorch implementation achieves:
- ~10× speedup on NVIDIA T4
- Batch processing of multiple cubes
- Gradient-friendly (enables end-to-end training with PCA layer)

**Attention visualization:**
Transformer attention maps highlight predicted planet locations, providing interpretability for astronomers.

---

## Summary: Scientific Impact

### Parameter Space Coverage

Our five-method framework achieves comprehensive detection capability:

**Orbital separation:**
- Transit/RV: 0.01–1 AU (hot/warm planets)
- Astrometry: 1–10 AU (cold giants)
- Microlensing: 1–10 AU (cold planets, any distance)
- Direct Imaging: >5 AU (wide, young planets)

**Host star distance:**
- Transit/RV: <1 kpc (local)
- Astrometry: <100 pc (nearby, Gaia limit)
- Microlensing: <10 kpc (Galactic bulge)
- Direct Imaging: <150 pc (young stellar associations)

**Information obtained:**
- Transit: Radius, period
- RV: Mass × sin(i), period
- Astrometry: True mass, orbit
- Microlensing: Existence, rough mass
- Direct Imaging: Spectrum, luminosity, atmosphere

### Validation Strategy (Future Work)

**Cross-validation with multi-method detections:**
For systems detected by multiple methods, compare AI predictions:
- HR 8799 (direct imaging): Test imaging CNN on published data
- 51 Pegasi b (RV): Validate RV pipeline on archival HARPS data
- TrES-2b (transit + RV): Compare mass/radius from both methods

**Blind challenge participation:**
- NASA Roman microlensing challenge (real labeled events)
- ESA PLATO transit mock data challenge
- Gaia DR4 astrometric detections (when released)

### Computational Requirements

**Training time** (NVIDIA T4 GPU):
- Transit ensemble: 45 min
- RV LightGBM: 8 min
- Microlensing Conv1D+GRU: 25 min
- Astrometry RF: 5 min
- Direct imaging CNN: 90 min

**Inference throughput:**
- Transit: 10,000 light curves/hour
- RV: 5,000 stars/hour
- Microlensing: 8,000 events/hour
- Astrometry: 50,000 sources/hour
- Direct imaging: 100 cubes/hour

**Total training dataset**: 21,000 transit + 800 RV + 293 microlensing + 816 astrometry + ~500 imaging = approximately 23,400 labeled examples

---

## References

### Radial Velocity
- Mayor & Queloz (1995). "A Jupiter-mass companion to a solar-type star." Nature 378, 355–359.
- HARPS Consortium: https://www.eso.org/sci/facilities/lasilla/instruments/harps.html
- Butler et al. (1996). "Attaining Doppler precision of 3 m/s." PASP 108, 500.

### Microlensing
- Mao & Paczynski (1991). "Gravitational microlensing by double stars and planetary systems." ApJ 374, L37.
- Gould & Loeb (1992). "Discovering planetary systems through gravitational microlenses." ApJ 396, 104.
- Microlensing Data Challenge: https://github.com/microlensing-data-challenge

### Astrometry
- Perryman et al. (2014). "Astrometric exoplanet detection with Gaia." ApJ 797, 14.
- Gaia Collaboration (2023). "Gaia Data Release 3." A&A 674, A1.
- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/

### Direct Imaging
- Beuzit et al. (2019). "SPHERE: The exoplanet imager for the Very Large Telescope." A&A 631, A155.
- Marois et al. (2008). "Direct imaging of multiple planets orbiting the star HR 8799." Science 322, 1348.
- VIP (Vortex Image Processing): Gomez Gonzalez et al. (2017). AJ 154, 7.
