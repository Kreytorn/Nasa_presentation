# Supplementary Detection Methods

## Strategic Rationale: Why Multiple Methods?

Transit photometry dominates exoplanet discovery statistics, but this reflects **observational bias**, not physical reality. Each detection method has fundamental geometric, orbital, and stellar-type constraints that create blind spots in parameter space. A comprehensive exoplanet characterization toolkit requires multi-method capability.

### The Problem with Single-Method Approaches

**Transit photometry limitations:**
- **Geometric constraint:** Requires orbital inclination \( i \gtrsim 80^\circ \) (edge-on alignment). For randomly oriented orbits, transit probability \(\propto R_\star/a\). A Jupiter analog (\(a = 5\,\mathrm{AU}\)) around a solar-type star has \<0.1% transit probability.
- **Period bias:** Short-period planets transit frequently, enabling confirmation within mission lifetimes. Long-period planets (\(P > 1\,\mathrm{yr}\)) require multi-year baselines and may transit only once.
- **Host star bias:** Optimized for quiet, main-sequence stars. Active stars, giants, and binaries produce false positives.
- **Information deficit:** Measures \(R_p/R_\star\) (planet-to-star radius), but cannot determine planet mass independently.

**Why this matters:**
Our Solar System’s architecture would be invisible to Kepler/TESS. Jupiter (\(P = 11.86\,\mathrm{yr},\ i \approx 1.3^\circ\)) would never transit. Saturn, Uranus, Neptune are non-transiting from the ecliptic plane. Transit surveys inherently miss the majority of planetary systems.

### Our Contribution: End-to-End Multi-Method Framework

Rather than building a single-purpose transit classifier, we developed **parallel AI pipelines for five detection methods**, each with:
- Domain-specific preprocessing (PCA-ADI for imaging, periodogram extraction for RV, etc.)
- Method-appropriate architectures (CNNs for images, tree models for tabular, RNNs for sequences)
- Rigorous evaluation with class-imbalanced metrics (PR-AUC, F1-optimized thresholds)

This demonstrates **technical versatility** and **scientific completeness**: the system mirrors professional workflows where complementary methods validate and characterize planetary systems.

---

## 1. Radial Velocity Spectroscopy

### Physical Principle

Orbital mechanics dictates that star and planet orbit their common barycenter. For a planet of mass \(M_p\) orbiting a star of mass \(M_\star\) with period \(P\) and eccentricity \(e\):

**Semi-amplitude of stellar wobble:**
\[
K \;=\; \left(\frac{2\pi G}{P}\right)^{1/3} \frac{M_p \sin i}{\left(M_\star + M_p\right)^{2/3}} \frac{1}{\sqrt{1 - e^2}}
\]

For Jupiter around the Sun: \(K \approx 12.5\,\mathrm{m\,s^{-1}}\) over \(P = 11.86\,\mathrm{yr}\).

We measure \(K\) via **Doppler shift** in stellar absorption lines. As the star moves toward us, lines blueshift; as it recedes, they redshift. High-resolution spectroscopy (\(R \gtrsim 100{,}000\)) resolves these shifts.

### Why RV Complements Transit

| Property                | Transit              | Radial Velocity          |
|-------------------------|----------------------|--------------------------|
| **Geometric requirement** | \(i \approx 90^\circ\) (edge-on) | None (works at any \(i\)) |
| **Period sensitivity**  | Biased toward \(P \lesssim 50\) d | Efficient for \(P = 1\)–10 yr |
| **Mass measurement**    | No (radius only)     | Yes (\(M_p \sin i\))     |
| **False positive rate** | ~10% (EBs, blends)   | Low (spectroscopic)      |

**Key advantage:** RV provides **planet mass**, enabling density \( \rho_p = M_p / V_p \) when combined with transit radius. This distinguishes rocky planets from gas giants.

### Dataset & Preprocessing

**Primary sources:**
- **HARPS** (ESO 3.6m, La Silla): 378–691 nm, precision ~1 m/s
- **HIRES** (Keck I 10m): precision ~2–3 m/s

**Data structure (per star time series):**
- `BJD` — Barycentric Julian Date (time)
- `RV` — line-of-sight velocity (m/s)
- `RV_err` — measurement uncertainty (m/s)
- `instrument` — HARPS/HIRES

**Preprocessing pipeline:**
1. **Outlier rejection:** MAD filtering (clip \(|z| > 7\sigma\))
2. **Detrending:** Remove long-term instrumental/stellar trends (poly fit)
3. **Normalization:** Per-star z-score
4. **Feature extraction:** Lomb-Scargle periodogram

**Extracted features (per star):**
- **Periodogram peaks:** top 3 frequencies, amplitudes, FAPs
- **Moments:** mean, std, skewness, kurtosis of RV
- **Semi-amplitude proxy:** half peak-to-peak RV
- **Phase coherence:** variance of phase-folded data at dominant period
- **Stellar context:** \(M_\star\), [Fe/H], \(T_\mathrm{eff}\)

### Model Architecture: LightGBM

**Why:** non-linear interactions, fast on tabular data, missing-value handling.

**Config:**
n_estimators = 500
max_depth = 8
learning_rate = 0.05
num_leaves = 63

**Training protocol:**
- Stratified split **by star** (no leakage)
- Balance classes (undersample negatives)
- Metrics: ROC-AUC, PR-AUC

### Performance

- **AUC:** 0.87
- **AP:** 0.82
- **Best-F1 threshold:** \(\tau = 0.42\)
- **Precision @ \(\tau\):** 0.79
- **Recall @ \(\tau\):** 0.84

**Top feature importances:**
1. Periodogram peak amplitude (32%)
2. Semi-amplitude estimate (24%)
3. Period stability (18%)
4. Velocity dispersion (12%)
5. Stellar mass (8%)

**Technical innovation:** robust periodogram with FAP correction, multi-instrument offset calibration, and **physically motivated** features for sample-efficient learning.

---

## 2. Gravitational Microlensing

### Physical Principle

Mass bends spacetime; a foreground “lens” star magnifies a background source when aligned.

**Einstein radius:**
\[
\theta_E = \sqrt{\frac{4GM}{c^2} \cdot \frac{D_{LS}}{D_L D_S}}
\]

**Single-lens magnification:**
\[
A(t) = \frac{u(t)^2 + 2}{u(t)\sqrt{u(t)^2 + 4}}
\]

A planetary companion creates a **short-duration spike** (hours–days) atop the weeks-long stellar event.

### Why Microlensing Complements Other Methods

| Property            | Transit/RV                   | Microlensing                           |
|---------------------|------------------------------|----------------------------------------|
| **Orbital distance**| Biased to \(a \lesssim 1\,\mathrm{AU}\) | Efficient for \(a \gtrsim 1\,\mathrm{AU}\) (cold planets) |
| **Galactic reach**  | \(\lesssim 1\,\mathrm{kpc}\) | Bulge (\(\sim 8\,\mathrm{kpc}\))                 |
| **Repeatable?**     | Yes (periodic)               | No (one-time)                           |
| **Host star**       | Bright, nearby               | Can be faint/invisible                  |

**Critical niche:** cold Jupiters/super-Earths at 2–10 AU.

### Dataset & Preprocessing

**Primary source:** **Roman (WFIRST) 2018 Microlensing Challenge**
**Data:** 293 events, dual-filter W149 (1.5 μm) & Z087 (0.9 μm)

**Columns:** `BJD`, `mag`, `err` (per event/filter)

**Pipeline:**
1. Pair W149/Z087 for same event
2. **Robust clipping:** MAD, \(|z| > 7\sigma\)
3. **Mag → Flux:** \(F \propto 10^{-0.4\,\mathrm{mag}}\)
4. Align time windows (overlap)
5. **Resample:** 512 uniform points (interpolate, edge fill)
6. **Normalize:** per-channel z-score

**Outputs:**
- `X_2ch.npy`: \((N=293, 512, 2)\)
- `y.npy`: labels (0/1)
- `manifest.csv`, `meta.json`

**Labels:** base set unlabeled ⇒ **synthetic planetary anomalies** injected (Gaussian bumps; random \(t_0, \sigma, A\); slight cross-channel mismatch).

### Model: Conv1D + Bidirectional GRU

**Input:** \((\text{batch}, 512, 2)\)

**Stack:**
- Conv1D(2→32, k=7) → ReLU → MaxPool(2)
- Conv1D(32→64, k=5) → ReLU → MaxPool(2)
- Conv1D(64→128, k=3) → ReLU
- BiGRU(128→64 per dir)
- Global mean pool → Dense(128→1) + Sigmoid

**Why:** Conv1D catches local bump shapes; GRU models long-range evolution.

**Training:** BCEWithLogits, `pos_weight`, AdamW (3e-4, wd=1e-4), 15 epochs w/ early stop, stratified 90/10 split.

**Synthetic-label performance (test 31 events, 11 pos):**
- **PR-AUC:** 1.000
- **Confusion:** TN=20, FP=0, FN=2, TP=9
- **Precision:** 1.00 • **Recall:** 0.82 • **Acc:** 93.5%

> Note: reflects detection of **synthetic** anomalies; real OGLE/MOA-labeled validation is future work.

**Production inference:** `predict_single_event()` / `predict_many_events()` accept CSV/NPY/TXT, auto-resample to 512, output calibrated probabilities.

---

## 3. Astrometry (Positional Wobble)

### Physical Principle

RV measures line-of-sight velocity; **astrometry** measures sky-plane motion.

**Angular wobble amplitude:**
\[
\alpha = \frac{M_p}{M_\star} \cdot \frac{a}{D}
\]
For Jupiter–Sun at 10 pc: \(\alpha \approx 500\,\mu\mathrm{as}\).
**Gaia DR3**: ~20–30 μas for bright stars over 5-yr baseline.

### Why Astrometry Complements RV

| Property           | Radial Velocity                | Astrometry                         |
|--------------------|--------------------------------|------------------------------------|
| **Measures**       | LOS velocity                   | Sky-plane position                 |
| **Inclination**    | \(M_p \sin i\) (degenerate)    | True \(M_p\) (breaks degeneracy)   |
| **Period range**   | Best for \(P \lesssim 10\) yr  | Best for \(P \gtrsim 1\) yr        |
| **Stellar activity** | Affects lines                 | Geometric; largely immune          |

### Dataset & Preprocessing

**Primary source:** **Gaia DR3 NSS Two-Body Orbit Catalog**
**After quality cuts:** 139,649 orbital solutions

**Quality filters (ADQL):**
```sql
-- Example filters
ruwe < 1.4
parallax_over_error > 10
visibility_periods_used >= 8
Joined tables:

gaiadr3.nss_two_body_orbit — orbital parameters

gaiadr3.gaia_source — photometry & quality

Features extracted (23 total):

Orbital: period (days), eccentricity, inclination (deg),
arg_periastron → sin/cos encoded (arg_periastron_sin, arg_periastron_cos)

Astrometric quality: parallax_over_error, ruwe, astrometric_chi2_al, astrometric_excess_noise, visibility_periods_used

Photometry: phot_g_mean_mag

Excluded (leakage risk):

mass_ratio and any derived mass fields

Labeling strategy:

Cross-match with NASA Exoplanet Archive confirmed planets:

Direct via Gaia source_id

Fallback: 2.0′′ cone (RA/Dec)

Result:
Positives: 16 confirmed hosts • Negatives: 139,633 (mostly binaries)
Imbalance: ~0.01%

Balanced training subset:

All 16 positives + 800 negatives (≈50:1) ⇒ 816 rows

Stratified 75/25 split ⇒ test: 204 rows (4 pos, 200 neg)

Model: Random Forest
Config:

n_estimators = 400
max_depth = 12
class_weight = "balanced_subsample"
random_state = 42
Test (204 samples, τ = 0.557 F1-optimal):

               Predicted
             No   | Planet
Actual No   200   |   0
Actual Yes    1   |   3
ROC-AUC: 0.994 • PR-AUC: 0.861

Accuracy: 99.5% • Precision: 1.00 • Recall: 0.75 • Specificity: 1.00

Top features:

period (28%)

eccentricity (19%)

parallax_over_error (16%)

inclination (12%)

phot_g_mean_mag (9%)

Technical innovation: angle sin/cos expansion for
ω
ω continuity; cross-archive integration (TAP + positional fallback) with Astropy validation.

4. Direct Imaging (High-Contrast Coronagraphy)
Physical Principle
All previous methods are indirect; direct imaging captures photons from the planet itself.

Contrast challenge:
Visible: Jupiter reflects $\sim 10^{-9}$ of solar light.
$\sim 10^{-9}$
$\sim 10^{-9}$
  of solar light.
Near-IR (young planets <100 Myr): thermal emission $\sim 10^{-6}$ contrast.
$\sim 10^{-6}$
$\sim 10^{-6}$
  contrast.

Solution: Extreme AO + Coronagraph + Post-processing (ADI).

ExAO: deformable mirror, ~kHz corrections

Coronagraph: blocks starlight, creates “dark hole”

Post-processing: subtract speckle patterns (reference star or ADI)

### Why Direct Imaging is Unique

| Property | Other Methods | Direct Imaging |
|---|---|---|
| Information | Indirect (gravity/occultation) | **Direct photons** (imaging) |
| Spectroscopy | Limited | **Yes** (atmospheric composition) |
| Orbit constraint | Various | Wide separations ($\gtrsim$ 5 AU, ground) |
| Target age | Any | Young (< 1 Gyr; self-luminous) |

**Key advantage:** Atmospheric spectroscopy of planet light reveals H\(_2\)O, CH\(_4\), CO absorption → temperature, chemistry, and cloud structure.

### Dataset & Preprocessing

**Primary source:** VLT/SPHERE (IRDIS, near‐IR)

**Data:** FITS cubes with shape `(n_frames, H, W)`; typically 50–200 frames per target.

PCA-ADI workflow:

Frame alignment

Centroid via 2D Gaussian

Sub-pixel registration

Rotate to north-up

PCA speckle subtraction

python
# cube: (n_frames, H, W)
X = cube.reshape(n_frames, -1)  # (n_frames, H*W)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
K = 10  # number of PCs
PSF_model = U[:, :K] @ np.diag(S[:K]) @ Vt[:K, :]
residual = (X - PSF_model).reshape(n_frames, H, W)
Why it works: Star dominates first PCs; planet remains in residuals.

GPU accel: pca_adi_gpu_exact() (PyTorch) for ~10× speedup.

Frame stacking

Median combine, sigma-clip outliers

Generate SNR map

Synthetic planet injection (for training)

Point sources w/ known positions/fluxes

PSF convolution

Calibrate SNR distribution

Model: CNN → Transformer Hybrid
Baseline (ExoplanetCNN):

Conv2D blocks (1→32→64→128) + ReLU + MaxPool

Flatten → Dense(256) → Dropout(0.3) → Dense(1) + Sigmoid

Advanced (CNNTransformerExoplanet):

**CNN features → reshape to sequence:** $(H/8 \cdot W/8, 128)$


Transformer encoder (8 heads, learnable positional enc.)

Global avg pool → Dense(1) + Sigmoid

Why transformer: long-range spatial context, focuses attention on faint point sources; robust to ADI artifacts.

Training: BCEWithLogits, AdamW (1e-4, wd=1e-5), Dropout(0.3), early stop, batch 16.
Eval: PR-AUC, FPR at fixed recall, detection limit vs SNR.

Performance (validation):

Precision: >0.90 @ SNR ≥ 5

Recall: >0.85 for injected planets

FPR: <5% in empty regions

Ablation: CNN 87% → CNN+Transformer 92% (+5%).

Tech innovation: GPU-accelerated PCA; attention map visualization for interpretability.

Summary: Scientific Impact
Parameter Space Coverage
Orbital separation:

Transit/RV: 0.01–1 AU (hot/warm)

Astrometry: 1–10 AU (cold giants)

Microlensing: 1–10 AU (cold planets; any distance)

Direct Imaging: >5 AU (wide, young)

Host star distance:

Transit/RV: <1 kpc

Astrometry: <100 pc (Gaia)

Microlensing: <10 kpc (bulge)

Direct Imaging: <150 pc (young assoc.)

Information obtained:

Transit: radius, period

RV: $M_p \sin i$, period  
Astrometry: true $M_p$, orbit

Microlensing: existence, rough mass

Direct Imaging: spectrum, luminosity, atmosphere

Validation Strategy (Future Work)
Cross-validation with known systems:

HR 8799 (direct imaging)

51 Peg b (RV)

TrES-2b (transit + RV)

Blind challenges:

NASA Roman microlensing challenge

ESA PLATO mock data

Gaia DR4 astrometric detections (when released)

Computational Requirements
Training (NVIDIA T4):

Transit ensemble: 45 min

RV LightGBM: 8 min

Microlensing Conv1D+GRU: 25 min

Astrometry RF: 5 min

Direct imaging CNN: 90 min

Inference throughput:

Transit: 10,000 LC/hr

RV: $M_p \sin i$, period

Microlensing: 8,000 events/hr

Astrometry: 50,000 sources/hr

Direct imaging: 100 cubes/hr

Total training set:
~21,000 transit + 800 RV + 293 microlensing + 816 astrometry + ~500 imaging ≈ 23,400 labeled examples

References
Radial Velocity
Mayor & Queloz (1995). A Jupiter-mass companion to a solar-type star. Nature 378, 355–359.

HARPS Consortium — https://www.eso.org/sci/facilities/lasilla/instruments/harps.html

Butler et al. (1996). Attaining Doppler precision of 3 m/s. PASP 108, 500.

Microlensing
Mao & Paczyński (1991). Gravitational microlensing by double stars and planetary systems. ApJ 374, L37.

Gould & Loeb (1992). Discovering planetary systems through gravitational microlenses. ApJ 396, 104.

Microlensing Data Challenge — https://github.com/microlensing-data-challenge

Astrometry
Perryman et al. (2014). Astrometric exoplanet detection with Gaia. ApJ 797, 14.

Gaia Collaboration (2023). Gaia Data Release 3. A&A 674, A1.

NASA Exoplanet Archive — https://exoplanetarchive.ipac.caltech.edu/

Direct Imaging
Beuzit et al. (2019). SPHERE: The exoplanet imager for the Very Large Telescope. A&A 631, A155.

Marois et al. (2008). Direct imaging of multiple planets orbiting HR 8799. Science 322, 1348.

VIP (Vortex Image Processing): Gomez Gonzalez et al. (2017). AJ 154, 7.
