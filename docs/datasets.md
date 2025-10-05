# Datasets

This document provides detailed information about all seven datasets used across the five detection methods in our multi-method exoplanet detection framework.

---

## Transit Photometry Datasets (3 datasets)

### 1. Kepler Objects of Interest (KOI)

**Mission**: NASA Kepler Space Telescope (2009-2018)
**Observatory**: Dedicated photometric transit survey
**Field of view**: 115 square degrees in Cygnus-Lyra region

**Data characteristics:**
- Targets: 150,000+ stars monitored continuously
- Temporal coverage: 4 years of nearly uninterrupted observations
- Cadence: Long cadence (29.4 minutes), short cadence (58.85 seconds) for select targets
- Photometric precision: ~20 ppm for 12th magnitude stars (30-minute integration)
- Confirmed planets: 2,700+ validated exoplanets

**Data format:**
- FITS light curve files with flux vs. time
- Systematic error correction flags (SAP_FLUX, PDCSAP_FLUX)
- Data validation flags for transits and false positives

**Features used:**
- Detrended light curves (PDCSAP_FLUX)
- Transit depth, duration, period, epoch
- Stellar parameters: Teff, R★, M★, log(g), [Fe/H]
- Data quality flags and systematic corrections

**Access:**
- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- MAST Archive: https://archive.stsci.edu/kepler/

**Citation:**
- Borucki et al. (2010). "Kepler Planet-Detection Mission: Introduction and First Results." Science 327, 977.
- Thompson et al. (2018). "Planetary Candidates Observed by Kepler. VIII." ApJS 235, 38.

---

### 2. TESS Objects of Interest (TOI)

**Mission**: NASA Transiting Exoplanet Survey Satellite (2018-present)
**Observatory**: All-sky photometric transit survey
**Coverage**: 85% of sky divided into 26 sectors

**Data characteristics:**
- Targets: 200,000+ bright stars (4 < V < 13 mag)
- Temporal coverage: 27 days per sector (extended to 1 year for continuous viewing zones)
- Cadence: 2-minute (20,000 targets), 30-minute (200,000 targets), 20-second (1,000 targets, Cycle 3+)
- Photometric precision: ~60 ppm for 9th magnitude stars (1-hour integration)
- Confirmed planets: 500+ validated (rapidly growing)

**Data format:**
- FITS light curve files (similar to Kepler structure)
- SPOC (Science Processing Operations Center) pipeline products
- Quality flags for momentum desaturation events, cosmic rays

**Features used:**
- SAP and PDCSAP light curves
- Transit parameters from Data Validation reports
- TIC (TESS Input Catalog) stellar parameters
- Sectors observed, data quality metrics

**Advantages over Kepler:**
- Brighter host stars (better for follow-up spectroscopy)
- All-sky coverage (diverse stellar populations)
- Shorter orbital periods emphasized (observing strategy)

**Access:**
- MAST TESS Archive: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
- ExoFOP-TESS: https://exofop.ipac.caltech.edu/tess/

**Citation:**
- Ricker et al. (2015). "Transiting Exoplanet Survey Satellite (TESS)." JATIS 1, 014003.
- Guerrero et al. (2021). "The TESS Objects of Interest Catalog." ApJS 254, 39.

---

### 3. K2 Planets and Candidates

**Mission**: Kepler Extended Mission (K2, 2014-2018)
**Observatory**: Repurposed Kepler spacecraft after reaction wheel failure
**Coverage**: 20 campaigns across ecliptic plane

**Data characteristics:**
- Targets: 500,000+ stars across diverse fields
- Temporal coverage: ~80 days per campaign
- Cadence: Long cadence (29.4 minutes), short cadence (58.85 seconds)
- Photometric precision: Degraded vs. prime Kepler (~100 ppm) due to spacecraft roll
- Confirmed planets: 500+ validated

**Data format:**
- FITS light curve files with systematic correction
- Campaign-specific pointing and roll corrections
- EVEREST/K2SFF/K2SC pipeline products (motion-corrected photometry)

**Features used:**
- Motion-corrected light curves
- Transit parameters
- EPIC (Ecliptic Plane Input Catalog) stellar parameters
- Campaign number, detector position

**Unique contributions:**
- Young stellar clusters (Pleiades, Hyades, Praesepe)
- Eclipsing binary systems
- Different Galactic environments (vary stellar age, metallicity)

**Access:**
- MAST K2 Archive: https://archive.stsci.edu/k2/
- K2 Campaign pages: https://keplerscience.arc.nasa.gov/k2-observing.html

**Citation:**
- Howell et al. (2014). "The K2 Mission: Characterization and Early Results." PASP 126, 398.
- Crossfield et al. (2016). "197 Candidates and 104 Validated Planets in K2's First Five Fields." ApJS 226, 7.

---

### Combined Transit Dataset Statistics

**Total labeled training samples**: 21,000 light curves
- Positive class (confirmed planets): ~2,800
- Negative class (false positives, non-planets): ~18,200
- Class imbalance ratio: 1:6.5

**Preprocessing pipeline:**
1. Download FITS files via astroquery/lightkurve
2. Extract PDCSAP_FLUX (systematics-corrected flux)
3. Remove flagged bad data (quality flags)
4. Normalize to median = 1.0
5. Phase-fold on known/suspected period
6. Extract features: depth, duration, shape, SNR
7. Join with stellar parameters from TIC/KIC/EPIC catalogs
8. Apply KNN imputation for missing stellar parameters (k=5)
9. Stratified train/test split (75/25)

**Storage requirements**: ~15 GB for processed light curves + metadata

---

## Radial Velocity Datasets (2 datasets)

### 4. HARPS Radial Velocity Archive

**Instrument**: High Accuracy Radial velocity Planet Searcher
**Observatory**: ESO 3.6m telescope, La Silla, Chile
**Wavelength range**: 378-691 nm (visible, cross-dispersed echelle)
**Resolving power**: R ≈ 115,000

**Data characteristics:**
- Targets: 1,000+ stars with multi-epoch spectroscopy
- Temporal baseline: 2003-present (20+ years)
- Velocity precision: ~1 m/s (state-of-the-art as of 2003)
- Cadence: Varies by program (hours to years between observations)
- Confirmed planets: 200+ discoveries

**Data format:**
- ASCII tables via ESO archive
- Columns: BJD, RV (m/s), RV_err, FWHM, contrast, bisector span, S-index (activity)
- Pipeline: HARPS Data Reduction Software (DRS)

**Features used:**
- Time-series velocity measurements
- Measurement uncertainties
- Activity indicators (correlate with stellar noise)
- Multi-season coverage (enables long-period planet detection)

**Access:**
- ESO Science Archive: http://archive.eso.org/wdb/wdb/adp/phase3_spectral/form
- RV Bank compilation: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/636/A74

**Citation:**
- Mayor et al. (2003). "Setting New Standards with HARPS." The Messenger 114, 20.
- Trifonov et al. (2020). "The HARPS search for southern extra-solar planets. XLIV." A&A 636, A74.

---

### 5. HIRES Radial Velocity Archive

**Instrument**: High Resolution Echelle Spectrometer
**Observatory**: Keck I 10m telescope, Mauna Kea, Hawaii
**Wavelength range**: 300-1000 nm (customizable)
**Resolving power**: R ≈ 60,000

**Data characteristics:**
- Targets: 500+ planet search stars
- Temporal baseline: 1996-present (28+ years)
- Velocity precision: ~2-3 m/s (upgraded iodine cell: ~1 m/s)
- Cadence: Varies by program
- Confirmed planets: 100+ discoveries (including 51 Peg b confirmation)

**Data format:**
- ASCII tables from Keck Observatory Archive (KOA)
- Columns: JD, RV (m/s), RV_err, S_HK (activity), instrument config
- Pipeline: HIRES reduction pipeline

**Features used:**
- Velocity time-series
- Uncertainties
- Long-term trends (secular acceleration)
- Cross-calibration with HARPS for overlapping targets

**Access:**
- Keck Observatory Archive: https://www2.keck.hawaii.edu/koa/public/koa.php
- California Planet Search: http://exoplanets.org/

**Citation:**
- Vogt et al. (1994). "HIRES: the high-resolution echelle spectrometer on the Keck 10-m Telescope." SPIE 2198, 362.
- Butler et al. (2017). "The California Planet Search. I." AJ 153, 208.

---

### Combined RV Dataset Statistics

**Total stars**: 1,200+ (after quality filtering)
- Labeled positives (confirmed planet hosts): 150
- Labeled negatives (non-detections, noise): 1,050
- Balanced training set: 300 stars (150 pos + 150 neg sampled)

**Preprocessing pipeline:**
1. Download per-star time-series from ESO/KOA
2. Outlier rejection: MAD clipping (|z| > 7σ)
3. Detrending: subtract linear/polynomial secular trends
4. Per-star normalization: z-score standardization
5. Feature extraction:
   - Lomb-Scargle periodogram (top 3 peaks)
   - Velocity statistics (mean, std, range, semi-amplitude)
   - Phase-folded variance at dominant period
6. Join with stellar parameters (M★, Teff, [Fe/H])
7. Stratified group split by star (prevent leakage)

**Storage requirements**: ~500 MB (time-series + features)

---

## Microlensing Dataset

### 6. Roman Space Telescope (WFIRST) 2018 Microlensing Challenge

**Mission**: Synthetic Galactic bulge survey simulation
**Purpose**: Algorithm development for future Roman microlensing campaign
**Release**: 2018 Microlensing Data Challenge

**Data characteristics:**
- Events: 293 gravitational microlensing light curves
- Filters: W149 (1.5 μm), Z087 (0.9 μm) dual-band photometry
- Temporal coverage: Weeks to months (event duration)
- Cadence: Minutes to hours (simulated rapid follow-up)
- Photometric precision: Realistic Poisson noise + systematic errors

**Data format:**
- Plain text files (3 columns: time, magnitude, error)
- Paired files per event: eventID_W149.txt, eventID_Z087.txt
- Metadata: event_info.txt (peak time, baseline magnitude)

**Ground truth:**
- Challenge dataset lacks planet labels (intentional for blind testing)
- We generated synthetic labels via Gaussian bump injection for proof-of-concept

**Features used:**
- Dual-channel brightness time-series
- Time-aligned W149 and Z087 measurements
- Magnification profile shape
- Chromatic effects (wavelength-dependent microlensing)

**Access:**
- GitHub repository: https://github.com/microlensing-data-challenge/data-challenge-1
- Challenge description: https://wfirst.ipac.caltech.edu/sims/Microlensing_Data_Challenge.html

**Supplementary data (production systems would use):**
- OGLE Early Warning System: http://ogle.astrouw.edu.pl/ogle4/ews/ews.html
- MOA Alerts: https://www.massey.ac.nz/~iabond/moa/alerts/

**Citation:**
- Penny et al. (2019). "Predictions of the WFIRST Microlensing Survey." ApJS 241, 3.
- Microlensing Data Challenge Committee (2018). Challenge specifications.

---

### Microlensing Dataset Statistics

**Total events**: 293
- Two-channel (W149 + Z087): 293 events
- Single-channel: 0 (all events have both filters)
- Synthetic positives injected: ~35% (102/293)

**Preprocessing pipeline:**
1. Parse text files (BJD, mag, err)
2. Robust outlier clipping (7σ MAD threshold)
3. Magnitude to flux conversion: F ∝ 10^(-0.4 × mag)
4. Time window alignment (overlap W149 and Z087 spans)
5. Uniform resampling: linear interpolation to 512 points
6. Per-channel z-score normalization
7. Output: (293, 512, 2) tensor

**Storage requirements**: ~50 MB (processed tensors)

---

## Astrometry Dataset

### 7. Gaia DR3 Non-Single-Star (NSS) Two-Body Orbits

**Mission**: ESA Gaia astrometric survey (2014-present)
**Purpose**: Precision parallax and proper motion for 1.8 billion sources
**Data Release**: DR3 (June 2022)

**Data characteristics:**
- NSS catalog size: 168,065 two-body orbital solutions
- After quality cuts: 139,649 solutions used
- Astrometric precision: ~20-30 μas for bright stars (G < 15 mag)
- Temporal baseline: 34 months (DR3), 66 months (DR4, upcoming)

**Quality cuts applied:**
- ruwe < 1.4 (Goodness-of-fit for single-star model)
- parallax_over_error > 10 (Distance SNR threshold)
- visibility_periods_used >= 8 (Minimum observation epochs)

**Data format:**
- FITS tables from Gaia Archive
- Joined tables: gaiadr3.nss_two_body_orbit + gaiadr3.gaia_source
- Columns: period, eccentricity, inclination, arg_periastron, parallax, RUWE, astrometric_chi2_al, etc.

**Labels:**
- Cross-matched with NASA Exoplanet Archive confirmed planets
- Direct match: Gaia source_id in NASA gaia_id
- Positional fallback: 2.0 arcsecond cone search (RA/Dec)
- Result: 16 confirmed planet hosts, 139,633 other companions (mostly binaries)

**Features used (23 total):**
- Orbital: period, eccentricity, inclination, sin(ω), cos(ω)
- Quality: parallax_over_error, RUWE, chi2, excess_noise, visibility_periods
- Photometry: phot_g_mean_mag
- Excluded: mass_ratio (label leakage)

**Access:**
- Gaia Archive: https://gea.esac.esa.int/archive/
- TAP query interface: https://gea.esac.esa.int/tap-server/tap
- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/

**Citation:**
- Gaia Collaboration (2023). "Gaia Data Release 3: Summary of the content and survey properties." A&A 674, A1.
- Gaia Collaboration (2023). "Gaia Data Release 3: Non-single stars." A&A 674, A34.

---

### Astrometry Dataset Statistics

**Full catalog**: 139,649 orbital solutions

**Labeled subset:**
- Positives (planet hosts): 16
- Negatives (sampled): 800
- Balanced training set: 816 rows (extreme 1:50000 imbalance mitigated)
- Train/test split: 612 train / 204 test (stratified)

**Preprocessing pipeline:**
1. ADQL query to Gaia Archive with quality filters
2. Cross-match with NASA Exoplanet Archive (TAP query)
3. Trigonometric expansion: arg_periastron to [sin(ω), cos(ω)]
4. NaN filling: 0.0 for missing values
5. Feature standardization: z-score for tree models (optional)
6. Class balancing: undersample negatives

**Storage requirements**: ~200 MB (full catalog), ~1 MB (balanced subset)

---

## Direct Imaging Dataset

### VLT/SPHERE High-Contrast Imaging Archive

**Instrument**: Spectro-Polarimetric High-contrast Exoplanet REsearch (SPHERE)
**Observatory**: ESO Very Large Telescope (VLT), Paranal, Chile
**Sub-instruments**: IRDIS (infrared dual-band imager), IFS (integral field spectrograph)

**Data characteristics:**
- Targets: ~500 young nearby stars (age < 1 Gyr, distance < 150 pc)
- Wavelength: Near-infrared (Y, J, H, K bands: 0.95-2.3 μm)
- Angular resolution: ~20 mas (diffraction-limited with adaptive optics)
- Contrast: 10^-6 at 0.5" separation (coronagraphic mode)

**Data format:**
- FITS image cubes: (n_frames, height, width)
- Typical dimensions: 50-200 frames × 1024 × 1024 pixels
- Metadata: parallactic angles (for ADI), filter, exposure time

**Observations:**
- Angular Differential Imaging (ADI): field rotation over hours
- Reference star observations (for PSF subtraction)
- Coronagraph settings: apodized pupil Lyot coronagraph (APLC)

**Ground truth:**
- Few confirmed directly imaged planets (HR 8799 b/c/d/e, β Pic b, 51 Eri b)
- Synthetic planet injections used for training:
  - Point sources with known positions, fluxes, SNR
  - Convolved with instrumental PSF
  - Added to real empty observations

**Features used:**
- PCA-ADI processed frames (10 components subtracted)
- Residual images after stellar speckle removal
- SNR maps (signal-to-noise per pixel)

**Access:**
- ESO Science Archive Portal: http://archive.eso.org/scienceportal/home
- SPHERE Data Center: https://sphere.osug.fr/
- Published datasets: Zenodo repositories (e.g., https://zenodo.org/records/2815298)

**Citation:**
- Beuzit et al. (2019). "SPHERE: the exoplanet imager for the Very Large Telescope." A&A 631, A155.
- Vigan et al. (2021). "The SPHERE infrared survey for exoplanets (SHINE). III." A&A 651, A72.

---

### Direct Imaging Dataset Statistics

**Training data construction:**
- Real observations: ~100 target stars (empty fields after PCA)
- Synthetic planets injected: ~500 point sources at SNR in [3, 10]
- Negatives: Empty regions, stellar residuals
- Total training samples: ~5,000 image patches (64×64 pixels)

**Preprocessing pipeline:**
1. Read FITS cube
2. Frame alignment: 2D Gaussian centroid fitting
3. PCA-ADI: Subtract first K=10 principal components (stellar PSF)
4. Frame stacking: median combination
5. Normalization: per-image z-score
6. Synthetic injection: add point sources with realistic PSF
7. Patch extraction: 64×64 crops centered on planet/empty regions
8. Augmentation: rotations, flips, noise injection

**Storage requirements**: ~5 GB (processed cubes + patches)

---

## Summary Table

| Dataset | Mission/Instrument | Samples | Detection Method | Temporal Baseline | Access |
|---------|-------------------|---------|------------------|-------------------|--------|
| Kepler KOI | NASA Kepler | 150k stars, 2.7k planets | Transit | 4 years | exoplanetarchive.ipac.caltech.edu |
| TESS TOI | NASA TESS | 200k stars, 500+ planets | Transit | 27 days/sector | mast.stsci.edu |
| K2 | Kepler Extended | 500k stars, 500+ planets | Transit | 80 days/campaign | archive.stsci.edu/k2 |
| HARPS RV | ESO 3.6m | 1k+ stars, 200+ planets | Radial Velocity | 20+ years | archive.eso.org |
| HIRES RV | Keck I 10m | 500+ stars, 100+ planets | Radial Velocity | 28+ years | koa.ipac.caltech.edu |
| Roman Microlensing | Simulation | 293 events | Microlensing | Weeks-months | github.com/microlensing-data-challenge |
| Gaia DR3 NSS | ESA Gaia | 140k orbits, 16 planets | Astrometry | 34 months | gea.esac.esa.int |
| VLT/SPHERE | ESO VLT | ~100 stars, synthetic | Direct Imaging | Hours (ADI) | archive.eso.org |

---

## Data Availability Statement

All datasets used in this project are publicly available through the cited archives. Processed data products, feature tables, and model training scripts are provided in this repository for full reproducibility.

For questions about data access or preprocessing pipelines, consult the individual method documentation or contact the repository maintainers.
