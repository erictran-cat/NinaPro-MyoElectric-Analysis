# NinaPro sEMG Pipeline: Comprehensive Technical Documentation

## Table of Contents
1. [File Overview](#1-file-overview)
2. [main_pipeline.m - Master Orchestration Script](#2-main_pipelinem---master-orchestration-script)
3. [feature_extraction.m - Feature Engineering Module](#3-feature_extractionm---feature-engineering-module)
4. [signal_processing.m - Signal Conditioning Module](#4-signal_processingm---signal-conditioning-module)
5. [visualization.m - Clinical Visualization Suite](#5-visualizationm---clinical-visualization-suite)
6. [ANALYTICAL_REFLECTION.md - Research Documentation](#6-analytical_reflectionmd---research-documentation)
7. [Visual Aid Units Reference](#7-visual-aid-units-reference)

---

## 1. File Overview

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `main_pipeline.m` | Orchestrates entire workflow | NinaPro .mat file | Feature matrix, figures, .mat results |
| `feature_extraction.m` | Computes EMG features | Segmented EMG windows | Feature vectors |
| `signal_processing.m` | Filters and normalizes signals | Raw EMG | Clean, normalized EMG |
| `visualization.m` | Generates clinical plots | Features + labels | PNG/PDF figures |
| `ANALYTICAL_REFLECTION.md` | Documents physiological interpretations | N/A | Reference document |

---

## 2. main_pipeline.m - Master Orchestration Script

### Purpose
This is the **entry point** that coordinates all processing stages. It loads data, calls processing functions, extracts features, and generates visualizations.

### Processing Stages

#### Stage 1: Configuration (Lines 15-46)
```matlab
config.fs = 2000;           % Sampling frequency in Hertz (samples/second)
config.filter_low = 20;     % High-pass cutoff in Hertz
config.filter_high = 450;   % Low-pass cutoff in Hertz
config.notch_freq = 60;     % Powerline frequency in Hertz (60 Hz USA, 50 Hz Europe)
config.window_ms = 200;     % Analysis window in milliseconds
config.stride_ms = 50;      % Window step size in milliseconds
```

**What it does:** Sets all parameters that control the analysis. The sampling frequency (fs) must match your NinaPro database version.

#### Stage 2: Data Loading (Lines 48-59)
**What it does:** Loads the four key variables from NinaPro .mat files:

| Variable | Description | Dimensions | Units |
|----------|-------------|------------|-------|
| `emg` | Raw EMG voltage readings | [samples × channels] | Volts (V) or millivolts (mV) |
| `stimulus` | Cue shown to subject | [samples × 1] | Integer (0 = rest, 1-17 = gesture ID) |
| `restimulus` | Actual movement label (delayed) | [samples × 1] | Integer (same as stimulus) |
| `repetition` | Trial repetition number | [samples × 1] | Integer (1-6 typically) |

**Why `restimulus` over `stimulus`?** When a visual cue appears, there's a 150-250ms delay before muscles actually activate (reaction time). `restimulus` is manually corrected by NinaPro researchers to align with actual muscle activity onset.

#### Stage 3: Pre-processing (Lines 61-76)
**What it does:** Cleans the raw EMG signal through three steps:

1. **Bandpass Filtering (20-450 Hz)**
   - Removes: DC drift (<20 Hz), motion artifacts (<20 Hz), high-frequency noise (>450 Hz)
   - Preserves: Motor Unit Action Potentials (MUAPs) which occur in 20-500 Hz range
   - Method: 4th-order Butterworth filter applied with `filtfilt` (zero-phase, no time delay)

2. **Notch Filtering (60 Hz)**
   - Removes: Powerline interference (electrical hum from nearby equipment)
   - Method: Narrow bandstop filter at 58-62 Hz

3. **MAD Normalization**
   - Formula: `x_normalized = (x - median(x)) / MAD(x)`
   - Where MAD = Median Absolute Deviation = `median(|x - median(x)|)`
   - Why MAD instead of standard deviation? MAD is **robust to outliers** (electrode pops, motion artifacts)
   - Output units: **Dimensionless** (normalized units, typically ranging from -5 to +5)

#### Stage 4: Segmentation (Lines 78-92)
**What it does:** Chops the continuous EMG signal into overlapping windows for analysis.

```
Signal: |-----------------------------------------------|
Window 1: |======|
Window 2:    |======|
Window 3:       |======|
         ↑___↑ = stride (50ms)
         |======| = window (200ms)
```

- **Window length:** 200ms = 400 samples at 2000 Hz
- **Stride:** 50ms = 100 samples
- **Overlap:** 75% (each sample appears in ~4 windows)
- **Label assignment:** Mode (most frequent) label within each window

**Output:** 3D array `[n_segments × window_length × n_channels]`

#### Stage 5: Feature Extraction (Lines 94-106)
Calls `extract_features_vectorized()` to compute 6 features per channel.

**Output:** MATLAB Table with columns like `MAV_Ch1`, `RMS_Ch1`, ... `TP_Ch12`

#### Stage 6: Signal Quality Analysis (Lines 108-119)
Computes Signal-to-Noise Ratio (SNR) per channel:
```
SNR = 10 × log₁₀(Power_during_gestures / Power_during_rest)
```
- **Units:** Decibels (dB)
- **Interpretation:** SNR > 10 dB = good, SNR < 3 dB = bad electrode

#### Stage 7: Visualization (Lines 121-131)
Generates four publication-quality figures (detailed in Section 5).

#### Stage 8: Gesture Distinctiveness (Lines 133-148)
Computes which gestures are most separable in feature space using Euclidean distance between class centroids.

---

## 3. feature_extraction.m - Feature Engineering Module

### Purpose
Converts raw EMG windows into numerical descriptors (features) that machine learning classifiers can use.

### Features Extracted

#### Time Domain Features (computed directly from voltage samples)

| Feature | Formula | Units | Physiological Meaning |
|---------|---------|-------|----------------------|
| **MAV** (Mean Absolute Value) | `(1/N) × Σ|xᵢ|` | Normalized units (dimensionless after MAD normalization) | Average muscle activation level; proportional to contraction intensity |
| **RMS** (Root Mean Square) | `√[(1/N) × Σxᵢ²]` | Normalized units | Signal energy/power; related to force output |
| **WL** (Waveform Length) | `Σ|xᵢ₊₁ - xᵢ|` | Normalized units | Signal complexity; indicates motor unit recruitment density |
| **WAMP** (Willison Amplitude) | Count of `|xᵢ₊₁ - xᵢ| > threshold` | Count (integer) | Number of motor unit firings; indicates firing rate |

#### Frequency Domain Features (computed from Power Spectral Density)

| Feature | Formula | Units | Physiological Meaning |
|---------|---------|-------|----------------------|
| **MNP** (Mean Power Frequency) | `Σ(fᵢ × PSDᵢ) / Σ(PSDᵢ)` | Hertz (Hz) | Center of mass of frequency spectrum; shifts with fatigue |
| **TP** (Total Power) | `Σ(PSDᵢ)` | (Normalized units)²/Hz | Overall spectral energy; related to contraction strength |

### How Frequency Features are Computed

```matlab
[psd, f] = pwelch(signal, [], [], nfft, fs);
```

1. **Welch's Method:** Divides signal into overlapping sub-segments
2. **FFT:** Computes frequency content of each sub-segment
3. **Averaging:** Reduces noise by averaging periodograms
4. **Output:** Power Spectral Density (PSD) in units²/Hz vs frequency (Hz)

### Feature Matrix Structure

For 12 channels × 6 features = 72 columns:
```
| MAV_Ch1 | RMS_Ch1 | WL_Ch1 | WAMP_Ch1 | MNP_Ch1 | TP_Ch1 | MAV_Ch2 | ... | TP_Ch12 |
|---------|---------|--------|----------|---------|--------|---------|-----|---------|
|  0.234  |  0.312  | 45.2   |    23    |  127.3  | 0.089  |  0.198  | ... |  0.045  |
|  0.567  |  0.621  | 89.1   |    45    |  134.5  | 0.156  |  0.445  | ... |  0.112  |
```

Each row = one 200ms window (segment)
Each column = one feature from one channel

---

## 4. signal_processing.m - Signal Conditioning Module

### Purpose
Contains standalone functions for signal cleaning that can be used independently of the main pipeline.

### Key Functions

#### `design_filter_bank_extended(config)`
Creates digital filter coefficients:
- **Bandpass:** Butterworth IIR filter
- **Notch:** Butterworth bandstop filter

#### `apply_filter_bank(emg, filter_bank)`
Applies filters using `filtfilt()`:
- **Zero-phase filtering:** Filters forward, then backward
- **Result:** No phase distortion (critical for preserving MUAP timing)

#### `apply_mad_normalization_extended(emg)`
**Why MAD normalization?**

| Problem | Standard Z-score | MAD Normalization |
|---------|-----------------|-------------------|
| Electrode pop (1000× normal amplitude) | Severely skews mean & std | Median/MAD barely affected |
| Impedance variation between channels | Different scales | Unified scale |
| Between-subject comparison | Not comparable | Comparable |

**Formula:**
```
MAD = median(|x - median(x)|)
x_norm = (x - median(x)) / MAD
```

#### `compute_quality_metrics(emg, stimulus, config)`
Comprehensive quality assessment:

| Metric | What it Measures | Good Value | Bad Value |
|--------|------------------|------------|-----------|
| SNR | Signal vs noise power | > 10 dB | < 3 dB |
| PLI Ratio | 60 Hz contamination | < 0.05 | > 0.10 |
| Baseline Power | Low-frequency drift | < 0.10 | > 0.20 |
| Clipping Ratio | ADC saturation | < 0.001 | > 0.01 |

#### `remove_motion_artifacts(emg, config)`
Detects and interpolates artifact regions:
1. Compute signal envelope using Hilbert transform
2. Flag samples where envelope > 5× median
3. Interpolate flagged regions using piecewise cubic interpolation

---

## 5. visualization.m - Clinical Visualization Suite

### Purpose
Generates four publication-ready figures for clinical and engineering analysis.

---

### Figure 1: Spatial Activation Map

**What it shows:** A 2D heatmap where:
- **Y-axis:** Gesture classes (G01, G02, ... G17)
- **X-axis:** Electrode channels (Ch1, Ch2, ... Ch12)
- **Color:** Average RMS activation intensity

**Units:**
- Color scale: **Normalized RMS (0 to 1)**, where 1 = maximum activation across all gestures/channels

**How to interpret:**
- Bright cells = high muscle activation
- Each row shows which muscles activate for that gesture
- Patterns across rows reveal **muscle synergies** (coordinated muscle groups)

**Example interpretation:**
```
Gesture 3 (Wrist Flexion): Ch1, Ch2 bright → Flexor muscles active
Gesture 6 (Wrist Extension): Ch6, Ch7 bright → Extensor muscles active
```

**Clinical use:** Identifies which electrodes are most informative for each gesture; helps optimize electrode placement.

---

### Figure 2: PCA Class Separability Analysis

**What it shows:** 
- **Main plot (3D scatter):** Each dot = one 200ms segment, colored by gesture class
- **Axes:** First three Principal Components (PC1, PC2, PC3)
- **Top-right subplot:** Scree plot (variance explained per component)
- **Bottom-right subplot:** Inter-class distance matrix

**Units:**
- PC axes: **Dimensionless** (linear combinations of standardized features)
- Variance explained: **Percentage (%)**
- Distance matrix: **Euclidean distance** (dimensionless, in standardized feature space)

**How to interpret:**
- Well-separated clusters = gestures are distinguishable = good for classification
- Overlapping clusters = gestures are confusable = need better features or more channels
- Variance explained (PC1-3) > 80% = low-dimensional representation is sufficient

**What PCA does mathematically:**
1. Standardizes all 72 features to zero mean, unit variance
2. Finds directions of maximum variance (eigenvectors of covariance matrix)
3. Projects 72D data onto top 3 eigenvectors

**Clinical use:** Quickly assess if your feature set can support accurate gesture classification before training a full classifier.

---

### Figure 3: Signal Quality Metrics (SNR Bar Chart)

**What it shows:** Bar chart of Signal-to-Noise Ratio per electrode channel

**Units:** 
- Y-axis: **Decibels (dB)**
- Formula: `SNR = 10 × log₁₀(σ²_active / σ²_rest)`

**Color coding:**
- Green bars: SNR ≥ 10 dB (good quality)
- Yellow bars: 3 ≤ SNR < 10 dB (marginal)
- Red bars: SNR < 3 dB (bad - exclude from analysis)

**Reference lines:**
- Red dashed line at 3 dB: Minimum acceptable threshold
- Green dashed line at 10 dB: Target quality level

**How to interpret:**
- SNR = 10 dB means signal power is 10× noise power
- SNR = 3 dB means signal power is only 2× noise power
- SNR < 3 dB means noise dominates — electrode may be disconnected, poorly placed, or have high impedance

**Clinical use:** Identifies "bad" electrodes to exclude from prosthetic control; guides electrode repositioning.

---

### Figure 4: Feature Importance (Fisher Discriminant Ratio)

**What it shows:** Horizontal bar chart ranking features by discriminative power

**Units:**
- X-axis: **Fisher Discriminant Ratio (FDR)** — dimensionless
- Formula: `FDR = (Between-class variance) / (Within-class variance)`

**How to interpret:**
- Higher FDR = feature better separates gesture classes
- Top features should be prioritized in classifier design
- If frequency features dominate → spectral information is important
- If time features dominate → amplitude patterns are sufficient

**Pie chart (subplot):**
- Shows distribution of feature types among top 20 features
- Example: "60% time-domain, 40% frequency-domain"

**Clinical use:** Feature selection for real-time prosthetic control (fewer features = faster computation).

---

### Figure 5: Gesture Similarity Matrix (k-NN Confusion)

**What it shows:** Confusion matrix estimated using 5-Nearest Neighbor classification

**Units:**
- Color scale: **Classification rate (0 to 1)**, where 1 = 100% correct
- Diagonal elements: Correct classifications
- Off-diagonal elements: Confusions between gesture pairs

**How it's computed:**
1. For each segment, find 5 nearest neighbors in feature space (leave-one-out)
2. Predict gesture by majority vote
3. Tally predictions into confusion matrix
4. Normalize rows to sum to 1

**How to interpret:**
- Bright diagonal = good separation
- Bright off-diagonal cells = frequently confused gesture pairs
- Example: Gestures 5 and 7 both bright in (5,7) and (7,5) cells → these gestures are similar

**Clinical use:** Identifies gesture pairs that may need to be merged or require additional sensors to distinguish.

---

## 6. ANALYTICAL_REFLECTION.md - Research Documentation

### Purpose
Provides physiological and engineering context for interpreting results.

### Contents

#### Section 1: Waveform Length Physiological Significance
Explains why WL correlates with motor intent:
- More motor units recruited → more zero-crossings → higher WL
- Faster firing rates → more rapid amplitude changes → higher WL
- Proportional to force without saturating like RMS

#### Section 2: Electrode Shift Problem
Analyzes what happens when the prosthetic socket rotates:
- Feature vectors become misaligned with training data
- Classification accuracy drops 5-10% per 10° rotation
- Mitigation strategies: rotation-invariant features, online adaptation

#### Section 3: Top 5 Most Distinct Gestures
Summary table showing which gestures are easiest to classify:
- Based on average Euclidean distance from other class centroids
- Higher distance = more distinct = more reliable for prosthetic control

---

## 7. Visual Aid Units Reference

### Quick Reference Table

| Figure | X-Axis Units | Y-Axis Units | Color Units |
|--------|--------------|--------------|-------------|
| Spatial Activation Map | Channel number (1-12) | Gesture ID (1-17) | Normalized RMS (0-1) |
| PCA 3D Scatter | PC1 (dimensionless) | PC2 (dimensionless) | Gesture class (categorical) |
| PCA Scree Plot | Component number | Variance (%) | N/A |
| Distance Matrix | Gesture ID | Gesture ID | Euclidean distance |
| SNR Bar Chart | Channel number | dB | Quality category (green/yellow/red) |
| Feature Importance | Fisher Ratio | Feature name | N/A |
| Confusion Matrix | Predicted gesture | Actual gesture | Classification rate (0-1) |

### Unit Conversions

| Quantity | Raw Units | After MAD Normalization |
|----------|-----------|------------------------|
| EMG amplitude | millivolts (mV) | Dimensionless (~-5 to +5) |
| MAV | mV | Dimensionless |
| RMS | mV | Dimensionless |
| WL | mV (cumulative) | Dimensionless |
| WAMP | Count | Count (unchanged) |
| MNP | Hz | Hz (unchanged) |
| TP | mV²/Hz | Dimensionless²/Hz |

### Decibel Reference

| SNR (dB) | Power Ratio | Interpretation |
|----------|-------------|----------------|
| 0 dB | 1:1 | Signal = Noise |
| 3 dB | 2:1 | Signal is 2× noise |
| 10 dB | 10:1 | Signal is 10× noise |
| 20 dB | 100:1 | Signal is 100× noise |

---

## Appendix A: NinaPro Database Quick Reference

| Database | Sampling Rate | Channels | Electrode Type | Gestures | Subjects |
|----------|---------------|----------|----------------|----------|----------|
| DB1 | 100 Hz | 10 | Otto Bock | 52 | 27 intact |
| DB2 | 2000 Hz | 12 | Delsys Trigno | 49 | 40 intact |
| DB3 | 2000 Hz | 12 | Delsys Trigno | 49 | 11 amputees |
| DB5 | 200 Hz | 16 | Myo Armband | 52 | 10 intact |
| DB6 | 2000 Hz | 14 | Delsys + Accelerometer | 49 | 10 intact |

**Important:** Always verify `config.fs` and `config.num_channels` match your database version!

---

## Appendix B: Complete NinaPro Gesture Descriptions

The NinaPro database organizes gestures into exercises. **DB2 and DB3** use Exercises B, C, and D (49 gestures total). **DB1** uses Exercises A, B, and C (52 gestures total).

### Exercise A: Basic Finger Movements (12 gestures)
*Used in DB1 only*

| Gesture ID | Movement Name | Description |
|------------|---------------|-------------|
| 1 | Index Flexion | Bend the index finger toward the palm |
| 2 | Index Extension | Straighten the index finger fully |
| 3 | Middle Flexion | Bend the middle finger toward the palm |
| 4 | Middle Extension | Straighten the middle finger fully |
| 5 | Ring Flexion | Bend the ring finger toward the palm |
| 6 | Ring Extension | Straighten the ring finger fully |
| 7 | Little Flexion | Bend the little (pinky) finger toward the palm |
| 8 | Little Extension | Straighten the little finger fully |
| 9 | Thumb Adduction | Move thumb toward the palm/index finger |
| 10 | Thumb Abduction | Move thumb away from the palm laterally |
| 11 | Thumb Flexion | Bend the thumb toward the palm |
| 12 | Thumb Extension | Straighten the thumb fully |

---

### Exercise B: Isometric/Isotonic Hand Configurations & Wrist Movements (17 gestures)
*Used in DB1, DB2, DB3 — This is the most commonly used exercise set*

#### Hand Configurations (Gestures 1-8)

| Gesture ID | Movement Name | Description | Muscles Involved |
|------------|---------------|-------------|------------------|
| 1 | Thumb Up | Closed fist with thumb pointing upward (hitchhiker gesture) | Extensor pollicis longus, Flexor digitorum |
| 2 | Index + Middle Extension | Extend index and middle fingers while others remain flexed ("peace sign" or pointing with two fingers) | Extensor indicis, Extensor digitorum |
| 3 | Ring + Little Flexion | Flex ring and little fingers while others remain extended | Flexor digitorum superficialis (ulnar side) |
| 4 | Thumb Opposition | Touch thumb tip to little finger tip (OK gesture without closing circle) | Opponens pollicis, Flexor pollicis brevis |
| 5 | Fingers Abduction | Spread all fingers apart as wide as possible | Dorsal interossei, Abductor digiti minimi |
| 6 | Fingers Adduction | Bring all fingers together, closing gaps | Palmar interossei |
| 7 | Fist / Hand Close | Close hand into a tight fist | Flexor digitorum superficialis & profundus |
| 8 | Pointing Index | Extend only index finger while making a fist with other fingers | Extensor indicis, Flexor digitorum |

#### Wrist Movements (Gestures 9-17)

| Gesture ID | Movement Name | Description | Muscles Involved |
|------------|---------------|-------------|------------------|
| 9 | Wrist Flexion | Bend wrist so palm moves toward forearm | Flexor carpi radialis, Flexor carpi ulnaris |
| 10 | Wrist Extension | Bend wrist so back of hand moves toward forearm | Extensor carpi radialis, Extensor carpi ulnaris |
| 11 | Wrist Radial Deviation | Tilt wrist toward thumb side (radial side) | Flexor carpi radialis, Extensor carpi radialis |
| 12 | Wrist Ulnar Deviation | Tilt wrist toward pinky side (ulnar side) | Flexor carpi ulnaris, Extensor carpi ulnaris |
| 13 | Wrist Supination | Rotate forearm so palm faces upward | Supinator, Biceps brachii |
| 14 | Wrist Pronation | Rotate forearm so palm faces downward | Pronator teres, Pronator quadratus |
| 15 | Power Grip | Grasp with full hand force (as if holding a hammer) | All flexors at maximum activation |
| 16 | Hand Open | Open hand fully with all fingers extended | Extensor digitorum, Extensor digiti minimi |
| 17 | Rest / Relaxed | Completely relaxed hand, no intentional movement | Minimal/baseline activity |

---

### Exercise C: Grasping and Functional Movements (23 gestures)
*Used in DB1, DB2, DB3 — Mimics activities of daily living*

| Gesture ID | Movement Name | Description | Real-World Analog |
|------------|---------------|-------------|-------------------|
| 1 | Large Diameter Grasp | Wrap fingers around a large cylindrical object | Holding a jar or bottle |
| 2 | Small Diameter Grasp | Wrap fingers around a small cylindrical object | Holding a pen or marker |
| 3 | Fixed Hook Grasp | Curl fingers without thumb involvement | Carrying a bag by the handle |
| 4 | Index Finger Extension Grasp | Grip with index finger extended along object | Pointing while holding (spray bottle trigger) |
| 5 | Medium Wrap | Moderate cylindrical grip | Holding a glass or cup |
| 6 | Light Tool Grasp | Precision grip with thumb and fingers | Holding a screwdriver |
| 7 | Adducted Thumb Grasp | Grip with thumb pressed against side of index | Holding a key |
| 8 | Parallel Extension Grasp | Grip flat object between extended fingers | Holding a plate or book |
| 9 | Extension Type Grasp | Grasp with fingers extended | Holding a CD or disc |
| 10 | Power Disk Grasp | Flat grip with palm pressure | Opening a jar lid |
| 11 | Open Lateral Tripod | Thumb, index, middle fingers form tripod | Picking up small object |
| 12 | Prismatic Pinch | Precision grip between thumb and index pad | Holding a needle or pin |
| 13 | Tip Pinch | Very fine pinch between thumb and index tips | Threading a needle |
| 14 | Prismatic Four Finger Grasp | Precision grip with four fingers | Writing with a pen |
| 15 | Lateral Grasp | Side grip between thumb and side of index | Holding a credit card |
| 16 | Parallel Flexion Grasp | Curved fingers grip flat object | Holding a smartphone |
| 17 | Power Sphere Grasp | Full hand wrap around spherical object | Holding a tennis ball |
| 18 | Tripod Grasp | Thumb, index, and middle finger pinch | Picking up a marble |
| 19 | Precision Sphere Grasp | Fingertip grip on small sphere | Holding a golf ball |
| 20 | Three Finger Sphere Grasp | Three-finger grip on medium sphere | Holding an egg |
| 21 | Palmar Grasp | Full palm contact with object | Pushing against a wall |
| 22 | Ring Grasp | Finger through a ring or loop | Holding scissors |
| 23 | Writing Tripod | Standard writing grip | Holding a pen for writing |

---

### Exercise D: Force Patterns (9 gestures)
*Used in DB2, DB3 only — Involves pressing force sensors*

| Gesture ID | Movement Name | Description |
|------------|---------------|-------------|
| 1 | Thumb Press | Apply force with thumb only |
| 2 | Index Press | Apply force with index finger only |
| 3 | Middle Press | Apply force with middle finger only |
| 4 | Ring Press | Apply force with ring finger only |
| 5 | Little Press | Apply force with little finger only |
| 6 | Thumb + Index Press | Apply force with thumb and index together |
| 7 | Thumb + Index + Middle Press | Apply force with three fingers together |
| 8 | Thumb + All Fingers Press | Apply force with thumb and all fingers |
| 9 | All Fingers Press (No Thumb) | Apply force with all four fingers (no thumb) |

---

### Gesture ID Mapping by Database

Since different databases use different exercise combinations, here's how gesture IDs map:

#### DB2 and DB3 (Exercises B + C + D = 49 gestures)

| File | Exercise | Gesture Range | Total in Exercise |
|------|----------|---------------|-------------------|
| S*_E1 | Exercise B | 1-17 | 17 gestures |
| S*_E2 | Exercise C | 1-23 | 23 gestures |
| S*_E3 | Exercise D | 1-9 | 9 gestures |

**Note:** Gesture label `0` always indicates **REST** (no movement).

#### DB1 (Exercises A + B + C = 52 gestures)

| File | Exercise | Gesture Range | Total in Exercise |
|------|----------|---------------|-------------------|
| S*_E1 | Exercise A | 1-12 | 12 gestures |
| S*_E2 | Exercise B | 1-17 | 17 gestures |
| S*_E3 | Exercise C | 1-23 | 23 gestures |

---

### Clinical Relevance of Gesture Groups

| Gesture Category | Prosthetic Application | Difficulty Level |
|------------------|------------------------|------------------|
| **Wrist Movements** (B9-B14) | Essential for positioning | Easy (large muscle groups) |
| **Power Grips** (B7, B15, C1-C5) | Heavy lifting, tool use | Medium |
| **Precision Pinches** (C12-C14) | Fine manipulation | Hard (small muscle differences) |
| **Finger Isolation** (A1-A12) | Typing, pointing | Very Hard (subtle EMG patterns) |
| **Force Patterns** (D1-D9) | Proportional control | Medium (force estimation) |

---

### Electrode Placement Reference

For **DB2/DB3** (Delsys Trigno, 12 channels):

| Channel | Anatomical Location | Primary Movements Detected |
|---------|--------------------|-----------------------------|
| 1-8 | Equally spaced around forearm at radio-humeral joint | Finger flexion/extension, wrist movements |
| 9 | Flexor Digitorum Superficialis (activity spot) | Finger flexion |
| 10 | Extensor Digitorum (activity spot) | Finger extension |
| 11 | Biceps Brachii | Elbow flexion, supination |
| 12 | Triceps Brachii | Elbow extension |

---

*Document Version 1.1 | NinaPro sEMG Processing Pipeline*
*Updated with complete gesture descriptions*
