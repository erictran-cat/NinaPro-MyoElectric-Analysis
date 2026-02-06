# Analytical Reflection: Physiological Interpretations

## Overview

This document provides physiological context for interpreting the pipeline outputs. Understanding the biological basis of sEMG signals is essential for meaningful analysis and clinical translation of gesture recognition systems.

---

## 1. Signal Processing Rationale

### 1.1 Bandpass Filter (20-450 Hz)

**Why these cutoffs?**

| Cutoff | Rationale |
|--------|-----------|
| **20 Hz highpass** | Removes motion artifacts (0-20 Hz) from electrode movement and cable sway. Also removes DC offset from electrode-skin interface potentials. |
| **450 Hz lowpass** | Captures the full sEMG frequency content (typically 20-500 Hz). Set below Nyquist (1000 Hz for 2 kHz sampling) to prevent aliasing. |

**Physiological basis**: Motor unit action potentials (MUAPs) have spectral energy primarily between 50-150 Hz, with some content extending to 400+ Hz depending on muscle fiber type and depth.

### 1.2 Notch Filter (50/60 Hz)

Removes powerline interference that couples capacitively into the recording system. This interference can be 10-100× larger than the sEMG signal if not removed.

**Implementation note**: We use a 4 Hz wide bandstop filter (58-62 Hz for 60 Hz regions) rather than an infinitely narrow notch to account for frequency drift in the power grid.

### 1.3 MAD Normalization

**Why Median Absolute Deviation instead of Z-score?**

| Method | Formula | Robustness |
|--------|---------|------------|
| Z-score | (x - μ) / σ | Sensitive to outliers |
| MAD | (x - median) / MAD | Robust to outliers |

**Clinical relevance**: sEMG signals contain spikes from motion artifacts, electrode pops, and crosstalk. MAD normalization prevents a single artifact from skewing the entire channel's scaling, which is critical for inter-subject comparisons.

---

## 2. Feature Interpretation

### 2.1 Time Domain Features

#### Mean Absolute Value (MAV)
```
MAV = (1/N) × Σ|xᵢ|
```

**Physiological meaning**: Average rectified EMG amplitude, proportional to the number of active motor units and their firing rates. Higher MAV indicates stronger muscle contraction.

**Clinical use**: Primary feature for proportional myoelectric control. Maps directly to prosthetic grip force.

#### Root Mean Square (RMS)
```
RMS = √[(1/N) × Σxᵢ²]
```

**Physiological meaning**: Signal power, related to muscle force output. The relationship between RMS and force is approximately linear up to 30-50% maximum voluntary contraction (MVC), then becomes nonlinear due to motor unit saturation.

**Why use both MAV and RMS?** MAV is more robust to noise; RMS better captures signal energy. They provide complementary information.

#### Waveform Length (WL)
```
WL = Σ|xᵢ₊₁ - xᵢ|
```

**Physiological meaning**: Cumulative signal complexity, capturing:
- Motor unit recruitment density (more units → more complex signal)
- Firing rate variability
- MUAP waveform shape changes

**Key insight**: WL does not saturate at high force levels like RMS does. This makes it valuable for proportional control across the full force range.

**Typical values** (normalized):
- Rest: WL ≈ 100-300
- Light grip: WL ≈ 500-1000
- Power grip: WL ≈ 2000-4000

#### Willison Amplitude (WAMP)
```
WAMP = Σf(|xᵢ₊₁ - xᵢ|), where f(x) = 1 if x > threshold, else 0
```

**Physiological meaning**: Counts the number of times the signal changes by more than a threshold, reflecting motor unit firing rate. Higher WAMP indicates faster motor unit firing (typically 8-30 Hz per unit).

**Threshold selection**: We use 10% of the maximum signal range. Too low → counts noise; too high → misses low-amplitude MUAPs.

### 2.2 Frequency Domain Features

#### Mean Power Frequency (MNP)
```
MNP = Σ(fᵢ × PSDᵢ) / Σ(PSDᵢ)
```

**Physiological meaning**: Spectral center of gravity. Reflects:
- Muscle fiber conduction velocity
- Motor unit recruitment pattern
- Muscle fatigue (MNP decreases with fatigue)

**Typical values**:
- Fresh muscle: MNP ≈ 80-120 Hz
- Fatigued muscle: MNP ≈ 50-80 Hz

**Clinical application**: Fatigue monitoring in prosthetic users. A dropping MNP over a session indicates the user may need rest.

#### Total Power (TP)
```
TP = Σ(PSDᵢ)
```

**Physiological meaning**: Overall spectral energy, closely related to RMS² by Parseval's theorem. Provides a frequency-domain view of signal intensity.

---

## 3. Spatial Activation Map Interpretation

The spatial activation heatmap shows normalized RMS across electrodes for each gesture. This reveals **muscle synergy patterns**—coordinated activation of multiple muscles for complex movements.

### Reading the Heatmap

| Pattern | Interpretation |
|---------|----------------|
| **Single bright column** | Gesture uses isolated muscle (e.g., single finger extension) |
| **Bright diagonal band** | Proximal-to-distal muscle chain activation |
| **Uniform row** | Global co-contraction (power grip) |
| **Sparse bright spots** | Precision grip with few muscle groups |

### Electrode Mapping (DB2/DB3)

| Channels | Anatomical Location | Primary Function |
|----------|---------------------|------------------|
| Ch1-Ch8 | Equally spaced around forearm | General flexor/extensor activity |
| Ch9 | Flexor digitorum superficialis | Finger flexion |
| Ch10 | Extensor digitorum | Finger extension |
| Ch11 | Biceps brachii | Elbow flexion, supination |
| Ch12 | Triceps brachii | Elbow extension |

### Expected Patterns by Gesture

| Gesture | Expected Activation Pattern |
|---------|----------------------------|
| Wrist Flexion (G9) | Ch1, Ch2, Ch9 dominant |
| Wrist Extension (G10) | Ch5, Ch6, Ch10 dominant |
| Supination (G13) | Ch11 (biceps) + Ch3-Ch4 |
| Pronation (G14) | Ch7-Ch8 + medial forearm |
| Power Grip (G15) | All channels, especially Ch9 |
| Hand Open (G16) | Ch10 dominant, extensors |

---

## 4. PCA Separability Analysis

### Interpreting the 3D Scatter Plot

**Well-separated clusters** indicate:
- Distinct EMG patterns for each gesture
- Good potential for classification accuracy
- Features capture meaningful physiological differences

**Overlapping clusters** suggest:
- Similar muscle activation patterns (e.g., index vs. middle finger)
- May need additional features or channels
- Consider merging confusable gestures

### Variance Explained

| PC1-3 Variance | Interpretation |
|----------------|----------------|
| > 80% | Excellent—linear classifier should work well |
| 60-80% | Good—may benefit from more PCs or nonlinear classifier |
| < 60% | Features may not capture gesture differences well |

### Common Cluster Confusion Pairs

Based on NinaPro studies, these gesture pairs often overlap in PCA space:

1. **Index extension ↔ Middle extension** (similar extensor recruitment)
2. **Ring flexion ↔ Little flexion** (ulnar nerve co-innervation)
3. **Wrist flexion ↔ Power grip** (overlapping flexor activation)

---

## 5. Signal Quality (SNR) Interpretation

### SNR Calculation
```
SNR (dB) = 10 × log₁₀(P_active / P_rest)
```

Where P_active is signal power during gestures and P_rest is power during rest periods.

### Quality Thresholds

| SNR | Quality | Action |
|-----|---------|--------|
| ≥ 10 dB | Excellent | Use in control loop |
| 3-10 dB | Marginal | Use with caution |
| < 3 dB | Poor | Exclude from classifier |

### Common Causes of Low SNR

| Cause | Solution |
|-------|----------|
| Electrode-skin impedance | Better skin prep (abrasion, alcohol) |
| Dry electrodes | Add conductive gel |
| Loose electrode | Secure with adhesive/band |
| Distant from muscle belly | Reposition electrode |
| Subcutaneous fat | Use higher gain or different electrode |

---

## 6. Real-Time Simulation Insights

### Latency Components

| Component | Typical Duration | Source |
|-----------|------------------|--------|
| Window acquisition | 100 ms | Half of 200 ms window |
| Feature extraction | 2-5 ms | MAV, RMS, etc. |
| Classification | 1-3 ms | LDA prediction |
| **Total** | **~105-110 ms** | Processing only |

Add mechanical delay of prosthetic motors (50-150 ms) for total user-perceived delay.

### Why 300 ms Maximum Latency?

Research shows that:
- **< 100 ms**: Imperceptible delay, natural feeling
- **100-200 ms**: Noticeable but acceptable
- **200-300 ms**: Annoying but usable
- **> 300 ms**: Significantly impairs control, user frustration

### Majority Voting Trade-off

| Window Size | Smoothing | Latency Added | Best For |
|-------------|-----------|---------------|----------|
| 1 (none) | None | 0 ms | Fast response |
| 3 | Moderate | ~100 ms | Balanced |
| 5 | Heavy | ~200 ms | Noisy signals |

---

## 7. Gesture Distinctiveness Analysis

### Distance Matrix Interpretation

The inter-class distance matrix shows Euclidean distances between gesture centroids in feature space.

| Distance | Interpretation |
|----------|----------------|
| > 5.0 | Highly distinct, easy to classify |
| 2.0-5.0 | Distinguishable with good classifier |
| < 2.0 | May be confused, consider merging |

### Top 5 Most Distinct Gestures (Typical DB2 Results)

| Rank | Gesture | Why Distinct |
|------|---------|--------------|
| 1 | Wrist Supination | Unique biceps activation |
| 2 | Index Extension | Isolated extensor indicis |
| 3 | Power Grip | Maximum global activation |
| 4 | Wrist Flexion | Strong flexor dominance |
| 5 | Hand Open | All extensors simultaneously |

---

## 8. Clinical Translation Considerations

### Gesture Set Reduction

For practical prosthetic control, reduce from 17-49 gestures to 5-8:

**Recommended core set:**
1. Rest (baseline)
2. Hand Open
3. Hand Close / Power Grip
4. Wrist Flexion
5. Wrist Extension
6. Pinch Grip (optional)
7. Point (optional)

### User Training Recommendations

1. **Initial calibration**: 5-10 repetitions per gesture
2. **Daily recalibration**: 2-3 repetitions (accounts for electrode shift)
3. **Progressive training**: Start with 3 gestures, add more as proficiency increases

### Electrode Shift Robustness

Socket rotation and electrode displacement is the #1 cause of classification accuracy drop in real-world use.

**Mitigation strategies:**
1. Train with intentional electrode variation
2. Use channel-agnostic features (relative ratios)
3. Implement online adaptation
4. Use high-density electrode arrays

---

## 9. References

1. Phinyomark, A., et al. (2012). "Feature reduction and selection for EMG signal classification." *Expert Systems with Applications*, 39(8), 7420-7431.

2. Farrell, T. R., & Weir, R. F. (2007). "The optimal controller delay for myoelectric prostheses." *IEEE Trans. Neural Syst. Rehabil. Eng.*, 15(1), 111-118.

3. Young, A. J., et al. (2012). "The effects of electrode size and orientation on the sensitivity of myoelectric pattern recognition systems." *IEEE Trans. Biomed. Eng.*, 59(12), 3403-3410.

4. Scheme, E., & Englehart, K. (2011). "Electromyogram pattern recognition for control of powered upper-limb prostheses." *IEEE Trans. Neural Syst. Rehabil. Eng.*, 19(4), 367-376.

---

*Document Version 2.0 | NinaPro sEMG Processing Pipeline*
