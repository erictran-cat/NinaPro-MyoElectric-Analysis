# NinaPro sEMG Processing Pipeline

[![MATLAB](https://img.shields.io/badge/MATLAB-R2016b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-NinaPro-orange.svg)](http://ninapro.hevs.ch/)

A comprehensive MATLAB pipeline for processing surface electromyography (sEMG) signals from the NinaPro database for robotic prosthetic hand control applications.

## 🎯 Overview

This pipeline provides end-to-end processing of sEMG signals for gesture recognition and prosthetic control, including:

- **Signal Processing**: Bandpass filtering, notch filtering, normalization
- **Feature Extraction**: Time and frequency domain features (MAV, RMS, WL, WAMP, MNP, TP)
- **Visualization**: Publication-quality figures for thesis/papers
- **Real-Time Simulation**: Prosthetic control simulation with latency analysis

## 📁 Repository Structure

```
ninapro_pipeline/
├── main_pipeline.m                 # Main entry point (self-contained)
├── feature_extraction.m            # Standalone feature extraction module
├── signal_processing.m             # Standalone signal processing module
├── visualization.m                 # Standalone visualization module
├── run_realtime_prosthetic_simulation.m  # Standalone simulation module
├── README.md                       # This file
├── COMPREHENSIVE_DOCUMENTATION.md  # Full technical documentation
└── ANALYTICAL_REFLECTION.md        # Physiological interpretations
```

## 🚀 Quick Start

### Option 1: Run Everything (Recommended)

```matlab
% Just run the main pipeline - everything is self-contained
run('main_pipeline.m')
```

### Option 2: Use Individual Modules

```matlab
% Load your NinaPro data
data = load('S1_E1_A1.mat');

% Process signals
[emg_filtered, config] = preprocess_emg(data.emg, config);

% Extract features
[features, labels] = extract_features(emg_filtered, data.restimulus, config);

% Visualize
generate_publication_figures(features, labels, snr, bad_ch, [], [], config);

% Run simulation
sim_results = run_realtime_prosthetic_simulation(features, labels, [], config);
```

## 📋 Requirements

### MATLAB Version
- MATLAB R2016b or later (for local functions in scripts)

### Required Toolboxes
| Toolbox | Functions Used |
|---------|----------------|
| Signal Processing Toolbox | `butter`, `filtfilt`, `pwelch` |
| Statistics and Machine Learning Toolbox | `fitcdiscr`, `pca`, `zscore`, `confusionmat` |

### Check Your Toolboxes
```matlab
license('test', 'Signal_Toolbox')       % Should return 1
license('test', 'Statistics_Toolbox')   % Should return 1
```

## ⚙️ Configuration

Edit the configuration section in `main_pipeline.m`:

```matlab
%% Configuration
config.fs = 2000;              % Sampling frequency (Hz)
config.num_channels = 12;      % Number of EMG channels
config.window_ms = 200;        % Analysis window (ms)
config.stride_ms = 50;         % Window stride (ms)
config.filter_low = 20;        % Highpass cutoff (Hz)
config.filter_high = 450;      % Lowpass cutoff (Hz)
config.notch_freq = 60;        % Powerline frequency (Hz)
config.use_restimulus = true;  % Use reaction-time corrected labels
```

### Database-Specific Settings

| Database | Sampling Rate | Channels | Electrode Type |
|----------|---------------|----------|----------------|
| DB1 | 100 Hz | 10 | Otto Bock |
| DB2 | 2000 Hz | 12 | Delsys Trigno |
| DB3 | 2000 Hz | 12 | Delsys Trigno |
| DB5 | 200 Hz | 16 | Myo Armband |

## 📊 Features Extracted

### Time Domain
| Feature | Description | Units |
|---------|-------------|-------|
| MAV | Mean Absolute Value | Normalized |
| RMS | Root Mean Square | Normalized |
| WL | Waveform Length | Normalized |
| WAMP | Willison Amplitude | Count |

### Frequency Domain
| Feature | Description | Units |
|---------|-------------|-------|
| MNP | Mean Power Frequency | Hz |
| TP | Total Power | Normalized²/Hz |

**Total**: 6 features × 12 channels = **72 features per segment**

## 📈 Output Files

### Data Files
| File | Description |
|------|-------------|
| `FeatureMatrix.csv` | Complete feature table |
| `processing_results.mat` | All variables and results |
| `realtime_simulation_results.mat` | Simulation metrics |

### Visualization Files (PNG, PDF, FIG)
| Figure | Description |
|--------|-------------|
| `Fig1_Spatial_Activation_Map` | Muscle synergy heatmap |
| `Fig2_PCA_Separability` | 3D PCA with scree plot |
| `Fig3_Signal_Quality_SNR` | Channel quality assessment |
| `Fig4_Feature_Distributions` | Box plots by gesture |
| `Fig5_Gesture_Distance_Matrix` | Inter-class distances |
| `Fig_RT_Timeline` | Real-time prediction stream |
| `Fig_RT_Latency` | Latency distribution analysis |
| `Fig_RT_Confusion` | Classification confusion matrix |
| `Fig_RT_Dashboard` | Performance summary dashboard |

## 🦾 Real-Time Simulation

The pipeline includes a prosthetic control simulation that:

1. **Trains an LDA classifier** on extracted features
2. **Simulates streaming predictions** at 20 Hz update rate
3. **Applies majority voting** for prediction smoothing
4. **Measures latency** for each prediction
5. **Evaluates clinical readiness** against thresholds

### Clinical Thresholds
| Metric | Target | Maximum Acceptable |
|--------|--------|-------------------|
| Accuracy | ≥90% | ≥80% |
| Latency | ≤200 ms | ≤300 ms |

### System Status
- **CLINICAL READY**: Accuracy ≥90% AND Latency ≤200ms
- **ACCEPTABLE**: Accuracy ≥80% AND Latency ≤300ms
- **NEEDS IMPROVEMENT**: Below acceptable thresholds

## 📚 Documentation

- **[COMPREHENSIVE_DOCUMENTATION.md](COMPREHENSIVE_DOCUMENTATION.md)**: Full technical reference with formulas, units, and gesture descriptions
- **[ANALYTICAL_REFLECTION.md](ANALYTICAL_REFLECTION.md)**: Physiological interpretations and clinical insights

## 🔬 NinaPro Database

This pipeline is designed for the [NinaPro Database](http://ninapro.hevs.ch/), a publicly available resource for myoelectric prosthetics research.

### Citing NinaPro
```bibtex
@article{atzori2014electromyography,
  title={Electromyography data for non-invasive naturally-controlled robotic hand prostheses},
  author={Atzori, Manfredo and Gijsberts, Arjan and Castellini, Claudio and others},
  journal={Scientific Data},
  volume={1},
  pages={140053},
  year={2014}
}
```

## 📖 References

1. Atzori, M. et al. (2014). "Electromyography data for non-invasive naturally-controlled robotic hand prostheses." *Scientific Data*, 1, 140053.

2. Hudgins, B. et al. (1993). "A new strategy for multifunction myoelectric control." *IEEE Trans. Biomed. Eng.*, 40(1), 82-94.

3. Scheme, E. & Englehart, K. (2011). "Electromyogram pattern recognition for control of powered upper-limb prostheses." *IEEE Trans. Neural Syst. Rehabil. Eng.*, 19(4), 367-376.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

*Developed for senior undergraduate thesis research in biomedical engineering.*
