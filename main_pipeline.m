%% =========================================================================
%  NinaPro sEMG Processing Pipeline for Robotic Prosthetic Integration
%  -------------------------------------------------------------------------
%  Senior Principal Research Engineer: Biomedical Signal Processing
%  Purpose: Transform raw NinaPro .mat structures into research-ready
%           feature matrices with high-fidelity visualizations
%  =========================================================================
%  Author: Research Engineering Team
%  Date: 2025
%  Version: 1.0
%  =========================================================================

clear; clc; close all;

%% ===================== CONFIGURATION PARAMETERS =========================
config = struct();

% =========================================================================
% IMPORTANT: Adjust these parameters for your NinaPro database version
% -------------------------------------------------------------------------
% DB1: fs = 100 Hz,  channels = 10
% DB2: fs = 2000 Hz, channels = 12
% DB3: fs = 2000 Hz, channels = 12
% DB5: fs = 200 Hz,  channels = 16 (Myo armband)
% =========================================================================

config.fs = 2000;                          % NinaPro DB sampling frequency (Hz)
config.filter_low = 20;                    % Bandpass lower cutoff (Hz)
config.filter_high = 450;                  % Bandpass upper cutoff (Hz) - keep below Nyquist!
config.notch_freq = 60;                    % Powerline interference (Hz) - use 50 for EU
config.notch_bw = 2;                       % Notch bandwidth (Hz)
config.filter_order = 4;                   % Butterworth filter order
config.window_ms = 200;                    % Sliding window length (ms)
config.stride_ms = 50;                     % Sliding window stride (ms)
config.wamp_threshold = 0.02;              % Willison Amplitude threshold (V)
config.num_channels = 12;                  % Expected EMG channels (DB2/DB3 = 12)
config.use_parallel = false;               % Set to true if Parallel Computing Toolbox available

% Validate configuration
assert(config.filter_high < config.fs / 2, ...
    'ERROR: filter_high (%.1f Hz) must be < Nyquist frequency (%.1f Hz)', ...
    config.filter_high, config.fs / 2);

% Derived parameters
config.window_samples = round(config.window_ms / 1000 * config.fs);
config.stride_samples = round(config.stride_ms / 1000 * config.fs);

%% ======================== LOAD DATA =====================================
fprintf('=== NinaPro sEMG Processing Pipeline ===\n');
fprintf('Loading NinaPro dataset...\n');

% Demonstration: Create synthetic NinaPro-like data for testing
% In production, replace with: data = load('S1_E1_A1.mat');
[emg, stimulus, restimulus, repetition] = generate_demo_data(config);

fprintf('  > EMG channels: %d\n', size(emg, 2));
fprintf('  > Total samples: %d\n', size(emg, 1));
fprintf('  > Unique gestures: %d\n', numel(unique(restimulus)) - 1);
fprintf('  > Duration: %.2f seconds\n\n', size(emg, 1) / config.fs);

%% =================== PRE-PROCESSING STAGE ===============================
fprintf('Pre-processing sEMG signals...\n');

% 1. Design filter bank
[filter_bank] = design_filter_bank(config);

% 2. Apply zero-phase filtering
emg_filtered = apply_zero_phase_filtering(emg, filter_bank);

% 3. Subject-specific MAD normalization
[emg_normalized, mad_values] = apply_mad_normalization(emg_filtered);

fprintf('  > Bandpass: %d-%d Hz (4th order Butterworth)\n', ...
    config.filter_low, config.filter_high);
fprintf('  > Notch filter: %d Hz\n', config.notch_freq);
fprintf('  > MAD normalization applied\n\n');

%% =================== SEGMENTATION =======================================
fprintf('Segmenting with sliding window buffer...\n');
fprintf('  > Window: %d ms (%d samples)\n', config.window_ms, config.window_samples);
fprintf('  > Stride: %d ms (%d samples)\n', config.stride_ms, config.stride_samples);
fprintf('  > Overlap: 75%%\n');

% Prioritize restimulus over stimulus (neuro-muscular latency compensation)
labels_to_use = restimulus;
fprintf('  > Using RESTIMULUS for temporal alignment\n\n');

% Segment data with sliding window
[segments, segment_labels, segment_info] = segment_signals(...
    emg_normalized, labels_to_use, config);

fprintf('  > Total segments extracted: %d\n', size(segments, 1));

%% =================== FEATURE EXTRACTION =================================
fprintf('\nExtracting features...\n');

% Extract comprehensive feature set
if config.use_parallel && ~isempty(gcp('nocreate'))
    fprintf('  > Parallel processing enabled\n');
end

[FeatureMatrix, feature_names] = extract_features_vectorized(segments, config);

fprintf('  > Features per channel: 6 (MAV, RMS, WL, WAMP, MNP, TP)\n');
fprintf('  > Total features: %d\n', width(FeatureMatrix) - 1);
fprintf('  > Feature matrix size: %d x %d\n\n', height(FeatureMatrix), width(FeatureMatrix));

%% =================== SIGNAL QUALITY ANALYSIS ============================
fprintf('Computing signal quality metrics...\n');

[snr_per_channel, bad_channels] = compute_signal_quality(emg_normalized, labels_to_use, config);

fprintf('  > SNR range: %.2f - %.2f dB\n', min(snr_per_channel), max(snr_per_channel));
if ~isempty(bad_channels)
    fprintf('  > WARNING: Bad electrodes detected: %s\n', mat2str(bad_channels));
else
    fprintf('  > All channels passed quality check\n');
end
fprintf('\n');

%% =================== VISUALIZATION ======================================
fprintf('Generating publication-quality visualizations...\n');

% Generate all plots using refactored publication module
generate_publication_figures(FeatureMatrix, segment_labels, snr_per_channel, ...
    bad_channels, emg_normalized, labels_to_use, config);

fprintf('  > Spatial Activation Map generated\n');
fprintf('  > PCA Class Separability Analysis generated\n');
fprintf('  > Signal Quality Metrics generated\n');
fprintf('  > Feature Distributions generated\n');
fprintf('  > Inter-Class Distance Matrix generated\n\n');

%% =================== GESTURE DISTINCTIVENESS ANALYSIS ===================
fprintf('Analyzing gesture distinctiveness...\n');

[top_gestures, distance_matrix] = analyze_gesture_distinctiveness(FeatureMatrix, segment_labels);

fprintf('\n=== TOP 5 MOST DISTINCT GESTURES ===\n');
fprintf('%-10s | %-25s | %-15s\n', 'Rank', 'Gesture Class', 'Avg Distance');
fprintf('%s\n', repmat('-', 1, 55));
for i = 1:min(5, size(top_gestures, 1))
    fprintf('%-10d | %-25s | %-15.4f\n', i, ...
        sprintf('Gesture %d', top_gestures(i, 1)), top_gestures(i, 2));
end
fprintf('\n');

%% =================== REAL-TIME PROSTHETIC SIMULATION ====================
fprintf('Running real-time prosthetic control simulation...\n\n');

% Create segment_info structure for simulation
segment_info = struct();
segment_info.window_ms = config.window_ms;
segment_info.stride_ms = config.stride_ms;

% Run simulation
sim_results = run_prosthetic_simulation(FeatureMatrix, segment_labels, ...
    segment_info, config);

%% =================== EXPORT RESULTS =====================================
fprintf('Exporting results...\n');

% Save feature matrix
writetable(FeatureMatrix, 'FeatureMatrix.csv');
save('processing_results.mat', 'FeatureMatrix', 'segment_labels', ...
    'snr_per_channel', 'distance_matrix', 'config', 'mad_values', 'sim_results');

fprintf('  > FeatureMatrix.csv exported\n');
fprintf('  > processing_results.mat saved\n');
fprintf('\n=== Pipeline Complete ===\n');

%% ========================================================================
%                         LOCAL FUNCTIONS
%% ========================================================================

function [emg, stimulus, restimulus, repetition] = generate_demo_data(config)
    % Generate synthetic NinaPro-like data for demonstration
    % In production, this would be replaced by actual data loading
    
    n_samples = config.fs * 30;  % 30 seconds of data
    n_channels = config.num_channels;
    n_gestures = 17;  % NinaPro DB2 exercise 1
    n_reps = 6;
    
    % Initialize arrays
    emg = zeros(n_samples, n_channels);
    stimulus = zeros(n_samples, 1);
    restimulus = zeros(n_samples, 1);
    repetition = zeros(n_samples, 1);
    
    % Generate gesture segments with realistic EMG patterns
    samples_per_gesture = floor(n_samples / (n_gestures * n_reps + n_reps));
    current_sample = 1;
    
    for rep = 1:n_reps
        for gest = 1:n_gestures
            if current_sample + samples_per_gesture > n_samples
                break;
            end
            
            % Create activation pattern (different channels for different gestures)
            activation = zeros(1, n_channels);
            primary_channels = mod(gest - 1, n_channels) + 1;
            secondary_channels = mod(gest, n_channels) + 1;
            activation(primary_channels) = 0.8 + 0.2 * rand();
            activation(secondary_channels) = 0.4 + 0.2 * rand();
            
            % Generate EMG-like signal (bandlimited noise with muscle burst)
            for ch = 1:n_channels
                base_noise = randn(samples_per_gesture, 1) * 0.01;
                burst = activation(ch) * randn(samples_per_gesture, 1) * 0.1;
                
                % Apply envelope
                envelope = gausswin(samples_per_gesture);
                emg(current_sample:current_sample+samples_per_gesture-1, ch) = ...
                    base_noise + burst .* envelope;
            end
            
            % Set labels (restimulus has slight delay to simulate reaction time)
            stimulus(current_sample:current_sample+samples_per_gesture-1) = gest;
            delay_samples = round(0.1 * config.fs);  % 100ms reaction delay
            restimulus(current_sample+delay_samples:current_sample+samples_per_gesture-1) = gest;
            repetition(current_sample:current_sample+samples_per_gesture-1) = rep;
            
            current_sample = current_sample + samples_per_gesture;
            
            % Add rest period
            rest_samples = round(samples_per_gesture * 0.3);
            if current_sample + rest_samples <= n_samples
                current_sample = current_sample + rest_samples;
            end
        end
    end
    
    % Add measurement noise
    emg = emg + randn(size(emg)) * 0.005;
end

function [filter_bank] = design_filter_bank(config)
    % Design filter bank: 4th-order Butterworth bandpass + 60Hz notch
    
    filter_bank = struct();
    
    % Nyquist frequency
    nyq = config.fs / 2;
    
    % Validate frequency parameters
    if config.filter_high >= nyq
        warning('High cutoff (%.1f Hz) >= Nyquist (%.1f Hz). Adjusting to %.1f Hz.', ...
            config.filter_high, nyq, nyq * 0.95);
        config.filter_high = nyq * 0.95;
    end
    
    % Bandpass filter (20-500 Hz)
    bp_low = config.filter_low / nyq;
    bp_high = config.filter_high / nyq;
    
    % Ensure valid normalized frequencies (0 < Wn < 1)
    bp_low = max(0.001, min(0.999, bp_low));
    bp_high = max(0.001, min(0.999, bp_high));
    
    [filter_bank.bp_b, filter_bank.bp_a] = butter(config.filter_order, ...
        [bp_low, bp_high], 'bandpass');
    
    % 60 Hz Notch filter using Butterworth bandstop (no DSP Toolbox required)
    % Design as a narrow bandstop filter around 60 Hz
    notch_low = (config.notch_freq - config.notch_bw) / nyq;
    notch_high = (config.notch_freq + config.notch_bw) / nyq;
    
    % Ensure valid normalized frequencies (0 < Wn < 1)
    notch_low = max(0.001, min(0.999, notch_low));
    notch_high = max(0.001, min(0.999, notch_high));
    
    [filter_bank.notch_b, filter_bank.notch_a] = butter(2, [notch_low, notch_high], 'stop');
    
    fprintf('  > Filter bank designed successfully\n');
end

function [emg_filtered] = apply_zero_phase_filtering(emg, filter_bank)
    % Apply zero-phase filtering using filtfilt
    
    [n_samples, n_channels] = size(emg);
    emg_filtered = zeros(n_samples, n_channels);
    
    for ch = 1:n_channels
        % Apply bandpass filter (zero-phase)
        temp = filtfilt(filter_bank.bp_b, filter_bank.bp_a, emg(:, ch));
        
        % Apply notch filter (zero-phase)
        emg_filtered(:, ch) = filtfilt(filter_bank.notch_b, filter_bank.notch_a, temp);
    end
end

function [emg_normalized, mad_values] = apply_mad_normalization(emg)
    % Subject-Specific Median Absolute Deviation normalization
    % Robust to electrode impedance outliers
    
    [n_samples, n_channels] = size(emg);
    emg_normalized = zeros(n_samples, n_channels);
    mad_values = zeros(1, n_channels);
    
    for ch = 1:n_channels
        channel_median = median(emg(:, ch));
        mad_values(ch) = median(abs(emg(:, ch) - channel_median));
        
        % Avoid division by zero
        if mad_values(ch) < eps
            mad_values(ch) = 1;
        end
        
        % Normalize: subtract median, divide by MAD
        emg_normalized(:, ch) = (emg(:, ch) - channel_median) / mad_values(ch);
    end
end

function [segments, segment_labels, segment_info] = segment_signals(emg, labels, config)
    % Sliding window segmentation with mode-based labeling
    
    [n_samples, n_channels] = size(emg);
    window_len = config.window_samples;
    stride = config.stride_samples;
    
    % Calculate number of segments
    n_segments = floor((n_samples - window_len) / stride) + 1;
    
    % Pre-allocate
    segments = zeros(n_segments, window_len, n_channels);
    segment_labels = zeros(n_segments, 1);
    segment_info = struct('start_idx', zeros(n_segments, 1), ...
                          'end_idx', zeros(n_segments, 1));
    
    % Segment with sliding window (parallel if available)
    if config.use_parallel && ~isempty(gcp('nocreate'))
        parfor i = 1:n_segments
            start_idx = (i - 1) * stride + 1;
            end_idx = start_idx + window_len - 1;
            
            segments(i, :, :) = emg(start_idx:end_idx, :);
            
            % Label by mode of restimulus values in window
            window_labels = labels(start_idx:end_idx);
            segment_labels(i) = mode(window_labels);
        end
    else
        for i = 1:n_segments
            start_idx = (i - 1) * stride + 1;
            end_idx = start_idx + window_len - 1;
            
            segments(i, :, :) = emg(start_idx:end_idx, :);
            window_labels = labels(start_idx:end_idx);
            segment_labels(i) = mode(window_labels);
            
            segment_info.start_idx(i) = start_idx;
            segment_info.end_idx(i) = end_idx;
        end
    end
end

function [FeatureMatrix, feature_names] = extract_features_vectorized(segments, config)
    % Vectorized feature extraction
    % Features: MAV, RMS, WL, WAMP (time domain); MNP, TP (frequency domain)
    
    [n_segments, window_len, n_channels] = size(segments);
    n_features_per_channel = 6;
    n_total_features = n_features_per_channel * n_channels;
    
    % Pre-allocate feature array
    features = zeros(n_segments, n_total_features);
    
    % Feature names
    feature_types = {'MAV', 'RMS', 'WL', 'WAMP', 'MNP', 'TP'};
    feature_names = cell(1, n_total_features);
    for ch = 1:n_channels
        for f = 1:n_features_per_channel
            idx = (ch - 1) * n_features_per_channel + f;
            feature_names{idx} = sprintf('%s_Ch%d', feature_types{f}, ch);
        end
    end
    
    % Vectorized extraction
    for ch = 1:n_channels
        channel_data = squeeze(segments(:, :, ch));  % [n_segments x window_len]
        base_idx = (ch - 1) * n_features_per_channel;
        
        % TIME DOMAIN FEATURES
        % 1. Mean Absolute Value (MAV)
        features(:, base_idx + 1) = mean(abs(channel_data), 2);
        
        % 2. Root Mean Square (RMS)
        features(:, base_idx + 2) = sqrt(mean(channel_data.^2, 2));
        
        % 3. Waveform Length (WL)
        features(:, base_idx + 3) = sum(abs(diff(channel_data, 1, 2)), 2);
        
        % 4. Willison Amplitude (WAMP)
        diff_data = abs(diff(channel_data, 1, 2));
        features(:, base_idx + 4) = sum(diff_data > config.wamp_threshold, 2);
        
        % FREQUENCY DOMAIN FEATURES
        % Compute PSD using Welch's method
        nfft = 2^nextpow2(window_len);
        
        for seg = 1:n_segments
            [psd, f] = pwelch(channel_data(seg, :), [], [], nfft, config.fs);
            
            % Ensure both are column vectors for element-wise operations
            f = f(:);
            psd = psd(:);
            
            % 5. Mean Power Spectral Frequency (MNP)
            % MNP = sum(f_i * P_i) / sum(P_i)
            features(seg, base_idx + 5) = sum(f .* psd) / sum(psd);
            
            % 6. Total Power (TP)
            features(seg, base_idx + 6) = sum(psd);
        end
    end
    
    % Create MATLAB Table
    FeatureMatrix = array2table(features, 'VariableNames', feature_names);
    
    % Note: Labels are stored separately in segment_labels to maintain
    % clean feature matrix structure
end

function [snr_per_channel, bad_channels] = compute_signal_quality(emg, labels, config)
    % Compute Signal-to-Noise Ratio per channel
    % SNR = 10 * log10(P_signal / P_noise)
    
    n_channels = size(emg, 2);
    snr_per_channel = zeros(1, n_channels);
    
    % Identify active (gesture) and rest periods
    active_idx = labels > 0;
    rest_idx = labels == 0;
    
    for ch = 1:n_channels
        if sum(active_idx) > 0 && sum(rest_idx) > 0
            signal_power = var(emg(active_idx, ch));
            noise_power = var(emg(rest_idx, ch));
            
            if noise_power > eps
                snr_per_channel(ch) = 10 * log10(signal_power / noise_power);
            else
                snr_per_channel(ch) = Inf;
            end
        else
            snr_per_channel(ch) = 0;
        end
    end
    
    % Identify bad channels (SNR < 3 dB threshold)
    snr_threshold = 3;
    bad_channels = find(snr_per_channel < snr_threshold);
end

%% =========================================================================
%  PUBLICATION-QUALITY VISUALIZATION FUNCTIONS
%  -------------------------------------------------------------------------
%  Designed for: Senior Undergraduate Thesis
%  Standard: IEEE/Academic Publication Ready
%  =========================================================================

function generate_publication_figures(FeatureMatrix, labels, snr_per_channel, ...
                                       bad_channels, ~, ~, config)
    % GENERATE_PUBLICATION_FIGURES Creates thesis-ready visualizations
    %
    % Inputs:
    %   FeatureMatrix   - MATLAB table with extracted features
    %   labels          - Segment labels (gesture classes)
    %   snr_per_channel - SNR values per electrode [1 x n_channels]
    %   bad_channels    - Indices of low-quality channels
    %   ~               - Unused (emg data placeholder)
    %   ~               - Unused (stimulus placeholder)
    %   config          - Configuration structure with num_channels, etc.
    
    %% --- Style Configuration ---
    style = configure_pub_style();
    
    %% --- Data Preparation ---
    feature_array = table2array(FeatureMatrix);
    unique_gestures = unique(labels(labels > 0));
    n_gestures = numel(unique_gestures);
    n_channels = config.num_channels;
    
    active_idx = labels > 0;
    active_features = feature_array(active_idx, :);
    active_labels = labels(active_idx);
    
    %% --- Generate All Figures ---
    create_spatial_activation_map(feature_array, labels, unique_gestures, ...
        n_gestures, n_channels, style);
    
    create_pca_separability_plot(active_features, active_labels, ...
        unique_gestures, n_gestures, style);
    
    create_signal_quality_plot(snr_per_channel, bad_channels, n_channels, style);
    
    create_feature_distribution_plot(feature_array, active_idx, ...
        active_labels, n_gestures, style);
    
    create_distance_matrix_plot(active_features, active_labels, ...
        unique_gestures, n_gestures, style);
end

function style = configure_pub_style()
    % CONFIGURE_PUB_STYLE Central style configuration
    
    % Font settings (sans-serif for figures)
    style.font.name = 'Arial';
    style.font.size.axis = 12;
    style.font.size.label = 14;
    style.font.size.title = 16;
    style.font.size.annotation = 11;
    style.font.weight.title = 'bold';
    
    % Color palette (ColorBrewer-based, colorblind-safe)
    style.colors.primary = [
        0.122, 0.467, 0.706;    % Blue
        0.890, 0.102, 0.110;    % Red
        0.173, 0.627, 0.173;    % Green
        1.000, 0.498, 0.000;    % Orange
        0.580, 0.404, 0.741;    % Purple
        0.549, 0.337, 0.294;    % Brown
        0.891, 0.467, 0.761;    % Pink
        0.498, 0.498, 0.498;    % Gray
        0.737, 0.741, 0.133;    % Olive
        0.090, 0.745, 0.812;    % Cyan
    ];
    
    % Semantic colors
    style.colors.good = [0.173, 0.627, 0.173];
    style.colors.bad = [0.890, 0.102, 0.110];
    style.colors.warning = [1.000, 0.498, 0.000];
    style.colors.neutral = [0.498, 0.498, 0.498];
    
    % Line settings
    style.line.width.primary = 2.0;
    style.line.width.secondary = 1.5;
    style.line.width.reference = 1.0;
    style.line.width.axis = 1.0;
    
    % Marker settings
    style.marker.size = 50;
    style.marker.alpha = 0.7;
    
    % Figure dimensions
    style.figure.width = 800;
    style.figure.height = 500;
    
    % Axis settings
    style.axis.grid.alpha = 0.3;
    
    % Export settings
    style.export.dpi = 300;
end

function colors = get_extended_palette(n)
    % GET_EXTENDED_PALETTE Generate n distinct colors using HSL
    hues = linspace(0, 1, n + 1);
    hues = hues(1:n);
    colors = zeros(n, 3);
    for i = 1:n
        h = hues(i); s = 0.7; l = 0.5;
        if l < 0.5, q = l * (1 + s); else, q = l + s - l * s; end
        p = 2 * l - q;
        colors(i, :) = [hue2rgb_val(p,q,h+1/3), hue2rgb_val(p,q,h), hue2rgb_val(p,q,h-1/3)];
    end
end

function c = hue2rgb_val(p, q, t)
    if t < 0, t = t + 1; end
    if t > 1, t = t - 1; end
    if t < 1/6, c = p + (q - p) * 6 * t;
    elseif t < 1/2, c = q;
    elseif t < 2/3, c = p + (q - p) * (2/3 - t) * 6;
    else, c = p; end
end

function apply_pub_style(ax, style)
    % APPLY_PUB_STYLE Apply consistent styling to axes
    ax.FontName = style.font.name;
    ax.FontSize = style.font.size.axis;
    ax.Box = 'off';
    ax.TickDir = 'out';
    ax.TickLength = [0.02, 0.02];
    ax.LineWidth = style.line.width.axis;
    ax.XGrid = 'on';
    ax.YGrid = 'on';
    ax.GridAlpha = style.axis.grid.alpha;
    ax.XMinorGrid = 'off';
    ax.YMinorGrid = 'off';
end

function export_pub_figure(fig, filename, style)
    % EXPORT_PUB_FIGURE Save figure in multiple formats
    exportgraphics(fig, [filename, '.png'], 'Resolution', style.export.dpi, ...
        'BackgroundColor', 'white');
    exportgraphics(fig, [filename, '.pdf'], 'ContentType', 'vector', ...
        'BackgroundColor', 'white');
    savefig(fig, [filename, '.fig']);
    fprintf('    Exported: %s\n', filename);
end

function create_spatial_activation_map(feature_array, labels, unique_gestures, ...
    n_gestures, n_channels, style)
    % CREATE_SPATIAL_ACTIVATION_MAP Muscle synergy heatmap
    
    %% Data Preparation
    rms_col_idx = 2:6:size(feature_array, 2);
    activation_map = zeros(n_gestures, n_channels);
    
    for g = 1:n_gestures
        gesture_idx = labels == unique_gestures(g);
        for ch = 1:min(n_channels, numel(rms_col_idx))
            activation_map(g, ch) = mean(feature_array(gesture_idx, rms_col_idx(ch)));
        end
    end
    activation_map = activation_map ./ max(activation_map(:));
    
    %% Create Figure
    fig = figure('Position', [100, 100, style.figure.width, style.figure.height], ...
        'Color', 'w');
    
    imagesc(activation_map);
    colormap(parula);
    
    cb = colorbar;
    cb.Label.String = 'Normalized RMS Activation';
    cb.Label.FontSize = style.font.size.label;
    cb.Label.FontName = style.font.name;
    cb.TickDirection = 'out';
    
    ax = gca;
    apply_pub_style(ax, style);
    
    xlabel('Electrode Channel', 'FontSize', style.font.size.label, ...
        'FontName', style.font.name);
    ylabel('Gesture Class', 'FontSize', style.font.size.label, ...
        'FontName', style.font.name);
    title('Spatial Activation Map: Muscle Synergy Patterns', ...
        'FontSize', style.font.size.title, 'FontName', style.font.name, ...
        'FontWeight', style.font.weight.title);
    
    xticks(1:n_channels);
    xticklabels(arrayfun(@(x) sprintf('Ch%d', x), 1:n_channels, 'UniformOutput', false));
    yticks(1:n_gestures);
    yticklabels(arrayfun(@(x) sprintf('G%02d', x), unique_gestures, 'UniformOutput', false));
    
    % Add value annotations
    for g = 1:n_gestures
        for ch = 1:n_channels
            val = activation_map(g, ch);
            if val > 0.5, txt_color = 'k'; else, txt_color = 'w'; end
            text(ch, g, sprintf('%.2f', val), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'Color', txt_color, ...
                'FontSize', style.font.size.annotation - 2, 'FontName', style.font.name, ...
                'FontWeight', 'bold');
        end
    end
    
    axis tight;
    export_pub_figure(fig, 'Fig1_Spatial_Activation_Map', style);
end

function create_pca_separability_plot(active_features, active_labels, ...
    unique_gestures, n_gestures, style)
    % CREATE_PCA_SEPARABILITY_PLOT 3D PCA with scree plot
    
    %% Data Preparation
    features_std = zscore(active_features);
    features_std(isnan(features_std)) = 0;
    [~, score, ~, ~, explained] = pca(features_std);
    
    %% Create Figure with Tiled Layout
    fig = figure('Position', [100, 100, style.figure.width + 200, ...
        style.figure.height + 100], 'Color', 'w');
    
    t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    %% 3D Scatter Plot
    nexttile([2, 2]);
    
    if n_gestures <= 10
        colors = style.colors.primary(1:n_gestures, :);
    else
        colors = get_extended_palette(n_gestures);
    end
    
    hold on;
    scatter_handles = gobjects(n_gestures, 1);
    for g = 1:n_gestures
        gesture_idx = active_labels == unique_gestures(g);
        scatter_handles(g) = scatter3(score(gesture_idx, 1), ...
            score(gesture_idx, 2), score(gesture_idx, 3), ...
            style.marker.size, colors(g, :), 'filled', ...
            'MarkerEdgeColor', 'k', 'MarkerEdgeAlpha', 0.3, ...
            'MarkerFaceAlpha', style.marker.alpha, ...
            'DisplayName', sprintf('G%02d', unique_gestures(g)));
    end
    hold off;
    
    ax = gca;
    apply_pub_style(ax, style);
    
    xlabel(sprintf('PC1 (%.1f%%)', explained(1)), 'FontSize', style.font.size.label, ...
        'FontName', style.font.name);
    ylabel(sprintf('PC2 (%.1f%%)', explained(2)), 'FontSize', style.font.size.label, ...
        'FontName', style.font.name);
    zlabel(sprintf('PC3 (%.1f%%)', explained(3)), 'FontSize', style.font.size.label, ...
        'FontName', style.font.name);
    title('PCA: Class Separability Analysis', 'FontSize', style.font.size.title, ...
        'FontName', style.font.name, 'FontWeight', style.font.weight.title);
    
    legend(scatter_handles, 'Location', 'eastoutside', ...
        'FontSize', style.font.size.annotation - 2, 'NumColumns', ceil(n_gestures / 9));
    grid on;
    view(45, 30);
    
    %% Scree Plot
    nexttile;
    n_comp = min(10, numel(explained));
    
    bar(1:n_comp, explained(1:n_comp), 'FaceColor', style.colors.primary(1, :), ...
        'EdgeColor', 'k', 'LineWidth', 0.5);
    hold on;
    plot(1:n_comp, cumsum(explained(1:n_comp)), '-o', ...
        'Color', style.colors.primary(2, :), 'LineWidth', style.line.width.primary, ...
        'MarkerFaceColor', style.colors.primary(2, :), 'MarkerSize', 6);
    yline(80, '--', 'Color', style.colors.neutral, 'LineWidth', style.line.width.reference);
    hold off;
    
    ax = gca;
    apply_pub_style(ax, style);
    xlabel('Principal Component', 'FontSize', style.font.size.label - 2, ...
        'FontName', style.font.name);
    ylabel('Variance (%)', 'FontSize', style.font.size.label - 2, ...
        'FontName', style.font.name);
    title('Scree Plot', 'FontSize', style.font.size.title - 2, ...
        'FontName', style.font.name, 'FontWeight', style.font.weight.title);
    legend({'Individual', 'Cumulative'}, 'Location', 'east', ...
        'FontSize', style.font.size.annotation - 2);
    ylim([0, 105]);
    xticks(1:n_comp);
    
    %% Summary Box
    nexttile;
    axis off;
    summary_text = {
        '\bf{Summary}', '', ...
        sprintf('Gestures: %d', n_gestures), ...
        sprintf('Samples: %d', size(active_features, 1)), ...
        sprintf('Features: %d', size(active_features, 2)), '', ...
        sprintf('PC1-3: %.1f%%', sum(explained(1:3)))
    };
    text(0.1, 0.9, summary_text, 'VerticalAlignment', 'top', ...
        'FontSize', style.font.size.annotation, 'FontName', style.font.name, ...
        'Interpreter', 'tex');
    
    export_pub_figure(fig, 'Fig2_PCA_Separability', style);
end

function create_signal_quality_plot(snr_per_channel, bad_channels, n_channels, style)
    % CREATE_SIGNAL_QUALITY_PLOT SNR bar chart
    
    fig = figure('Position', [100, 100, style.figure.width, style.figure.height - 100], ...
        'Color', 'w');
    
    %% Determine bar colors based on quality
    bar_colors = zeros(n_channels, 3);
    for ch = 1:n_channels
        if snr_per_channel(ch) >= 10
            bar_colors(ch, :) = style.colors.good;
        elseif snr_per_channel(ch) >= 3
            bar_colors(ch, :) = style.colors.warning;
        else
            bar_colors(ch, :) = style.colors.bad;
        end
    end
    
    %% Create bar chart
    b = bar(1:n_channels, snr_per_channel, 'FaceColor', 'flat', ...
        'EdgeColor', 'k', 'LineWidth', 0.5);
    b.CData = bar_colors;
    
    hold on;
    yline(3, '--', 'Color', style.colors.bad, 'LineWidth', style.line.width.reference, ...
        'Label', 'Min (3 dB)', 'LabelHorizontalAlignment', 'left', ...
        'FontSize', style.font.size.annotation);
    yline(10, '--', 'Color', style.colors.good, 'LineWidth', style.line.width.reference, ...
        'Label', 'Target (10 dB)', 'LabelHorizontalAlignment', 'left', ...
        'FontSize', style.font.size.annotation);
    hold off;
    
    ax = gca;
    apply_pub_style(ax, style);
    
    xlabel('Electrode Channel', 'FontSize', style.font.size.label, ...
        'FontName', style.font.name);
    ylabel('Signal-to-Noise Ratio (dB)', 'FontSize', style.font.size.label, ...
        'FontName', style.font.name);
    title('Signal Quality Assessment: SNR per Channel', ...
        'FontSize', style.font.size.title, 'FontName', style.font.name, ...
        'FontWeight', style.font.weight.title);
    
    xticks(1:n_channels);
    xticklabels(arrayfun(@(x) sprintf('Ch%d', x), 1:n_channels, 'UniformOutput', false));
    ylim([0, max(snr_per_channel) * 1.25]);
    
    % Value labels
    for ch = 1:n_channels
        text(ch, snr_per_channel(ch) + max(snr_per_channel) * 0.03, ...
            sprintf('%.1f', snr_per_channel(ch)), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontSize', style.font.size.annotation, ...
            'FontName', style.font.name, 'FontWeight', 'bold');
    end
    
    % Legend
    hold on;
    h1 = scatter(nan, nan, 100, style.colors.good, 's', 'filled', 'DisplayName', 'Good (≥10 dB)');
    h2 = scatter(nan, nan, 100, style.colors.warning, 's', 'filled', 'DisplayName', 'Marginal (3-10 dB)');
    h3 = scatter(nan, nan, 100, style.colors.bad, 's', 'filled', 'DisplayName', 'Poor (<3 dB)');
    hold off;
    legend([h1, h2, h3], 'Location', 'northeast', 'FontSize', style.font.size.annotation);
    
    export_pub_figure(fig, 'Fig3_Signal_Quality_SNR', style);
end

function create_feature_distribution_plot(feature_array, active_idx, ...
    active_labels, n_gestures, style)
    % CREATE_FEATURE_DISTRIBUTION_PLOT Box plots by gesture
    
    fig = figure('Position', [100, 100, style.figure.width + 200, ...
        style.figure.height - 100], 'Color', 'w');
    
    t = tiledlayout(1, 6, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    feature_names = {'MAV (norm.)', 'RMS (norm.)', 'WL (norm.)', ...
        'WAMP (count)', 'MNP (Hz)', 'TP (norm.²/Hz)'};
    ch = 1;  % Use Channel 1
    
    for f = 1:6
        nexttile;
        feature_idx = (ch - 1) * 6 + f;
        feature_data = feature_array(active_idx, feature_idx);
        
        boxplot(feature_data, active_labels, 'Colors', style.colors.primary(1, :), ...
            'Symbol', 'o', 'OutlierSize', 4);
        
        % Style box plot
        h = findobj(gca, 'Tag', 'Box');
        for j = 1:length(h)
            patch(get(h(j), 'XData'), get(h(j), 'YData'), ...
                style.colors.primary(1, :), 'FaceAlpha', 0.3);
        end
        
        ax = gca;
        apply_pub_style(ax, style);
        
        xlabel('Gesture', 'FontSize', style.font.size.label - 2, 'FontName', style.font.name);
        ylabel(feature_names{f}, 'FontSize', style.font.size.label - 2, 'FontName', style.font.name);
        
        if n_gestures > 10
            xticks(1:3:n_gestures);
        end
    end
    
    title(t, 'Feature Distributions by Gesture Class (Channel 1)', ...
        'FontSize', style.font.size.title, 'FontName', style.font.name, ...
        'FontWeight', style.font.weight.title);
    
    export_pub_figure(fig, 'Fig4_Feature_Distributions', style);
end

function create_distance_matrix_plot(active_features, active_labels, ...
    unique_gestures, n_gestures, style)
    % CREATE_DISTANCE_MATRIX_PLOT Inter-class distance heatmap
    
    %% Compute centroids and distances
    centroids = zeros(n_gestures, size(active_features, 2));
    for g = 1:n_gestures
        gesture_idx = active_labels == unique_gestures(g);
        centroids(g, :) = mean(active_features(gesture_idx, :), 1);
    end
    
    centroids_std = zscore(centroids);
    centroids_std(isnan(centroids_std)) = 0;
    distance_matrix = pdist2(centroids_std, centroids_std, 'euclidean');
    
    %% Create figure
    fig = figure('Position', [100, 100, style.figure.height + 50, style.figure.height], ...
        'Color', 'w');
    
    imagesc(distance_matrix);
    colormap(flipud(hot(256)));
    
    cb = colorbar;
    cb.Label.String = 'Euclidean Distance';
    cb.Label.FontSize = style.font.size.label;
    cb.Label.FontName = style.font.name;
    cb.TickDirection = 'out';
    
    ax = gca;
    apply_pub_style(ax, style);
    
    xlabel('Gesture Class', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    ylabel('Gesture Class', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    title('Inter-Class Distance Matrix', 'FontSize', style.font.size.title, ...
        'FontName', style.font.name, 'FontWeight', style.font.weight.title);
    
    xticks(1:n_gestures);
    yticks(1:n_gestures);
    xticklabels(arrayfun(@(x) sprintf('%d', x), unique_gestures, 'UniformOutput', false));
    yticklabels(arrayfun(@(x) sprintf('%d', x), unique_gestures, 'UniformOutput', false));
    
    axis square;
    
    % Add values for small matrices
    if n_gestures <= 10
        for i = 1:n_gestures
            for j = 1:n_gestures
                val = distance_matrix(i, j);
                if val < median(distance_matrix(:)), txt_color = 'w'; else, txt_color = 'k'; end
                text(j, i, sprintf('%.1f', val), 'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', 'Color', txt_color, ...
                    'FontSize', style.font.size.annotation - 2, 'FontName', style.font.name);
            end
        end
    end
    
    export_pub_figure(fig, 'Fig5_Gesture_Distance_Matrix', style);
end

function [top_gestures, distance_matrix] = analyze_gesture_distinctiveness(FeatureMatrix, labels)
    % Compute Euclidean distances between gesture centroids in feature space
    
    feature_array = table2array(FeatureMatrix);
    unique_gestures = unique(labels(labels > 0));
    n_gestures = numel(unique_gestures);
    
    % Compute centroids
    centroids = zeros(n_gestures, size(feature_array, 2));
    for g = 1:n_gestures
        gesture_idx = labels == unique_gestures(g);
        centroids(g, :) = mean(feature_array(gesture_idx, :), 1);
    end
    
    % Standardize centroids
    centroids_std = zscore(centroids);
    
    % Compute pairwise Euclidean distance matrix
    distance_matrix = pdist2(centroids_std, centroids_std, 'euclidean');
    
    % Compute average distance for each gesture (distinctiveness score)
    avg_distances = mean(distance_matrix, 2);
    
    % Sort by distinctiveness (descending)
    [sorted_dist, sort_idx] = sort(avg_distances, 'descend');
    
    top_gestures = [unique_gestures(sort_idx), sorted_dist];
end

%% =========================================================================
%  REAL-TIME PROSTHETIC CONTROL SIMULATION FUNCTIONS
%  =========================================================================

function [sim_results] = run_prosthetic_simulation(FeatureMatrix, ...
    segment_labels, ~, config)
    % RUN_PROSTHETIC_SIMULATION Simulate real-time prosthetic control
    
    fprintf('===========================================================\n');
    fprintf('  REAL-TIME PROSTHETIC CONTROL SIMULATION\n');
    fprintf('===========================================================\n\n');
    
    %% Configuration
    sim_config = struct();
    sim_config.classifier_type = 'LDA';
    sim_config.train_ratio = 0.7;
    sim_config.max_latency_ms = 300;
    sim_config.target_latency_ms = 200;
    sim_config.base_processing_time_ms = 5;
    sim_config.classifier_time_ms = 2;
    sim_config.use_majority_vote = true;
    sim_config.vote_window = 3;
    
    fprintf('Simulation Parameters:\n');
    fprintf('  > Window size: %d ms\n', config.window_ms);
    fprintf('  > Stride: %d ms (Update rate: %.1f Hz)\n', config.stride_ms, 1000/config.stride_ms);
    fprintf('  > Classifier: %s\n', sim_config.classifier_type);
    fprintf('\n');
    
    %% Train Classifier
    fprintf('Training classifier...\n');
    [classifier, train_results] = train_sim_classifier(FeatureMatrix, segment_labels, sim_config);
    fprintf('  > Training accuracy: %.2f%%\n', train_results.train_accuracy * 100);
    fprintf('  > Validation accuracy: %.2f%%\n', train_results.val_accuracy * 100);
    fprintf('  > Classes: %d gestures\n\n', train_results.n_classes);
    
    %% Simulate Real-Time Stream
    fprintf('Running real-time simulation...\n');
    [predictions, timing, ground_truth] = simulate_rt_stream(...
        FeatureMatrix, segment_labels, classifier, config, sim_config);
    fprintf('  > Total predictions: %d\n', numel(predictions));
    fprintf('  > Duration: %.2f seconds\n\n', timing.total_duration_sec);
    
    %% Analyze Performance
    fprintf('Analyzing performance...\n');
    metrics = analyze_rt_performance(predictions, ground_truth, timing, sim_config);
    fprintf('  > Overall accuracy: %.2f%%\n', metrics.accuracy * 100);
    fprintf('  > Mean latency: %.2f ms\n', metrics.mean_latency_ms);
    fprintf('  > Within spec (<%d ms): %.1f%%\n\n', sim_config.max_latency_ms, metrics.within_spec_pct);
    
    %% Generate Visualizations
    fprintf('Generating simulation visualizations...\n');
    style = get_sim_style();
    
    create_rt_timeline(predictions, ground_truth, timing, metrics, style, config);
    create_rt_latency_plot(timing, metrics, sim_config, style);
    create_rt_confusion(predictions, ground_truth, train_results.class_names, style);
    create_rt_dashboard(metrics, train_results, sim_config, style);
    
    %% Compile Results
    sim_results = struct();
    sim_results.config = sim_config;
    sim_results.train_results = train_results;
    sim_results.predictions = predictions;
    sim_results.ground_truth = ground_truth;
    sim_results.timing = timing;
    sim_results.metrics = metrics;
    
    % Print summary
    fprintf('\n-----------------------------------------------------------\n');
    fprintf('SIMULATION SUMMARY\n');
    fprintf('-----------------------------------------------------------\n');
    fprintf('  Accuracy: %.2f%% | Mean Latency: %.1f ms\n', ...
        metrics.accuracy * 100, metrics.mean_latency_ms);
    
    if metrics.accuracy >= 0.9 && metrics.mean_latency_ms <= 200
        fprintf('  Status: ✓ CLINICAL READY\n');
    elseif metrics.accuracy >= 0.8 && metrics.mean_latency_ms <= 300
        fprintf('  Status: ~ ACCEPTABLE\n');
    else
        fprintf('  Status: ✗ NEEDS IMPROVEMENT\n');
    end
    fprintf('-----------------------------------------------------------\n\n');
end

function [classifier, results] = train_sim_classifier(FeatureMatrix, labels, sim_config)
    % Train gesture classifier
    X = table2array(FeatureMatrix);
    y = labels;
    
    % Remove rest class
    active_idx = y > 0;
    X = X(active_idx, :);
    y = y(active_idx);
    
    classes = unique(y);
    n_classes = numel(classes);
    class_names = arrayfun(@(x) sprintf('G%02d', x), classes, 'UniformOutput', false);
    
    % Split data
    cv = cvpartition(y, 'HoldOut', 1 - sim_config.train_ratio);
    X_train = X(training(cv), :);
    y_train = y(training(cv));
    X_val = X(test(cv), :);
    y_val = y(test(cv));
    
    % Normalize
    [X_train_norm, mu, sigma] = zscore(X_train);
    X_val_norm = (X_val - mu) ./ sigma;
    X_train_norm(isnan(X_train_norm)) = 0;
    X_val_norm(isnan(X_val_norm)) = 0;
    
    % Train LDA
    classifier.model = fitcdiscr(X_train_norm, y_train, 'DiscrimType', 'linear');
    classifier.mu = mu;
    classifier.sigma = sigma;
    classifier.classes = classes;
    classifier.class_names = class_names;
    
    % Evaluate
    y_train_pred = predict(classifier.model, X_train_norm);
    y_val_pred = predict(classifier.model, X_val_norm);
    
    results.train_accuracy = mean(y_train_pred == y_train);
    results.val_accuracy = mean(y_val_pred == y_val);
    results.n_classes = n_classes;
    results.class_names = class_names;
    results.classes = classes;
end

function [predictions, timing, ground_truth] = simulate_rt_stream(...
    FeatureMatrix, labels, classifier, config, sim_config)
    % Simulate streaming predictions
    
    X = table2array(FeatureMatrix);
    y = labels;
    n_samples = size(X, 1);
    
    predictions = zeros(n_samples, 1);
    timing.prediction_time_ms = zeros(n_samples, 1);
    timing.cumulative_time_ms = zeros(n_samples, 1);
    timing.latency_ms = zeros(n_samples, 1);
    ground_truth = y;
    
    vote_buffer = zeros(sim_config.vote_window, 1);
    buffer_idx = 1;
    cumulative_time = 0;
    
    for i = 1:n_samples
        % Normalize and predict
        x_norm = (X(i, :) - classifier.mu) ./ classifier.sigma;
        x_norm(isnan(x_norm)) = 0;
        raw_pred = predict(classifier.model, x_norm);
        
        % Majority voting
        if sim_config.use_majority_vote
            vote_buffer(buffer_idx) = raw_pred;
            buffer_idx = mod(buffer_idx, sim_config.vote_window) + 1;
            if i >= sim_config.vote_window
                predictions(i) = mode(vote_buffer);
            else
                predictions(i) = raw_pred;
            end
        else
            predictions(i) = raw_pred;
        end
        
        % Timing
        simulated_time = sim_config.base_processing_time_ms + ...
            sim_config.classifier_time_ms + randn * 1;
        simulated_time = max(1, simulated_time);
        
        timing.prediction_time_ms(i) = simulated_time;
        cumulative_time = cumulative_time + config.stride_ms;
        timing.cumulative_time_ms(i) = cumulative_time;
        timing.latency_ms(i) = simulated_time + config.window_ms / 2;
    end
    
    timing.total_duration_sec = cumulative_time / 1000;
    timing.stride_ms = config.stride_ms;
end

function metrics = analyze_rt_performance(predictions, ground_truth, timing, sim_config)
    % Compute performance metrics
    
    active_idx = ground_truth > 0;
    pred_active = predictions(active_idx);
    true_active = ground_truth(active_idx);
    latency_active = timing.latency_ms(active_idx);
    
    % Accuracy
    metrics.accuracy = mean(pred_active == true_active);
    metrics.n_correct = sum(pred_active == true_active);
    metrics.n_total = numel(pred_active);
    
    % Per-class accuracy
    classes = unique(true_active);
    metrics.per_class_accuracy = zeros(numel(classes), 1);
    for i = 1:numel(classes)
        class_idx = true_active == classes(i);
        metrics.per_class_accuracy(i) = mean(pred_active(class_idx) == true_active(class_idx));
    end
    metrics.classes = classes;
    
    % Latency
    metrics.mean_latency_ms = mean(latency_active);
    metrics.std_latency_ms = std(latency_active);
    metrics.median_latency_ms = median(latency_active);
    metrics.p95_latency_ms = prctile(latency_active, 95);
    metrics.max_latency_ms = max(latency_active);
    
    metrics.within_target = sum(latency_active <= sim_config.target_latency_ms);
    metrics.within_target_pct = metrics.within_target / numel(latency_active) * 100;
    metrics.within_spec = sum(latency_active <= sim_config.max_latency_ms);
    metrics.within_spec_pct = metrics.within_spec / numel(latency_active) * 100;
    
    % Confusion matrix
    metrics.confusion_matrix = confusionmat(true_active, pred_active);
    
    % F1 scores
    n_classes = numel(classes);
    metrics.f1_score = zeros(n_classes, 1);
    for i = 1:n_classes
        tp = sum(pred_active == classes(i) & true_active == classes(i));
        fp = sum(pred_active == classes(i) & true_active ~= classes(i));
        fn = sum(pred_active ~= classes(i) & true_active == classes(i));
        prec = tp / max(tp + fp, 1);
        rec = tp / max(tp + fn, 1);
        metrics.f1_score(i) = 2 * prec * rec / max(prec + rec, eps);
    end
    metrics.macro_f1 = mean(metrics.f1_score);
end

function style = get_sim_style()
    % Get visualization style
    style.font.name = 'Arial';
    style.font.size.axis = 12;
    style.font.size.label = 14;
    style.font.size.title = 16;
    style.font.size.annotation = 11;
    style.font.weight.title = 'bold';
    
    style.colors.correct = [0.173, 0.627, 0.173];
    style.colors.incorrect = [0.890, 0.102, 0.110];
    style.colors.warning = [1.000, 0.498, 0.000];
    style.colors.primary = [0.122, 0.467, 0.706];
    style.colors.neutral = [0.498, 0.498, 0.498];
    
    style.line.width = 2.0;
    style.figure.width = 900;
    style.figure.height = 500;
    style.export.dpi = 300;
end

function apply_rt_style(ax, style)
    ax.FontName = style.font.name;
    ax.FontSize = style.font.size.axis;
    ax.Box = 'off';
    ax.TickDir = 'out';
    ax.LineWidth = 1.0;
    ax.XGrid = 'on';
    ax.YGrid = 'on';
    ax.GridAlpha = 0.3;
end

function export_rt_figure(fig, filename, style)
    exportgraphics(fig, [filename, '.png'], 'Resolution', style.export.dpi, ...
        'BackgroundColor', 'white');
    exportgraphics(fig, [filename, '.pdf'], 'ContentType', 'vector', ...
        'BackgroundColor', 'white');
    savefig(fig, [filename, '.fig']);
    fprintf('    Exported: %s\n', filename);
end

function create_rt_timeline(predictions, ground_truth, timing, metrics, style, config)
    % Timeline visualization
    
    fig = figure('Position', [100, 100, style.figure.width + 200, style.figure.height], 'Color', 'w');
    t = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    time_sec = timing.cumulative_time_ms / 1000;
    max_time = min(10, max(time_sec));
    win_idx = time_sec <= max_time;
    
    time_win = time_sec(win_idx);
    pred_win = predictions(win_idx);
    true_win = ground_truth(win_idx);
    latency_win = timing.latency_ms(win_idx);
    
    % Ground truth
    nexttile;
    stairs(time_win, true_win, 'LineWidth', style.line.width, 'Color', style.colors.primary);
    ax = gca; apply_rt_style(ax, style);
    ylabel('True Gesture', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    title('Ground Truth', 'FontSize', style.font.size.title - 2, 'FontName', style.font.name);
    xlim([0, max_time]);
    
    % Predictions
    nexttile;
    hold on;
    correct_idx = pred_win == true_win;
    scatter(time_win(correct_idx), pred_win(correct_idx), 20, style.colors.correct, 'filled', 'MarkerFaceAlpha', 0.7);
    scatter(time_win(~correct_idx), pred_win(~correct_idx), 20, style.colors.incorrect, 'filled', 'MarkerFaceAlpha', 0.7);
    stairs(time_win, pred_win, 'LineWidth', 1.5, 'Color', [0.3, 0.3, 0.3]);
    hold off;
    ax = gca; apply_rt_style(ax, style);
    ylabel('Predicted', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    title(sprintf('Predictions (Accuracy: %.1f%%)', metrics.accuracy * 100), ...
        'FontSize', style.font.size.title - 2, 'FontName', style.font.name);
    xlim([0, max_time]);
    legend({'Correct', 'Incorrect'}, 'Location', 'eastoutside');
    
    % Latency
    nexttile;
    hold on;
    good_idx = latency_win <= 200;
    warn_idx = latency_win > 200 & latency_win <= 300;
    bad_idx = latency_win > 300;
    scatter(time_win(good_idx), latency_win(good_idx), 15, style.colors.correct, 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(time_win(warn_idx), latency_win(warn_idx), 15, style.colors.warning, 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(time_win(bad_idx), latency_win(bad_idx), 15, style.colors.incorrect, 'filled', 'MarkerFaceAlpha', 0.6);
    yline(200, '--', 'Color', style.colors.correct, 'LineWidth', 1.5);
    yline(300, '--', 'Color', style.colors.incorrect, 'LineWidth', 1.5);
    hold off;
    ax = gca; apply_rt_style(ax, style);
    xlabel('Time (seconds)', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    ylabel('Latency (ms)', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    title(sprintf('Latency (Mean: %.1f ms)', metrics.mean_latency_ms), ...
        'FontSize', style.font.size.title - 2, 'FontName', style.font.name);
    xlim([0, max_time]);
    ylim([0, max(350, max(latency_win) * 1.1)]);
    
    title(t, 'Real-Time Prosthetic Control Simulation', 'FontSize', style.font.size.title, ...
        'FontName', style.font.name, 'FontWeight', style.font.weight.title);
    
    export_rt_figure(fig, 'Fig_RT_Timeline', style);
end

function create_rt_latency_plot(timing, metrics, sim_config, style)
    % Latency analysis
    
    fig = figure('Position', [100, 100, style.figure.width, style.figure.height - 100], 'Color', 'w');
    t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % Histogram
    nexttile;
    histogram(timing.latency_ms, 30, 'FaceColor', style.colors.primary, 'EdgeColor', 'w', 'FaceAlpha', 0.8);
    hold on;
    xline(metrics.mean_latency_ms, '-', 'Color', [0.8, 0.4, 0.7], 'LineWidth', 2, ...
        'Label', sprintf('Mean: %.1f ms', metrics.mean_latency_ms));
    xline(sim_config.target_latency_ms, '--', 'Color', style.colors.correct, 'LineWidth', 2);
    xline(sim_config.max_latency_ms, '--', 'Color', style.colors.incorrect, 'LineWidth', 2);
    hold off;
    ax = gca; apply_rt_style(ax, style);
    xlabel('Latency (ms)', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    ylabel('Count', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    title('Latency Distribution', 'FontSize', style.font.size.title - 2, 'FontName', style.font.name);
    
    % Statistics bar
    nexttile;
    categories = {'Mean', 'Median', 'P95', 'Max'};
    values = [metrics.mean_latency_ms, metrics.median_latency_ms, metrics.p95_latency_ms, metrics.max_latency_ms];
    bar_colors = zeros(4, 3);
    for i = 1:4
        if values(i) <= sim_config.target_latency_ms
            bar_colors(i, :) = style.colors.correct;
        elseif values(i) <= sim_config.max_latency_ms
            bar_colors(i, :) = style.colors.warning;
        else
            bar_colors(i, :) = style.colors.incorrect;
        end
    end
    b = bar(1:4, values, 'FaceColor', 'flat', 'EdgeColor', 'k');
    b.CData = bar_colors;
    hold on;
    yline(sim_config.target_latency_ms, '--', 'Color', style.colors.correct, 'LineWidth', 1.5);
    yline(sim_config.max_latency_ms, '--', 'Color', style.colors.incorrect, 'LineWidth', 1.5);
    hold off;
    ax = gca; apply_rt_style(ax, style);
    xticks(1:4); xticklabels(categories);
    ylabel('Latency (ms)', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    title('Latency Statistics', 'FontSize', style.font.size.title - 2, 'FontName', style.font.name);
    for i = 1:4
        text(i, values(i) + 5, sprintf('%.0f', values(i)), 'HorizontalAlignment', 'center', ...
            'FontSize', style.font.size.annotation, 'FontWeight', 'bold');
    end
    
    title(t, 'Prediction Latency Analysis', 'FontSize', style.font.size.title, ...
        'FontName', style.font.name, 'FontWeight', style.font.weight.title);
    
    export_rt_figure(fig, 'Fig_RT_Latency', style);
end

function create_rt_confusion(predictions, ground_truth, class_names, style)
    % Confusion matrix
    
    active_idx = ground_truth > 0;
    pred_active = predictions(active_idx);
    true_active = ground_truth(active_idx);
    
    cm = confusionmat(true_active, pred_active);
    cm_norm = cm ./ sum(cm, 2);
    cm_norm(isnan(cm_norm)) = 0;
    n_classes = size(cm, 1);
    
    fig = figure('Position', [100, 100, style.figure.height + 100, style.figure.height], 'Color', 'w');
    
    imagesc(cm_norm);
    colormap(parula);
    cb = colorbar;
    cb.Label.String = 'Classification Rate';
    cb.Label.FontSize = style.font.size.label;
    
    for i = 1:n_classes
        for j = 1:n_classes
            val = cm_norm(i, j);
            if val > 0.5, txt_color = 'w'; else, txt_color = 'k'; end
            if val > 0.01
                text(j, i, sprintf('%.2f', val), 'HorizontalAlignment', 'center', ...
                    'Color', txt_color, 'FontSize', style.font.size.annotation - 2, 'FontWeight', 'bold');
            end
        end
    end
    
    ax = gca; apply_rt_style(ax, style);
    xlabel('Predicted', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    ylabel('True', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    title('Real-Time Confusion Matrix', 'FontSize', style.font.size.title, ...
        'FontName', style.font.name, 'FontWeight', style.font.weight.title);
    xticks(1:n_classes); yticks(1:n_classes);
    if n_classes <= numel(class_names)
        xticklabels(class_names(1:n_classes));
        yticklabels(class_names(1:n_classes));
    end
    axis square;
    
    export_rt_figure(fig, 'Fig_RT_Confusion', style);
end

function create_rt_dashboard(metrics, train_results, sim_config, style)
    % Performance dashboard
    
    fig = figure('Position', [100, 100, style.figure.width, style.figure.height + 50], 'Color', 'w');
    t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % Per-class accuracy
    nexttile;
    n_classes = numel(metrics.classes);
    bar_colors = zeros(n_classes, 3);
    for i = 1:n_classes
        if metrics.per_class_accuracy(i) >= 0.9
            bar_colors(i, :) = style.colors.correct;
        elseif metrics.per_class_accuracy(i) >= 0.7
            bar_colors(i, :) = style.colors.warning;
        else
            bar_colors(i, :) = style.colors.incorrect;
        end
    end
    b = bar(1:n_classes, metrics.per_class_accuracy * 100, 'FaceColor', 'flat', 'EdgeColor', 'k');
    b.CData = bar_colors;
    hold on; yline(90, '--', 'Color', style.colors.correct, 'LineWidth', 1.5); hold off;
    ax = gca; apply_rt_style(ax, style);
    xlabel('Gesture', 'FontSize', style.font.size.label - 2);
    ylabel('Accuracy (%)', 'FontSize', style.font.size.label - 2);
    title('Per-Gesture Accuracy', 'FontSize', style.font.size.title - 2, 'FontWeight', 'bold');
    ylim([0, 105]);
    
    % F1 Scores
    nexttile;
    bar(1:n_classes, metrics.f1_score, 'FaceColor', style.colors.primary, 'EdgeColor', 'k');
    ax = gca; apply_rt_style(ax, style);
    xlabel('Gesture', 'FontSize', style.font.size.label - 2);
    ylabel('F1 Score', 'FontSize', style.font.size.label - 2);
    title(sprintf('F1 Scores (Macro: %.3f)', metrics.macro_f1), ...
        'FontSize', style.font.size.title - 2, 'FontWeight', 'bold');
    ylim([0, 1.05]);
    
    % Metrics summary
    nexttile;
    axis off;
    summary_text = {
        '\bf{Performance Summary}', '', ...
        sprintf('Overall Accuracy: %.2f%%', metrics.accuracy * 100), ...
        sprintf('Macro F1-Score: %.3f', metrics.macro_f1), ...
        sprintf('Gesture Classes: %d', train_results.n_classes), '', ...
        '\bf{Latency Metrics}', '', ...
        sprintf('Mean: %.1f ms', metrics.mean_latency_ms), ...
        sprintf('95th Percentile: %.1f ms', metrics.p95_latency_ms), ...
        sprintf('Within Spec: %.1f%%', metrics.within_spec_pct)
    };
    text(0.1, 0.95, summary_text, 'VerticalAlignment', 'top', ...
        'FontSize', style.font.size.annotation, 'FontName', style.font.name, 'Interpreter', 'tex');
    
    % Status
    nexttile;
    axis off;
    if metrics.accuracy >= 0.9 && metrics.mean_latency_ms <= 200
        status = 'CLINICAL READY'; status_color = style.colors.correct;
    elseif metrics.accuracy >= 0.8 && metrics.mean_latency_ms <= 300
        status = 'ACCEPTABLE'; status_color = style.colors.warning;
    else
        status = 'NEEDS WORK'; status_color = style.colors.incorrect;
    end
    text(0.5, 0.6, status, 'HorizontalAlignment', 'center', 'FontSize', 22, ...
        'FontWeight', 'bold', 'Color', status_color);
    text(0.5, 0.3, 'System Status', 'HorizontalAlignment', 'center', ...
        'FontSize', style.font.size.label);
    
    title(t, 'Real-Time Prosthetic Control: Performance Dashboard', ...
        'FontSize', style.font.size.title, 'FontWeight', style.font.weight.title);
    
    export_rt_figure(fig, 'Fig_RT_Dashboard', style);
end
