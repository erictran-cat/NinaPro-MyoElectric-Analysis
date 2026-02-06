%% =========================================================================
%  FEATURE EXTRACTION MODULE
%  -------------------------------------------------------------------------
%  NinaPro sEMG Processing Pipeline
%  
%  This standalone module extracts time and frequency domain features from
%  preprocessed sEMG signals for gesture classification.
%
%  Usage:
%    [FeatureMatrix, labels] = extract_emg_features(emg, stimulus, config)
%
%  Required Toolboxes:
%    - Signal Processing Toolbox (pwelch)
%    - Statistics and Machine Learning Toolbox (none directly, but useful)
%
%  =========================================================================

function [FeatureMatrix, segment_labels, segment_info] = extract_emg_features(emg, stimulus, config)
    % EXTRACT_EMG_FEATURES Main entry point for feature extraction
    %
    % Inputs:
    %   emg      - Preprocessed EMG data [samples x channels]
    %   stimulus - Gesture labels [samples x 1]
    %   config   - Configuration struct with fields:
    %              .fs          - Sampling frequency (Hz)
    %              .window_ms   - Window size (ms)
    %              .stride_ms   - Stride (ms)
    %              .num_channels - Number of EMG channels
    %
    % Outputs:
    %   FeatureMatrix  - MATLAB table with named feature columns
    %   segment_labels - Label for each segment [n_segments x 1]
    %   segment_info   - Struct with segmentation details
    
    fprintf('Extracting features...\n');
    
    %% Validate inputs
    assert(size(emg, 2) == config.num_channels, ...
        'EMG channel count does not match config.num_channels');
    assert(length(stimulus) == size(emg, 1), ...
        'Stimulus length does not match EMG samples');
    
    %% Convert window parameters to samples
    window_samples = round(config.window_ms / 1000 * config.fs);
    stride_samples = round(config.stride_ms / 1000 * config.fs);
    
    %% Segment the data
    [segments, segment_labels, segment_info] = segment_emg(emg, stimulus, ...
        window_samples, stride_samples);
    
    fprintf('  > Segmented into %d windows\n', size(segments, 1));
    fprintf('  > Window: %d samples (%.1f ms)\n', window_samples, config.window_ms);
    fprintf('  > Stride: %d samples (%.1f ms)\n', stride_samples, config.stride_ms);
    
    %% Extract features from each segment
    [features, feature_names] = compute_features(segments, config);
    
    fprintf('  > Extracted %d features per segment\n', size(features, 2));
    
    %% Create output table
    FeatureMatrix = array2table(features, 'VariableNames', feature_names);
    
    % Store metadata
    segment_info.window_samples = window_samples;
    segment_info.stride_samples = stride_samples;
    segment_info.n_segments = size(segments, 1);
    segment_info.n_features = size(features, 2);
end

%% =========================================================================
%                         SEGMENTATION
%% =========================================================================

function [segments, labels, info] = segment_emg(emg, stimulus, window_samples, stride_samples)
    % SEGMENT_EMG Divide continuous EMG into overlapping windows
    %
    % Uses sliding window approach with mode-based labeling.
    
    n_samples = size(emg, 1);
    n_channels = size(emg, 2);
    
    % Calculate number of segments
    n_segments = floor((n_samples - window_samples) / stride_samples) + 1;
    
    % Preallocate
    segments = zeros(n_segments, window_samples, n_channels);
    labels = zeros(n_segments, 1);
    
    for i = 1:n_segments
        start_idx = (i - 1) * stride_samples + 1;
        end_idx = start_idx + window_samples - 1;
        
        % Extract window for all channels
        segments(i, :, :) = emg(start_idx:end_idx, :);
        
        % Assign label as mode of stimulus in window
        labels(i) = mode(stimulus(start_idx:end_idx));
    end
    
    % Store info
    info.start_indices = (0:n_segments-1)' * stride_samples + 1;
    info.end_indices = info.start_indices + window_samples - 1;
end

%% =========================================================================
%                         FEATURE COMPUTATION
%% =========================================================================

function [features, feature_names] = compute_features(segments, config)
    % COMPUTE_FEATURES Extract all features from segmented data
    %
    % Features per channel:
    %   1. MAV  - Mean Absolute Value
    %   2. RMS  - Root Mean Square
    %   3. WL   - Waveform Length
    %   4. WAMP - Willison Amplitude
    %   5. MNP  - Mean Power Frequency
    %   6. TP   - Total Power
    
    n_segments = size(segments, 1);
    n_channels = config.num_channels;
    n_features_per_channel = 6;
    
    % Preallocate feature matrix
    features = zeros(n_segments, n_channels * n_features_per_channel);
    
    % Generate feature names
    feature_names = cell(1, n_channels * n_features_per_channel);
    feature_types = {'MAV', 'RMS', 'WL', 'WAMP', 'MNP', 'TP'};
    
    for ch = 1:n_channels
        for f = 1:n_features_per_channel
            idx = (ch - 1) * n_features_per_channel + f;
            feature_names{idx} = sprintf('%s_Ch%02d', feature_types{f}, ch);
        end
    end
    
    % WAMP threshold (10% of typical signal range)
    wamp_threshold = 0.1;
    
    % FFT parameters for frequency features
    nfft = 2^nextpow2(size(segments, 2));
    
    % Extract features for each channel
    for ch = 1:n_channels
        % Get all segments for this channel [n_segments x window_samples]
        channel_data = squeeze(segments(:, :, ch));
        
        % Column indices for this channel's features
        col_start = (ch - 1) * n_features_per_channel + 1;
        
        % --- Time Domain Features (vectorized) ---
        
        % MAV: Mean Absolute Value
        features(:, col_start) = mean(abs(channel_data), 2);
        
        % RMS: Root Mean Square
        features(:, col_start + 1) = sqrt(mean(channel_data.^2, 2));
        
        % WL: Waveform Length
        features(:, col_start + 2) = sum(abs(diff(channel_data, 1, 2)), 2);
        
        % WAMP: Willison Amplitude
        features(:, col_start + 3) = sum(abs(diff(channel_data, 1, 2)) > wamp_threshold, 2);
        
        % --- Frequency Domain Features (loop required for pwelch) ---
        for seg = 1:n_segments
            segment_data = channel_data(seg, :);
            
            % Power spectral density via Welch's method
            [psd, f] = pwelch(segment_data, [], [], nfft, config.fs);
            
            % Ensure column vectors for computation
            psd = psd(:);
            f = f(:);
            
            % MNP: Mean Power Frequency (spectral centroid)
            if sum(psd) > eps
                features(seg, col_start + 4) = sum(f .* psd) / sum(psd);
            else
                features(seg, col_start + 4) = 0;
            end
            
            % TP: Total Power
            features(seg, col_start + 5) = sum(psd);
        end
    end
    
    % Handle NaN/Inf values
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
end

%% =========================================================================
%                    INDIVIDUAL FEATURE FUNCTIONS
%% =========================================================================
% These functions can be called independently if needed

function mav = compute_mav(x)
    % COMPUTE_MAV Mean Absolute Value
    %
    % Formula: MAV = (1/N) * sum(|x_i|)
    %
    % Physiological meaning: Average rectified EMG amplitude,
    % proportional to muscle activation level.
    
    mav = mean(abs(x));
end

function rms = compute_rms(x)
    % COMPUTE_RMS Root Mean Square
    %
    % Formula: RMS = sqrt((1/N) * sum(x_i^2))
    %
    % Physiological meaning: Signal power, approximately proportional
    % to muscle force up to ~50% MVC.
    
    rms = sqrt(mean(x.^2));
end

function wl = compute_wl(x)
    % COMPUTE_WL Waveform Length
    %
    % Formula: WL = sum(|x_{i+1} - x_i|)
    %
    % Physiological meaning: Cumulative length of the waveform,
    % reflects signal complexity and motor unit recruitment density.
    % Does not saturate at high force levels like RMS.
    
    wl = sum(abs(diff(x)));
end

function wamp = compute_wamp(x, threshold)
    % COMPUTE_WAMP Willison Amplitude
    %
    % Formula: WAMP = sum(f(|x_{i+1} - x_i|))
    %          where f(x) = 1 if x > threshold, else 0
    %
    % Physiological meaning: Number of amplitude changes exceeding
    % threshold, reflects motor unit firing rate.
    
    if nargin < 2
        threshold = 0.1;  % Default: 10% of normalized range
    end
    
    wamp = sum(abs(diff(x)) > threshold);
end

function [mnp, tp] = compute_frequency_features(x, fs)
    % COMPUTE_FREQUENCY_FEATURES Mean Power Frequency and Total Power
    %
    % MNP Formula: MNP = sum(f_i * PSD_i) / sum(PSD_i)
    % TP Formula:  TP = sum(PSD_i)
    %
    % Physiological meaning:
    %   MNP - Spectral center of gravity, decreases with muscle fatigue
    %   TP  - Overall spectral energy
    
    nfft = 2^nextpow2(length(x));
    [psd, f] = pwelch(x, [], [], nfft, fs);
    
    psd = psd(:);
    f = f(:);
    
    if sum(psd) > eps
        mnp = sum(f .* psd) / sum(psd);
    else
        mnp = 0;
    end
    
    tp = sum(psd);
end

%% =========================================================================
%                    ADDITIONAL FEATURES (OPTIONAL)
%% =========================================================================
% These features are not used in the main pipeline but are available
% for extended analysis if needed.

function zc = compute_zc(x, threshold)
    % COMPUTE_ZC Zero Crossings
    %
    % Counts the number of times the signal crosses zero.
    % Threshold prevents noise from triggering false crossings.
    
    if nargin < 2
        threshold = 0.01;
    end
    
    x_shifted = x(2:end);
    x_orig = x(1:end-1);
    
    % Zero crossing with threshold
    zc = sum((x_orig .* x_shifted < 0) & (abs(x_shifted - x_orig) > threshold));
end

function ssc = compute_ssc(x, threshold)
    % COMPUTE_SSC Slope Sign Changes
    %
    % Counts the number of times the slope changes sign.
    % Indicates frequency content of the signal.
    
    if nargin < 2
        threshold = 0.01;
    end
    
    diff1 = x(2:end-1) - x(1:end-2);
    diff2 = x(2:end-1) - x(3:end);
    
    ssc = sum((diff1 .* diff2 > 0) & ...
              ((abs(diff1) > threshold) | (abs(diff2) > threshold)));
end

function iemg = compute_iemg(x)
    % COMPUTE_IEMG Integrated EMG
    %
    % Sum of absolute values. Similar to MAV but not normalized by length.
    
    iemg = sum(abs(x));
end

function var_val = compute_var(x)
    % COMPUTE_VAR Variance
    %
    % Signal variance, measure of power.
    
    var_val = var(x);
end

function ssi = compute_ssi(x)
    % COMPUTE_SSI Simple Square Integral
    %
    % Sum of squared values. Energy measure.
    
    ssi = sum(x.^2);
end

function ar_coeffs = compute_ar(x, order)
    % COMPUTE_AR Autoregressive Coefficients
    %
    % AR model coefficients capture spectral shape.
    % Requires System Identification Toolbox or manual implementation.
    
    if nargin < 2
        order = 4;
    end
    
    % Simple Burg method implementation
    % For full implementation, use arburg() from Signal Processing Toolbox
    try
        ar_coeffs = arburg(x, order);
        ar_coeffs = ar_coeffs(2:end);  % Exclude first coefficient (always 1)
    catch
        ar_coeffs = zeros(1, order);
    end
end

function mdf = compute_mdf(x, fs)
    % COMPUTE_MDF Median Frequency
    %
    % Frequency that divides the power spectrum into two equal halves.
    % More robust to noise than mean frequency.
    
    nfft = 2^nextpow2(length(x));
    [psd, f] = pwelch(x, [], [], nfft, fs);
    
    cumulative_power = cumsum(psd);
    total_power = cumulative_power(end);
    
    mdf_idx = find(cumulative_power >= total_power / 2, 1, 'first');
    mdf = f(mdf_idx);
end

%% =========================================================================
%                    FEATURE NORMALIZATION
%% =========================================================================

function [features_norm, params] = normalize_features(features, method, params)
    % NORMALIZE_FEATURES Normalize feature matrix
    %
    % Methods:
    %   'zscore'  - Zero mean, unit variance
    %   'minmax'  - Scale to [0, 1]
    %   'robust'  - Median and MAD based (robust to outliers)
    %
    % Inputs:
    %   features - Feature matrix [n_samples x n_features]
    %   method   - Normalization method string
    %   params   - (Optional) Pre-computed normalization parameters
    %
    % Outputs:
    %   features_norm - Normalized features
    %   params        - Normalization parameters for future use
    
    if nargin < 2
        method = 'zscore';
    end
    
    compute_params = nargin < 3 || isempty(params);
    
    switch lower(method)
        case 'zscore'
            if compute_params
                params.mu = mean(features, 1);
                params.sigma = std(features, 0, 1);
                params.sigma(params.sigma < eps) = 1;  % Prevent division by zero
            end
            features_norm = (features - params.mu) ./ params.sigma;
            
        case 'minmax'
            if compute_params
                params.min_val = min(features, [], 1);
                params.max_val = max(features, [], 1);
                params.range = params.max_val - params.min_val;
                params.range(params.range < eps) = 1;
            end
            features_norm = (features - params.min_val) ./ params.range;
            
        case 'robust'
            if compute_params
                params.median_val = median(features, 1);
                params.mad_val = mad(features, 1, 1);
                params.mad_val(params.mad_val < eps) = 1;
            end
            features_norm = (features - params.median_val) ./ params.mad_val;
            
        otherwise
            error('Unknown normalization method: %s', method);
    end
    
    % Handle NaN values
    features_norm(isnan(features_norm)) = 0;
end
