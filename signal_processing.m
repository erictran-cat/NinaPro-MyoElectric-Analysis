%% =========================================================================
%  SIGNAL PROCESSING MODULE
%  -------------------------------------------------------------------------
%  NinaPro sEMG Processing Pipeline
%  
%  This standalone module provides signal conditioning functions for
%  surface EMG signals including filtering, normalization, and quality
%  assessment.
%
%  Usage:
%    [emg_processed, info] = preprocess_emg(emg_raw, config)
%
%  Required Toolboxes:
%    - Signal Processing Toolbox (butter, filtfilt)
%
%  =========================================================================

function [emg_processed, processing_info] = preprocess_emg(emg_raw, config)
    % PREPROCESS_EMG Main entry point for signal preprocessing
    %
    % Inputs:
    %   emg_raw  - Raw EMG data [samples x channels]
    %   config   - Configuration struct with fields:
    %              .fs          - Sampling frequency (Hz)
    %              .filter_low  - Highpass cutoff (Hz)
    %              .filter_high - Lowpass cutoff (Hz)
    %              .notch_freq  - Powerline frequency (Hz), 50 or 60
    %              .num_channels - Number of EMG channels
    %
    % Outputs:
    %   emg_processed  - Preprocessed EMG [samples x channels]
    %   processing_info - Struct with processing details and MAD values
    
    fprintf('Preprocessing EMG signals...\n');
    
    %% Validate inputs
    assert(config.filter_high < config.fs / 2, ...
        'Highpass cutoff must be below Nyquist frequency (%.1f Hz)', config.fs / 2);
    
    n_samples = size(emg_raw, 1);
    n_channels = size(emg_raw, 2);
    
    fprintf('  > Input: %d samples x %d channels\n', n_samples, n_channels);
    fprintf('  > Sampling rate: %d Hz\n', config.fs);
    
    %% Step 1: Bandpass Filtering
    fprintf('  > Applying bandpass filter (%.0f-%.0f Hz)...\n', ...
        config.filter_low, config.filter_high);
    
    emg_bandpass = apply_bandpass_filter(emg_raw, config.fs, ...
        config.filter_low, config.filter_high);
    
    %% Step 2: Notch Filtering (Powerline Removal)
    fprintf('  > Applying notch filter (%.0f Hz)...\n', config.notch_freq);
    
    emg_notched = apply_notch_filter(emg_bandpass, config.fs, config.notch_freq);
    
    %% Step 3: Normalization (MAD-based)
    fprintf('  > Normalizing with MAD...\n');
    
    [emg_processed, mad_values] = apply_mad_normalization(emg_notched);
    
    %% Compile processing info
    processing_info.n_samples = n_samples;
    processing_info.n_channels = n_channels;
    processing_info.fs = config.fs;
    processing_info.filter_low = config.filter_low;
    processing_info.filter_high = config.filter_high;
    processing_info.notch_freq = config.notch_freq;
    processing_info.mad_values = mad_values;
    
    fprintf('  > Preprocessing complete.\n');
end

%% =========================================================================
%                         FILTERING FUNCTIONS
%% =========================================================================

function emg_filtered = apply_bandpass_filter(emg, fs, low_cutoff, high_cutoff)
    % APPLY_BANDPASS_FILTER 4th-order Butterworth bandpass filter
    %
    % Uses zero-phase filtering (filtfilt) to prevent phase distortion,
    % which is critical for preserving MUAP timing.
    %
    % Inputs:
    %   emg        - Raw EMG [samples x channels]
    %   fs         - Sampling frequency (Hz)
    %   low_cutoff - Lower cutoff frequency (Hz)
    %   high_cutoff - Upper cutoff frequency (Hz)
    %
    % Output:
    %   emg_filtered - Bandpass filtered EMG
    
    % Design 4th-order Butterworth filter
    filter_order = 4;
    nyquist = fs / 2;
    
    % Normalized cutoff frequencies
    Wn = [low_cutoff, high_cutoff] / nyquist;
    
    % Design filter
    [b, a] = butter(filter_order, Wn, 'bandpass');
    
    % Apply zero-phase filtering to each channel
    emg_filtered = zeros(size(emg));
    for ch = 1:size(emg, 2)
        emg_filtered(:, ch) = filtfilt(b, a, emg(:, ch));
    end
end

function emg_filtered = apply_notch_filter(emg, fs, notch_freq)
    % APPLY_NOTCH_FILTER Remove powerline interference
    %
    % Uses a narrow bandstop filter centered at the powerline frequency.
    % Filter width is 4 Hz (e.g., 58-62 Hz for 60 Hz powerline).
    %
    % Inputs:
    %   emg        - EMG data [samples x channels]
    %   fs         - Sampling frequency (Hz)
    %   notch_freq - Powerline frequency (50 or 60 Hz)
    %
    % Output:
    %   emg_filtered - Notch filtered EMG
    
    % Notch filter bandwidth (Hz)
    notch_width = 2;  % +/- 2 Hz around center frequency
    
    % Normalized frequencies
    nyquist = fs / 2;
    Wn = [(notch_freq - notch_width), (notch_freq + notch_width)] / nyquist;
    
    % Ensure valid frequency range
    Wn = max(0.001, min(0.999, Wn));
    
    % Design 2nd-order Butterworth bandstop filter
    [b, a] = butter(2, Wn, 'stop');
    
    % Apply zero-phase filtering
    emg_filtered = zeros(size(emg));
    for ch = 1:size(emg, 2)
        emg_filtered(:, ch) = filtfilt(b, a, emg(:, ch));
    end
end

function emg_filtered = apply_highpass_filter(emg, fs, cutoff)
    % APPLY_HIGHPASS_FILTER Remove low-frequency drift and DC offset
    %
    % Inputs:
    %   emg    - EMG data [samples x channels]
    %   fs     - Sampling frequency (Hz)
    %   cutoff - Cutoff frequency (Hz)
    %
    % Output:
    %   emg_filtered - Highpass filtered EMG
    
    filter_order = 4;
    nyquist = fs / 2;
    Wn = cutoff / nyquist;
    
    [b, a] = butter(filter_order, Wn, 'high');
    
    emg_filtered = zeros(size(emg));
    for ch = 1:size(emg, 2)
        emg_filtered(:, ch) = filtfilt(b, a, emg(:, ch));
    end
end

function emg_filtered = apply_lowpass_filter(emg, fs, cutoff)
    % APPLY_LOWPASS_FILTER Anti-aliasing or smoothing filter
    %
    % Inputs:
    %   emg    - EMG data [samples x channels]
    %   fs     - Sampling frequency (Hz)
    %   cutoff - Cutoff frequency (Hz)
    %
    % Output:
    %   emg_filtered - Lowpass filtered EMG
    
    filter_order = 4;
    nyquist = fs / 2;
    Wn = cutoff / nyquist;
    
    [b, a] = butter(filter_order, Wn, 'low');
    
    emg_filtered = zeros(size(emg));
    for ch = 1:size(emg, 2)
        emg_filtered(:, ch) = filtfilt(b, a, emg(:, ch));
    end
end

%% =========================================================================
%                         NORMALIZATION FUNCTIONS
%% =========================================================================

function [emg_normalized, mad_values] = apply_mad_normalization(emg)
    % APPLY_MAD_NORMALIZATION Median Absolute Deviation normalization
    %
    % MAD is more robust to outliers than standard deviation, making it
    % better suited for EMG signals that may contain motion artifacts.
    %
    % Formula: x_norm = (x - median(x)) / MAD(x)
    %
    % Inputs:
    %   emg - EMG data [samples x channels]
    %
    % Outputs:
    %   emg_normalized - Normalized EMG
    %   mad_values     - MAD value for each channel
    
    n_channels = size(emg, 2);
    emg_normalized = zeros(size(emg));
    mad_values = zeros(1, n_channels);
    
    for ch = 1:n_channels
        channel_data = emg(:, ch);
        
        % Compute median
        med = median(channel_data);
        
        % Compute MAD (Median Absolute Deviation)
        mad_val = median(abs(channel_data - med));
        
        % Prevent division by zero
        if mad_val < eps
            mad_val = 1;
        end
        
        % Normalize
        emg_normalized(:, ch) = (channel_data - med) / mad_val;
        mad_values(ch) = mad_val;
    end
end

function [emg_normalized, params] = apply_zscore_normalization(emg, params)
    % APPLY_ZSCORE_NORMALIZATION Standard z-score normalization
    %
    % Formula: x_norm = (x - mean(x)) / std(x)
    %
    % Inputs:
    %   emg    - EMG data [samples x channels]
    %   params - (Optional) Pre-computed mean and std
    %
    % Outputs:
    %   emg_normalized - Normalized EMG
    %   params         - Normalization parameters
    
    if nargin < 2 || isempty(params)
        params.mu = mean(emg, 1);
        params.sigma = std(emg, 0, 1);
        params.sigma(params.sigma < eps) = 1;
    end
    
    emg_normalized = (emg - params.mu) ./ params.sigma;
end

function [emg_normalized, params] = apply_minmax_normalization(emg, params)
    % APPLY_MINMAX_NORMALIZATION Scale to [0, 1] range
    %
    % Inputs:
    %   emg    - EMG data [samples x channels]
    %   params - (Optional) Pre-computed min/max values
    %
    % Outputs:
    %   emg_normalized - Normalized EMG in [0, 1]
    %   params         - Normalization parameters
    
    if nargin < 2 || isempty(params)
        params.min_val = min(emg, [], 1);
        params.max_val = max(emg, [], 1);
        params.range = params.max_val - params.min_val;
        params.range(params.range < eps) = 1;
    end
    
    emg_normalized = (emg - params.min_val) ./ params.range;
end

function [emg_normalized, mvc_values] = apply_mvc_normalization(emg, mvc_values)
    % APPLY_MVC_NORMALIZATION Normalize to Maximum Voluntary Contraction
    %
    % Clinical standard for comparing EMG across subjects.
    % Requires MVC values from calibration session.
    %
    % Inputs:
    %   emg        - EMG data [samples x channels]
    %   mvc_values - MVC value for each channel [1 x channels]
    %
    % Output:
    %   emg_normalized - EMG as percentage of MVC
    
    if nargin < 2 || isempty(mvc_values)
        % Estimate MVC as 95th percentile of RMS in 200ms windows
        warning('MVC values not provided. Estimating from data.');
        mvc_values = prctile(abs(emg), 95, 1);
    end
    
    mvc_values(mvc_values < eps) = 1;
    emg_normalized = emg ./ mvc_values * 100;  % Percentage of MVC
end

%% =========================================================================
%                         SIGNAL QUALITY ASSESSMENT
%% =========================================================================

function [snr_per_channel, bad_channels] = assess_signal_quality(emg, stimulus, config)
    % ASSESS_SIGNAL_QUALITY Compute SNR and identify bad channels
    %
    % SNR is computed as the ratio of signal power during active gestures
    % to noise power during rest periods.
    %
    % Inputs:
    %   emg      - Preprocessed EMG [samples x channels]
    %   stimulus - Gesture labels [samples x 1]
    %   config   - Configuration struct
    %
    % Outputs:
    %   snr_per_channel - SNR in dB for each channel
    %   bad_channels    - Indices of channels with SNR < 3 dB
    
    n_channels = size(emg, 2);
    snr_per_channel = zeros(1, n_channels);
    
    % Identify rest and active periods
    rest_idx = stimulus == 0;
    active_idx = stimulus > 0;
    
    for ch = 1:n_channels
        channel_data = emg(:, ch);
        
        % Compute power during rest (noise floor)
        if any(rest_idx)
            noise_power = mean(channel_data(rest_idx).^2);
        else
            noise_power = prctile(channel_data.^2, 10);  % Use 10th percentile
        end
        
        % Compute power during activity (signal + noise)
        if any(active_idx)
            signal_power = mean(channel_data(active_idx).^2);
        else
            signal_power = mean(channel_data.^2);
        end
        
        % Compute SNR in dB
        if noise_power > eps
            snr_per_channel(ch) = 10 * log10(signal_power / noise_power);
        else
            snr_per_channel(ch) = Inf;
        end
    end
    
    % Identify bad channels (SNR < 3 dB)
    snr_threshold = 3;
    bad_channels = find(snr_per_channel < snr_threshold);
end

function quality_metrics = compute_quality_metrics(emg, fs)
    % COMPUTE_QUALITY_METRICS Comprehensive signal quality assessment
    %
    % Computes multiple quality indicators for each channel.
    %
    % Inputs:
    %   emg - EMG data [samples x channels]
    %   fs  - Sampling frequency (Hz)
    %
    % Output:
    %   quality_metrics - Struct with quality indicators per channel
    
    n_channels = size(emg, 2);
    
    quality_metrics.baseline_noise = zeros(1, n_channels);
    quality_metrics.powerline_ratio = zeros(1, n_channels);
    quality_metrics.high_freq_content = zeros(1, n_channels);
    quality_metrics.saturation_pct = zeros(1, n_channels);
    
    for ch = 1:n_channels
        channel_data = emg(:, ch);
        
        % Baseline noise (std of lowest 10% amplitude periods)
        sorted_abs = sort(abs(channel_data));
        low_10pct = sorted_abs(1:round(length(sorted_abs) * 0.1));
        quality_metrics.baseline_noise(ch) = std(low_10pct);
        
        % Powerline contamination (power at 50/60 Hz relative to total)
        nfft = 2^nextpow2(length(channel_data));
        [psd, f] = pwelch(channel_data, [], [], nfft, fs);
        
        % Find power at 50 and 60 Hz
        [~, idx_50] = min(abs(f - 50));
        [~, idx_60] = min(abs(f - 60));
        powerline_power = max(psd(idx_50), psd(idx_60));
        total_power = sum(psd);
        quality_metrics.powerline_ratio(ch) = powerline_power / total_power;
        
        % High frequency content (>400 Hz, potential noise/aliasing)
        high_freq_idx = f > 400;
        quality_metrics.high_freq_content(ch) = sum(psd(high_freq_idx)) / total_power;
        
        % Saturation percentage (samples at max/min values)
        max_val = max(abs(channel_data));
        saturated = abs(channel_data) > 0.99 * max_val;
        quality_metrics.saturation_pct(ch) = sum(saturated) / length(channel_data) * 100;
    end
end

%% =========================================================================
%                         ARTIFACT DETECTION
%% =========================================================================

function [artifact_mask, artifact_info] = detect_artifacts(emg, fs, config)
    % DETECT_ARTIFACTS Identify motion artifacts and transients
    %
    % Uses amplitude thresholding and derivative analysis to detect
    % non-physiological signal components.
    %
    % Inputs:
    %   emg    - EMG data [samples x channels]
    %   fs     - Sampling frequency (Hz)
    %   config - Configuration with artifact thresholds
    %
    % Outputs:
    %   artifact_mask - Logical array [samples x channels], true = artifact
    %   artifact_info - Struct with artifact statistics
    
    if ~isfield(config, 'artifact_threshold')
        config.artifact_threshold = 5;  % MAD units
    end
    if ~isfield(config, 'derivative_threshold')
        config.derivative_threshold = 10;  % MAD units
    end
    
    n_samples = size(emg, 1);
    n_channels = size(emg, 2);
    
    artifact_mask = false(n_samples, n_channels);
    artifact_info.n_artifacts = zeros(1, n_channels);
    artifact_info.artifact_pct = zeros(1, n_channels);
    
    for ch = 1:n_channels
        channel_data = emg(:, ch);
        
        % Amplitude-based detection
        med = median(abs(channel_data));
        mad_val = median(abs(abs(channel_data) - med));
        amplitude_artifact = abs(channel_data) > med + config.artifact_threshold * mad_val;
        
        % Derivative-based detection (sudden changes)
        deriv = abs(diff(channel_data));
        deriv_med = median(deriv);
        deriv_mad = median(abs(deriv - deriv_med));
        derivative_artifact = [false; deriv > deriv_med + config.derivative_threshold * deriv_mad];
        
        % Combine
        artifact_mask(:, ch) = amplitude_artifact | derivative_artifact;
        
        % Expand artifact regions by 10 samples on each side
        expand_samples = round(0.01 * fs);  % 10 ms
        artifact_expanded = artifact_mask(:, ch);
        for i = 1:expand_samples
            artifact_expanded = artifact_expanded | [false; artifact_mask(1:end-1, ch)] | ...
                                                   [artifact_mask(2:end, ch); false];
        end
        artifact_mask(:, ch) = artifact_expanded;
        
        % Statistics
        artifact_info.n_artifacts(ch) = sum(diff([0; artifact_mask(:, ch)]) == 1);
        artifact_info.artifact_pct(ch) = sum(artifact_mask(:, ch)) / n_samples * 100;
    end
end

function emg_clean = remove_artifacts(emg, artifact_mask, method)
    % REMOVE_ARTIFACTS Replace artifact regions
    %
    % Methods:
    %   'interpolate' - Linear interpolation across artifact
    %   'zero'        - Replace with zeros
    %   'median'      - Replace with channel median
    %
    % Inputs:
    %   emg           - EMG data [samples x channels]
    %   artifact_mask - Logical artifact indicator [samples x channels]
    %   method        - Replacement method string
    %
    % Output:
    %   emg_clean - Artifact-corrected EMG
    
    if nargin < 3
        method = 'interpolate';
    end
    
    emg_clean = emg;
    n_channels = size(emg, 2);
    
    for ch = 1:n_channels
        artifact_idx = find(artifact_mask(:, ch));
        
        if isempty(artifact_idx)
            continue;
        end
        
        switch lower(method)
            case 'interpolate'
                % Find clean samples for interpolation
                clean_idx = find(~artifact_mask(:, ch));
                if length(clean_idx) > 1
                    emg_clean(artifact_idx, ch) = interp1(clean_idx, ...
                        emg(clean_idx, ch), artifact_idx, 'linear', 'extrap');
                end
                
            case 'zero'
                emg_clean(artifact_idx, ch) = 0;
                
            case 'median'
                emg_clean(artifact_idx, ch) = median(emg(~artifact_mask(:, ch), ch));
                
            otherwise
                error('Unknown artifact removal method: %s', method);
        end
    end
end

%% =========================================================================
%                         UTILITY FUNCTIONS
%% =========================================================================

function emg_rectified = full_wave_rectify(emg)
    % FULL_WAVE_RECTIFY Take absolute value of EMG
    %
    % Standard preprocessing step for envelope detection.
    
    emg_rectified = abs(emg);
end

function emg_envelope = compute_envelope(emg, fs, cutoff)
    % COMPUTE_ENVELOPE Linear envelope via lowpass filtering
    %
    % Rectifies signal and applies lowpass filter to extract amplitude
    % envelope, commonly used for proportional control.
    %
    % Inputs:
    %   emg    - EMG data [samples x channels]
    %   fs     - Sampling frequency (Hz)
    %   cutoff - Envelope cutoff frequency (Hz), typically 2-10 Hz
    %
    % Output:
    %   emg_envelope - Smoothed amplitude envelope
    
    if nargin < 3
        cutoff = 5;  % Default 5 Hz cutoff
    end
    
    % Rectify
    emg_rect = abs(emg);
    
    % Lowpass filter
    emg_envelope = apply_lowpass_filter(emg_rect, fs, cutoff);
end

function emg_downsampled = downsample_emg(emg, original_fs, target_fs)
    % DOWNSAMPLE_EMG Reduce sampling rate
    %
    % Applies anti-aliasing filter before downsampling.
    %
    % Inputs:
    %   emg         - EMG data [samples x channels]
    %   original_fs - Original sampling frequency (Hz)
    %   target_fs   - Target sampling frequency (Hz)
    %
    % Output:
    %   emg_downsampled - Downsampled EMG
    
    downsample_factor = round(original_fs / target_fs);
    
    if downsample_factor <= 1
        emg_downsampled = emg;
        return;
    end
    
    % Anti-aliasing filter
    nyquist_new = target_fs / 2;
    emg_filtered = apply_lowpass_filter(emg, original_fs, nyquist_new * 0.8);
    
    % Downsample
    emg_downsampled = emg_filtered(1:downsample_factor:end, :);
end

function [emg_synced, offset] = synchronize_channels(emg, reference_channel)
    % SYNCHRONIZE_CHANNELS Align channels based on cross-correlation
    %
    % Corrects for small timing offsets between channels that may occur
    % with certain acquisition systems.
    %
    % Inputs:
    %   emg               - EMG data [samples x channels]
    %   reference_channel - Channel index to use as reference
    %
    % Outputs:
    %   emg_synced - Synchronized EMG
    %   offset     - Sample offset applied to each channel
    
    if nargin < 2
        reference_channel = 1;
    end
    
    n_channels = size(emg, 2);
    max_lag = 50;  % Maximum expected offset in samples
    
    emg_synced = emg;
    offset = zeros(1, n_channels);
    
    ref_signal = emg(:, reference_channel);
    
    for ch = 1:n_channels
        if ch == reference_channel
            continue;
        end
        
        % Cross-correlation
        [xcorr_vals, lags] = xcorr(ref_signal, emg(:, ch), max_lag);
        [~, max_idx] = max(abs(xcorr_vals));
        offset(ch) = lags(max_idx);
        
        % Apply offset
        if offset(ch) > 0
            emg_synced(1:end-offset(ch), ch) = emg(offset(ch)+1:end, ch);
        elseif offset(ch) < 0
            emg_synced(-offset(ch)+1:end, ch) = emg(1:end+offset(ch), ch);
        end
    end
end
