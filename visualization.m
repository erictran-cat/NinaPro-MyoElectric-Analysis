%% =========================================================================
%  VISUALIZATION MODULE
%  -------------------------------------------------------------------------
%  NinaPro sEMG Processing Pipeline
%  
%  This standalone module provides publication-quality visualization
%  functions for sEMG analysis results.
%
%  Design Philosophy:
%    - Clean, minimalist aesthetic following IEEE/Nature guidelines
%    - Consistent styling via centralized configuration
%    - Print-friendly and colorblind-safe color palettes
%    - Multi-format export (PNG, PDF, FIG)
%
%  Usage:
%    generate_all_figures(FeatureMatrix, labels, snr, bad_ch, config)
%
%  Required Toolboxes:
%    - Statistics and Machine Learning Toolbox (pca, boxplot)
%
%  =========================================================================

function generate_all_figures(FeatureMatrix, labels, snr_per_channel, ...
    bad_channels, config)
    % GENERATE_ALL_FIGURES Main entry point for visualization
    %
    % Generates all publication-quality figures and exports in multiple formats.
    
    fprintf('Generating publication figures...\n');
    
    % Get style configuration
    style = get_publication_style();
    
    % Prepare data
    feature_array = table2array(FeatureMatrix);
    unique_gestures = unique(labels(labels > 0));
    n_gestures = numel(unique_gestures);
    n_channels = config.num_channels;
    
    active_idx = labels > 0;
    active_features = feature_array(active_idx, :);
    active_labels = labels(active_idx);
    
    % Generate each figure
    fig1 = create_spatial_activation_map(feature_array, labels, ...
        unique_gestures, n_gestures, n_channels, style);
    export_figure(fig1, 'Fig1_Spatial_Activation_Map', style);
    
    fig2 = create_pca_plot(active_features, active_labels, ...
        unique_gestures, n_gestures, style);
    export_figure(fig2, 'Fig2_PCA_Separability', style);
    
    fig3 = create_snr_plot(snr_per_channel, bad_channels, n_channels, style);
    export_figure(fig3, 'Fig3_Signal_Quality_SNR', style);
    
    fig4 = create_feature_boxplots(feature_array, active_idx, ...
        active_labels, n_gestures, style);
    export_figure(fig4, 'Fig4_Feature_Distributions', style);
    
    fig5 = create_distance_matrix(active_features, active_labels, ...
        unique_gestures, n_gestures, style);
    export_figure(fig5, 'Fig5_Gesture_Distance_Matrix', style);
    
    fprintf('  > All figures exported successfully.\n');
end

%% =========================================================================
%                         STYLE CONFIGURATION
%% =========================================================================

function style = get_publication_style()
    % GET_PUBLICATION_STYLE Centralized style configuration
    
    % Font Settings
    style.font.name = 'Arial';
    style.font.size.axis = 12;
    style.font.size.label = 14;
    style.font.size.title = 16;
    style.font.size.annotation = 11;
    style.font.weight.title = 'bold';
    
    % Color Palette (ColorBrewer-based, colorblind-safe)
    style.colors.primary = [
        0.122, 0.467, 0.706;
        0.890, 0.102, 0.110;
        0.173, 0.627, 0.173;
        1.000, 0.498, 0.000;
        0.580, 0.404, 0.741;
        0.549, 0.337, 0.294;
        0.891, 0.467, 0.761;
        0.498, 0.498, 0.498;
        0.737, 0.741, 0.133;
        0.090, 0.745, 0.812;
    ];
    
    style.colors.good = [0.173, 0.627, 0.173];
    style.colors.bad = [0.890, 0.102, 0.110];
    style.colors.warning = [1.000, 0.498, 0.000];
    style.colors.neutral = [0.498, 0.498, 0.498];
    
    % Line Settings
    style.line.width.primary = 2.0;
    style.line.width.secondary = 1.5;
    style.line.width.reference = 1.0;
    style.line.width.axis = 1.0;
    
    % Marker Settings
    style.marker.size = 50;
    style.marker.alpha = 0.7;
    
    % Figure Dimensions
    style.figure.width = 800;
    style.figure.height = 500;
    
    % Axis Settings
    style.axis.grid.alpha = 0.3;
    
    % Export Settings
    style.export.dpi = 300;
end

function colors = get_extended_palette(n)
    % GET_EXTENDED_PALETTE Generate n visually distinct colors
    hues = linspace(0, 1, n + 1);
    hues = hues(1:n);
    colors = zeros(n, 3);
    for i = 1:n
        h = hues(i); s = 0.7; l = 0.5;
        if l < 0.5, q = l * (1 + s); else, q = l + s - l * s; end
        p = 2 * l - q;
        colors(i, :) = [hue_to_rgb(p,q,h+1/3), hue_to_rgb(p,q,h), hue_to_rgb(p,q,h-1/3)];
    end
end

function c = hue_to_rgb(p, q, t)
    if t < 0, t = t + 1; end
    if t > 1, t = t - 1; end
    if t < 1/6, c = p + (q - p) * 6 * t;
    elseif t < 1/2, c = q;
    elseif t < 2/3, c = p + (q - p) * (2/3 - t) * 6;
    else, c = p; end
end

%% =========================================================================
%                         FIGURE CREATION FUNCTIONS
%% =========================================================================

function fig = create_spatial_activation_map(feature_array, labels, ...
    unique_gestures, n_gestures, n_channels, style)
    % CREATE_SPATIAL_ACTIVATION_MAP Muscle synergy heatmap
    
    rms_col_idx = 2:6:size(feature_array, 2);
    activation_map = zeros(n_gestures, n_channels);
    
    for g = 1:n_gestures
        gesture_idx = labels == unique_gestures(g);
        for ch = 1:min(n_channels, numel(rms_col_idx))
            activation_map(g, ch) = mean(feature_array(gesture_idx, rms_col_idx(ch)));
        end
    end
    activation_map = activation_map ./ max(activation_map(:));
    
    fig = figure('Position', [100, 100, style.figure.width, style.figure.height], 'Color', 'w');
    imagesc(activation_map);
    colormap(parula);
    
    cb = colorbar;
    cb.Label.String = 'Normalized RMS Activation';
    cb.Label.FontSize = style.font.size.label;
    cb.Label.FontName = style.font.name;
    
    ax = gca;
    apply_axis_style(ax, style);
    
    xlabel('Electrode Channel', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    ylabel('Gesture Class', 'FontSize', style.font.size.label, 'FontName', style.font.name);
    title('Spatial Activation Map: Muscle Synergy Patterns', ...
        'FontSize', style.font.size.title, 'FontName', style.font.name, ...
        'FontWeight', style.font.weight.title);
    
    xticks(1:n_channels);
    xticklabels(arrayfun(@(x) sprintf('Ch%d', x), 1:n_channels, 'UniformOutput', false));
    yticks(1:n_gestures);
    yticklabels(arrayfun(@(x) sprintf('G%02d', x), unique_gestures, 'UniformOutput', false));
    
    for g = 1:n_gestures
        for ch = 1:n_channels
            val = activation_map(g, ch);
            if val > 0.5, txt_color = 'k'; else, txt_color = 'w'; end
            text(ch, g, sprintf('%.2f', val), 'HorizontalAlignment', 'center', ...
                'Color', txt_color, 'FontSize', style.font.size.annotation - 2, ...
                'FontWeight', 'bold');
        end
    end
    axis tight;
end

function fig = create_pca_plot(active_features, active_labels, unique_gestures, n_gestures, style)
    % CREATE_PCA_PLOT 3D PCA visualization with scree plot
    
    features_std = zscore(active_features);
    features_std(isnan(features_std)) = 0;
    [~, score, ~, ~, explained] = pca(features_std);
    
    fig = figure('Position', [100, 100, style.figure.width + 200, style.figure.height + 100], 'Color', 'w');
    t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % 3D Scatter
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
        scatter_handles(g) = scatter3(score(gesture_idx, 1), score(gesture_idx, 2), ...
            score(gesture_idx, 3), style.marker.size, colors(g, :), 'filled', ...
            'MarkerEdgeColor', 'k', 'MarkerEdgeAlpha', 0.3, ...
            'MarkerFaceAlpha', style.marker.alpha, ...
            'DisplayName', sprintf('G%02d', unique_gestures(g)));
    end
    hold off;
    
    ax = gca; apply_axis_style(ax, style);
    xlabel(sprintf('PC1 (%.1f%%)', explained(1)), 'FontSize', style.font.size.label);
    ylabel(sprintf('PC2 (%.1f%%)', explained(2)), 'FontSize', style.font.size.label);
    zlabel(sprintf('PC3 (%.1f%%)', explained(3)), 'FontSize', style.font.size.label);
    title('PCA: Class Separability', 'FontSize', style.font.size.title, 'FontWeight', 'bold');
    legend(scatter_handles, 'Location', 'eastoutside', 'FontSize', style.font.size.annotation - 2);
    grid on; view(45, 30);
    
    % Scree Plot
    nexttile;
    n_comp = min(10, numel(explained));
    bar(1:n_comp, explained(1:n_comp), 'FaceColor', style.colors.primary(1, :), 'EdgeColor', 'k');
    hold on;
    plot(1:n_comp, cumsum(explained(1:n_comp)), '-o', 'Color', style.colors.primary(2, :), ...
        'LineWidth', style.line.width.primary, 'MarkerFaceColor', style.colors.primary(2, :));
    yline(80, '--', 'Color', style.colors.neutral, 'LineWidth', 1);
    hold off;
    ax = gca; apply_axis_style(ax, style);
    xlabel('PC'); ylabel('Variance (%)');
    title('Scree Plot', 'FontWeight', 'bold');
    legend({'Individual', 'Cumulative'}, 'Location', 'east');
    ylim([0, 105]); xticks(1:n_comp);
    
    % Summary
    nexttile; axis off;
    text(0.1, 0.9, {'\bf{Summary}', '', sprintf('Gestures: %d', n_gestures), ...
        sprintf('Samples: %d', size(active_features, 1)), ...
        sprintf('PC1-3: %.1f%%', sum(explained(1:min(3,numel(explained)))))}, ...
        'VerticalAlignment', 'top', 'FontSize', style.font.size.annotation, 'Interpreter', 'tex');
end

function fig = create_snr_plot(snr_per_channel, bad_channels, n_channels, style)
    % CREATE_SNR_PLOT Signal quality bar chart
    
    fig = figure('Position', [100, 100, style.figure.width, style.figure.height - 100], 'Color', 'w');
    
    bar_colors = zeros(n_channels, 3);
    for ch = 1:n_channels
        if snr_per_channel(ch) >= 10, bar_colors(ch, :) = style.colors.good;
        elseif snr_per_channel(ch) >= 3, bar_colors(ch, :) = style.colors.warning;
        else, bar_colors(ch, :) = style.colors.bad; end
    end
    
    b = bar(1:n_channels, snr_per_channel, 'FaceColor', 'flat', 'EdgeColor', 'k');
    b.CData = bar_colors;
    
    hold on;
    yline(3, '--', 'Color', style.colors.bad, 'LineWidth', 1, 'Label', 'Min (3 dB)');
    yline(10, '--', 'Color', style.colors.good, 'LineWidth', 1, 'Label', 'Target (10 dB)');
    hold off;
    
    ax = gca; apply_axis_style(ax, style);
    xlabel('Electrode Channel', 'FontSize', style.font.size.label);
    ylabel('SNR (dB)', 'FontSize', style.font.size.label);
    title('Signal Quality: SNR per Channel', 'FontSize', style.font.size.title, 'FontWeight', 'bold');
    xticks(1:n_channels);
    xticklabels(arrayfun(@(x) sprintf('Ch%d', x), 1:n_channels, 'UniformOutput', false));
    ylim([0, max(snr_per_channel) * 1.25]);
    
    for ch = 1:n_channels
        text(ch, snr_per_channel(ch) + max(snr_per_channel) * 0.03, ...
            sprintf('%.1f', snr_per_channel(ch)), 'HorizontalAlignment', 'center', ...
            'FontSize', style.font.size.annotation, 'FontWeight', 'bold');
    end
    
    hold on;
    h1 = scatter(nan, nan, 100, style.colors.good, 's', 'filled', 'DisplayName', 'Good (≥10 dB)');
    h2 = scatter(nan, nan, 100, style.colors.warning, 's', 'filled', 'DisplayName', 'Marginal (3-10 dB)');
    h3 = scatter(nan, nan, 100, style.colors.bad, 's', 'filled', 'DisplayName', 'Poor (<3 dB)');
    hold off;
    legend([h1, h2, h3], 'Location', 'northeast');
end

function fig = create_feature_boxplots(feature_array, active_idx, active_labels, n_gestures, style)
    % CREATE_FEATURE_BOXPLOTS Box plots of features by gesture
    
    fig = figure('Position', [100, 100, style.figure.width + 200, style.figure.height - 100], 'Color', 'w');
    t = tiledlayout(1, 6, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    feature_names = {'MAV', 'RMS', 'WL', 'WAMP', 'MNP', 'TP'};
    ch = 1;
    
    for f = 1:6
        nexttile;
        feature_idx = (ch - 1) * 6 + f;
        boxplot(feature_array(active_idx, feature_idx), active_labels, ...
            'Colors', style.colors.primary(1, :), 'Symbol', 'o', 'OutlierSize', 4);
        
        h = findobj(gca, 'Tag', 'Box');
        for j = 1:length(h)
            patch(get(h(j), 'XData'), get(h(j), 'YData'), ...
                style.colors.primary(1, :), 'FaceAlpha', 0.3);
        end
        
        ax = gca; apply_axis_style(ax, style);
        xlabel('Gesture'); ylabel(feature_names{f});
        if n_gestures > 10, xticks(1:3:n_gestures); end
    end
    
    title(t, 'Feature Distributions by Gesture (Channel 1)', ...
        'FontSize', style.font.size.title, 'FontWeight', 'bold');
end

function fig = create_distance_matrix(active_features, active_labels, unique_gestures, n_gestures, style)
    % CREATE_DISTANCE_MATRIX Inter-class distance heatmap
    
    centroids = zeros(n_gestures, size(active_features, 2));
    for g = 1:n_gestures
        gesture_idx = active_labels == unique_gestures(g);
        centroids(g, :) = mean(active_features(gesture_idx, :), 1);
    end
    
    centroids_std = zscore(centroids);
    centroids_std(isnan(centroids_std)) = 0;
    distance_matrix = pdist2(centroids_std, centroids_std, 'euclidean');
    
    fig = figure('Position', [100, 100, style.figure.height + 50, style.figure.height], 'Color', 'w');
    imagesc(distance_matrix);
    colormap(flipud(hot(256)));
    
    cb = colorbar;
    cb.Label.String = 'Euclidean Distance';
    cb.Label.FontSize = style.font.size.label;
    
    ax = gca; apply_axis_style(ax, style);
    xlabel('Gesture Class'); ylabel('Gesture Class');
    title('Inter-Class Distance Matrix', 'FontSize', style.font.size.title, 'FontWeight', 'bold');
    xticks(1:n_gestures); yticks(1:n_gestures);
    xticklabels(arrayfun(@(x) sprintf('%d', x), unique_gestures, 'UniformOutput', false));
    yticklabels(arrayfun(@(x) sprintf('%d', x), unique_gestures, 'UniformOutput', false));
    axis square;
    
    if n_gestures <= 10
        for i = 1:n_gestures
            for j = 1:n_gestures
                val = distance_matrix(i, j);
                if val < median(distance_matrix(:)), txt_color = 'w'; else, txt_color = 'k'; end
                text(j, i, sprintf('%.1f', val), 'HorizontalAlignment', 'center', ...
                    'Color', txt_color, 'FontSize', style.font.size.annotation - 2);
            end
        end
    end
end

%% =========================================================================
%                         UTILITY FUNCTIONS
%% =========================================================================

function apply_axis_style(ax, style)
    % APPLY_AXIS_STYLE Apply consistent styling to axes
    ax.FontName = style.font.name;
    ax.FontSize = style.font.size.axis;
    ax.Box = 'off';
    ax.TickDir = 'out';
    ax.TickLength = [0.02, 0.02];
    ax.LineWidth = style.line.width.axis;
    ax.XGrid = 'on';
    ax.YGrid = 'on';
    ax.GridAlpha = style.axis.grid.alpha;
end

function export_figure(fig, filename, style)
    % EXPORT_FIGURE Save figure in multiple formats (PNG, PDF, FIG)
    exportgraphics(fig, [filename, '.png'], 'Resolution', style.export.dpi, 'BackgroundColor', 'white');
    exportgraphics(fig, [filename, '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
    savefig(fig, [filename, '.fig']);
    fprintf('    Exported: %s\n', filename);
end
