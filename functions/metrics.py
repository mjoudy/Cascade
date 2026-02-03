import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def calculate_cumsum_slope(upsampled_spikes, reconstructed_spikes):
    """
    Calculates the slope of the fitted line in the space of 
    cumulative summation of upsampled spikes vs cumulative summation of reconstructed spikes.
    
    Args:
        upsampled_spikes (np.ndarray): Array of upsampled spike values.
        reconstructed_spikes (np.ndarray): Array of reconstructed spike values.
        
    Returns:
        float: The slope of the linear fit.
    """
    # Ensure inputs are numpy arrays
    upsampled_spikes = np.asarray(upsampled_spikes)
    reconstructed_spikes = np.asarray(reconstructed_spikes)
    
    cs_upsampled = np.cumsum(upsampled_spikes)
    cs_rec = np.cumsum(reconstructed_spikes)
    
    # Fit a line: y = mx + c
    # x = cs_upsampled, y = cs_rec
    # We want m.
    
    # robust check for singular matrix or empty data
    if len(cs_upsampled) == 0 or np.all(cs_upsampled == cs_upsampled[0]):
        return np.nan
        
    slope, intercept = np.polyfit(cs_upsampled, cs_rec, 1)
    return slope

def cumsum_slope(spikes_train, reconstructed_spikes):
    """
    Calculates the slope of the linear fit between the cumulative sums of the spike train
    and the reconstructed spikes.
    
    Args:
        spikes_train (np.ndarray): Binary spike train.
        reconstructed_spikes (np.ndarray): Reconstructed spikes trace.
        
    Returns:
        float: The slope of the linear fit (y = mx + c), where y is cumsum(reconstructed)
               and x is cumsum(spikes_train).
    """

    spikes_z = zscore(reconstructed_spikes)
    x = np.cumsum(spikes_train)
    y = np.cumsum(spikes_z)
    
    # Handle empty or trivial cases
    if len(x) < 2 or len(y) < 2:
        return np.nan
        
    # If x is constant (e.g. no spikes), slope is undefined/infinite or handled gracefully
    if np.all(x == x[0]):
         return np.nan

    slope, intercept = np.polyfit(x, y, 1)
    return slope

def plot_binned_comparison(true_spikes_train, reconstructed_spikes, bin_size=50, sampling_rate=None, plot=True, ax=None):
    """
    Bins the true and reconstructed spike trains and plots their superimposed summation.
    Calculates Pearson correlation between the binned traces.
    
    Args:
        true_spikes_train (np.ndarray): Binary spike train (1s and 0s).
        reconstructed_spikes (np.ndarray): Reconstructed continuous spike signal.
        bin_size (int): Size of the bin in samples (data points).
        sampling_rate (float, optional): Sampling rate in Hz. If provided, x-axis will be in seconds.
        plot (bool): Whether to generate the plot.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None and plot is True, creates new figure.
        
    Returns:
        float: Pearson correlation coefficient.
    """
    
    # Ensure bin_size is at least 1
    bin_width = int(bin_size)
    if bin_width < 1:
        bin_width = 1
        
    # Determine the length that is a multiple of bin_width
    n_bins = len(true_spikes_train) // bin_width
    limit = n_bins * bin_width
    
    # Truncate to limit
    t_spikes_trunc = true_spikes_train[:limit]
    r_spikes_trunc = reconstructed_spikes[:limit]
    
    # Reshape and sum
    # Shape: (n_bins, bin_width) -> sum along axis 1
    binned_true = t_spikes_trunc.reshape(n_bins, bin_width).sum(axis=1)
    binned_rec = r_spikes_trunc.reshape(n_bins, bin_width).sum(axis=1)
    
    # Calculate Correlation
    if np.std(binned_true) == 0 or np.std(binned_rec) == 0:
        correlation = np.nan
    else:
        correlation, _ = pearsonr(binned_true, binned_rec)
    
    if plot:
        if ax is None:
            plt.figure(figsize=(15, 6))
            ax = plt.gca()
            show = True
        else:
            show = False
            
        # Calculate X-axis
        # If sampling_rate is provided, convert to seconds
        if sampling_rate:
            # Time per bin in seconds
            dt_bin = bin_width / sampling_rate
            time_axis = np.arange(n_bins) * dt_bin
            xlabel = 'Time (s)'
            bin_label_width = dt_bin
            step_offset = dt_bin / 2
        else:
            # Use samples/indices
            time_axis = np.arange(n_bins) * bin_width
            xlabel = 'Time (samples)'
            bin_label_width = bin_width
            step_offset = bin_width / 2

        # Plot True Spikes (Bar)
        ax.bar(time_axis, binned_true, width=bin_label_width, alpha=0.3, color='gray', label='True Spikes', align='edge')
        
        # Plot Reconstructed Spikes (Step) with visual shift correction
        ax.step(time_axis + step_offset, binned_rec, where='mid', color='#e67e22', label='Reconstructed Spikes')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Summed Activity per Bin')
        ax.set_title(f'Binned Spike Comparison (Bin Size: {bin_width}) | r = {correlation:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.show()
        
    return correlation

def analyze_cumsum_calibration(true_spikes_train, reconstructed_spikes):
    """
    Performs comprehensive cumulative sum analysis for spike train calibration.
    
    1. Calculates slope/R2 for Raw reconstruction vs True spikes.
    2. Calibrates (normalizes) the reconstruction using the calculated slope.
    3. Calculates slope/R2 for Calibrated reconstruction vs True spikes.
    
    Args:
        true_spikes_train (np.ndarray): Binary spike train (0s and 1s).
        reconstructed_spikes (np.ndarray): Continuous reconstructed trace (raw).
        
    Returns:
        dict: containing:
            - 'slope_raw': Slope of CumSum(Rec) vs CumSum(True).
            - 'r2_raw': R2 score for Raw.
            - 'pearson_raw': Pearson correlation for Raw.
            - 'calibrated_trace': The calibrated reconstructed spikes trace.
            - 'slope_calib': Slope after calibration (should be ~1.0).
            - 'r2_calib': R2 score after calibration.
            - 'pearson_calib': Pearson correlation after calibration.
    """
       
    # --- Helper Inner Function for Single Pass Analysis ---
    def _analyze_single_pass(true_train, rec_trace):
        x = np.cumsum(true_train).reshape(-1, 1)  # Reshape for sklearn (N, 1)
        y = np.cumsum(rec_trace)
        
        # Avoid division by zero / empty
        if len(x) == 0:
            return np.nan, np.nan, np.nan
            
        # Linear Regression forced through origin (fit_intercept=False)
        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)
        
        slope = model.coef_[0]
        y_pred = model.predict(x)
        
        # Calculate R2
        r2 = r2_score(y, y_pred)
            
        # Pearson
        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
            corr, _ = pearsonr(x.flatten(), y)
        else:
            corr = np.nan
            
        return slope, r2, corr

    # 1. Analyze Raw
    slope_raw, r2_raw, pearson_raw = _analyze_single_pass(true_spikes_train, reconstructed_spikes)
    
    # 2. Calibrate
    if np.isnan(slope_raw) or slope_raw == 0:
        calibrated_trace = np.zeros_like(reconstructed_spikes)
    else:
        calibrated_trace = reconstructed_spikes / slope_raw
        
    # 3. Analyze Calibrated
    slope_calib, r2_calib, pearson_calib = _analyze_single_pass(true_spikes_train, calibrated_trace)
    
    return {
        'slope_raw': slope_raw,
        'r2_raw': r2_raw,
        'pearson_raw': pearson_raw,
        'calibrated_trace': calibrated_trace,
        'slope_calib': slope_calib,
        'r2_calib': r2_calib,
        'pearson_calib': pearson_calib
    }

def plot_calibration_dashboard(true_spikes_train, reconstructed_spikes, calibration_results, bin_size=50, sampling_rate=None):
    """
    Plots a 4-panel dashboard using pre-calculated calibration results.
    
    Args:
        true_spikes_train (np.ndarray): Binary spike train.
        reconstructed_spikes (np.ndarray): Raw reconstructed trace.
        calibration_results (dict): Output from analyze_cumsum_calibration.
        bin_size (int): Bin size in samples.
        sampling_rate (float, optional): Sampling rate in Hz.
    """
    
    # Extract metrics
    slope_raw = calibration_results['slope_raw']
    r2_raw = calibration_results['r2_raw']
    slope_calib = calibration_results['slope_calib']
    r2_calib = calibration_results['r2_calib']
    calibrated_spikes = calibration_results['calibrated_trace']
    
    # Prepare Plot Data
    x_cumsum = np.cumsum(true_spikes_train)
    y_cumsum_raw = np.cumsum(reconstructed_spikes)
    y_cumsum_calib = np.cumsum(calibrated_spikes)
    
    if sampling_rate:
        time_axis = np.arange(len(true_spikes_train)) / sampling_rate
        xlabel_trace = 'Time (s)'
    else:
        time_axis = np.arange(len(true_spikes_train))
        xlabel_trace = 'Time (samples)'
        
    true_spike_times = time_axis[np.where(true_spikes_train > 0)[0]]

    # Plotting
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    
    # Panel 1: Binned Comparison (Top)
    ax1 = fig.add_subplot(gs[0, :])
    plot_binned_comparison(
        true_spikes_train, 
        calibrated_spikes, 
        bin_size=bin_size, 
        sampling_rate=sampling_rate, 
        plot=True, 
        ax=ax1
    )
    ax1.set_title(f"1. Binned Spikes Comparison (Calibrated | Bin Size: {bin_size})")

    # Panel 2: Trace Comparison (Middle)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time_axis, calibrated_spikes, color='black', alpha=0.8, label='Calibrated Rec Trace')
    
    ymin, ymax = np.min(calibrated_spikes), np.max(calibrated_spikes)
    range_y = ymax - ymin if ymax != ymin else 1.0
    ax2.vlines(true_spike_times, ymin=ymax, ymax=ymax + 0.1*range_y, color='red', alpha=0.6, label='True Spikes')
    
    ax2.set_xlabel(xlabel_trace)
    ax2.set_ylabel('Amplitude (Calibrated)')
    ax2.set_title("2. Reconstructed Trace vs True Spike Events")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: CumSum Raw (Bottom Left)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(x_cumsum, y_cumsum_raw, color='C0', label='Data')
    if not np.isnan(slope_raw):
        ax3.plot(x_cumsum, slope_raw * x_cumsum, 'k--', alpha=0.7, label=f'Fit (Slope={slope_raw:.4f})')
    
    ax3.set_title(f"3. Raw: CumSum(Rec) vs CumSum(True)\nR2={r2_raw:.4f}, Slope={slope_raw:.4f}")
    ax3.set_xlabel('CumSum(True Spikes)')
    ax3.set_ylabel('CumSum(Raw Rec)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: CumSum Calibrated (Bottom Right)
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(x_cumsum, y_cumsum_calib, color='C1', label='Data')
    if not np.isnan(slope_calib):
        ax4.plot(x_cumsum, slope_calib * x_cumsum, 'k--', alpha=0.7, label=f'Fit (Slope={slope_calib:.4f})')
        
    ax4.set_title(f"4. Calibrated: CumSum(Rec) vs CumSum(True)\nR2={r2_calib:.4f}, Slope={slope_calib:.4f}")
    ax4.set_xlabel('CumSum(True Spikes)')
    ax4.set_ylabel('CumSum(Calibrated Rec)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_cross_corr_max(sig1, sig2):
    """
    Calculates the maximum cross-correlation and the lag at which it occurs.
    
    Args:
        sig1 (np.ndarray): Reference signal (e.g., True Calcium).
        sig2 (np.ndarray): Test signal (e.g., Reconstructed Calcium).
        
    Returns:
        tuple: (max_correlation, lag_in_samples)
    """
    # Handle NaNs or infinite values by replacing them with 0 or mean
    if np.any(np.isnan(sig1)) or np.any(np.isinf(sig1)):
        return np.nan, 0
    if np.any(np.isnan(sig2)) or np.any(np.isinf(sig2)):
        return np.nan, 0
        
    # Standardize (Z-score) to normalize amplitude
    if np.std(sig1) == 0 or np.std(sig2) == 0:
         return np.nan, 0

    sig1_z = (sig1 - np.mean(sig1)) / np.std(sig1)
    sig2_z = (sig2 - np.mean(sig2)) / np.std(sig2)
    
    # Calculate cross-correlation
    # 'full' mode returns interaction at all lags
    corr = np.correlate(sig1_z, sig2_z, mode='full') / len(sig1)
    
    # Create lag array matching the correlation output
    lags = np.arange(-(len(sig1) - 1), len(sig1))
    
    # Find index of maximum absolute correlation
    max_idx = np.argmax(np.abs(corr))
    max_corr = corr[max_idx]
    lag = lags[max_idx]
    
    return max_corr, lag
