import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, zscore, linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.signal import correlate
import nitime.algorithms as tsa

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

def estimate_tau_lorentzian(signal, fs=100.0, plot=True):
    """
    Estimates the decay time constant (tau) of a calcium signal by fitting a Lorentzian model
    to the Multitaper Power Spectral Density (PSD).

    Model: P(f) = A / (1 + (2 * pi * f * tau)^2) + White_Noise

    Args:
        signal (np.ndarray): The calcium signal (1D array).
        fs (float): Sampling frequency in Hz.
        plot (bool): Whether to plot the PSD and the fit.

    Returns:
        tuple:
            - tau_fit (float): Estimated decay constant in seconds.
            - params (tuple): (A, tau, noise_floor) fitted parameters.
    """
    # 1. Preprocessing
    if len(signal) == 0 or np.std(signal) == 0:
        return np.nan, (np.nan, np.nan, np.nan)

    # 2. Multitaper PSD Estimation
    # adaptive=True provides a better estimate for signals with large dynamic range
    try:
        f, psd_mt, _ = tsa.multi_taper_psd(signal, Fs=fs, adaptive=True)
    except Exception as e:
        print(f"PSD estimation failed: {e}")
        return np.nan, (np.nan, np.nan, np.nan)

    # 3. Define Lorentzian Model
    def lorentzian(f, A, tau, noise_floor):
        return A / (1 + (2 * np.pi * f * tau)**2) + noise_floor

    # 4. Fit the model
    # We skip the 0Hz (DC) component to avoid bias
    # Initial Guess: A=max(PSD), tau=0.5s, noise=min(PSD)
    p0 = [np.max(psd_mt[1:]), 0.5, np.min(psd_mt[1:])]
    
    # Bounds: A>0, tau>0, noise>=0
    bounds = ([0, 0, 0], [np.inf, 10.0, np.inf])

    try:
        popt, _ = curve_fit(lorentzian, f[1:], psd_mt[1:], p0=p0, bounds=bounds)
        A_fit, tau_fit, noise_fit = popt
    except Exception as e:
        print(f"Lorentzian fit failed: {e}")
        return np.nan, (np.nan, np.nan, np.nan)

    # 5. Visualization
    if plot:
        plt.figure(figsize=(10, 5))
        plt.loglog(f, psd_mt, label='Multitaper PSD', alpha=0.6)
        
        # Plot fit
        if not np.isnan(tau_fit):
            plt.loglog(f, lorentzian(f, *popt), 'r--', label=f'Lorentzian Fit (τ={tau_fit:.3f}s)')
            # Mark corner frequency: fc = 1 / (2 * pi * tau)
            fc = 1 / (2 * np.pi * tau_fit)
            plt.axvline(fc, color='k', linestyle=':', label=f'Corner Freq ({fc:.2f} Hz)')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Calcium Signal PSD and Decay Estimation (Lorentzian)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()

    return tau_fit, (A_fit, tau_fit, noise_fit)


# =============================================================================
# DEPRECATED — ACF-based tau estimation methods
# These functions are no longer used in the active analysis pipeline.
# Tau estimation is now done exclusively via PSD Lorentzian fitting.
# Kept here for reference only.
# =============================================================================

def estimate_tau_acf(signal, fs, min_lag_s=0.0, max_lag_s=2.0, fit_until_drop=True, plot=False, ax=None):
    """
    Estimates the decay time constant (tau) of a calcium signal using its
    autocorrelation function (ACF) fitted with a bi-exponential decay.

    The function calculates the ACF and fits: ACF(t) = A * exp(-t/tau1) + B * exp(-t/tau2) + C
    It fits only the positive lags (t >= 0).
    It returns the larger tau (assuming it represents the calcium decay).

    Args:
        signal (np.ndarray): The 1D calcium signal array (e.g., dff or upsampled signal).
        fs (float): Sampling frequency in Hz.
        min_lag_s (float): Minimum lag in seconds to start the fit (e.g., 0.3 to skip initial effects).
        max_lag_s (float): Maximum lag in seconds to include in the fit.
        fit_until_drop (bool): If True, truncates the ACF data at the first point where
                               correlation drops below zero (i.e. fit only the initial positive correlation bump).
        plot (bool): Whether to plot the ACF and the fit.
        ax (matplotlib.axes.Axes): Optional ax to plot on.

    Returns:
        float: The estimated time constant (tau) in seconds. Returns np.nan if fit fails.
    """

    # Pre-checks
    if len(signal) == 0:
        return np.nan

    signal = np.asarray(signal)
    signal = signal - np.mean(signal) # Zero mean

    if np.std(signal) == 0:
        return np.nan

    # 1. Calculate ACF
    # Use 'full' mode and take the second half to get Positive Lags (0 to +lag)
    # Autocorrelation is symmetric, we only need to fit one side (the decay).
    acf = correlate(signal, signal, mode='full')
    acf = acf[len(acf)//2:]

    # Normalize
    if acf[0] != 0:
        acf = acf / acf[0]

    # 2. Limit lags (Max)
    max_lag_samples = int(max_lag_s * fs)
    if max_lag_samples > len(acf):
        max_lag_samples = len(acf)

    y_data = acf[:max_lag_samples]
    x_data = np.arange(len(y_data)) / fs

    # Handle "until corr drops" (Zero-crossing cutoff)
    if fit_until_drop:
        neg_indices = np.where(y_data < 0)[0]
        if len(neg_indices) > 0:
            first_neg = neg_indices[0]
            if first_neg > 0:
                y_data = y_data[:first_neg]
                x_data = x_data[:first_neg]

    # 3. Limit lags (Min)
    # Apply min_lag_s. This effectively slices the arrays but preserves the time axis values.
    min_lag_samples = int(min_lag_s * fs)
    if min_lag_samples < len(y_data):
        y_data = y_data[min_lag_samples:]
        x_data = x_data[min_lag_samples:]
    else:
        # If min lag is beyond valid data (e.g. signal drops to zero before min_lag), return NaN
        return np.nan

    if len(y_data) < 6:
        # Not enough data points to fit 5 parameters
        return np.nan

    # 4. Define Bi-Exponential Function
    def bi_exp(t, A, tau1, B, tau2, C):
        return A * np.exp(-t/tau1) + B * np.exp(-t/tau2) + C

    # 5. Fit
    # Initial Guess:
    # A=0.5, tau1=0.1s (fast), B=0.5, tau2=1.0s (slow), C=0
    p0 = [0.5, 0.1, 0.5, 1.0, 0]

    # Bounds: A, B > 0; tau > 0; C can be small
    # (A, tau1, B, tau2, C)
    bounds = ([0, 0, 0, 0, -np.inf], [np.inf, 10, np.inf, 10, np.inf])

    try:
        popt, _ = curve_fit(bi_exp, x_data, y_data, p0=p0, bounds=bounds, maxfev=2000)
        A, tau1, B, tau2, C = popt

        # We assume the user wants the dominant/slower decay
        estimated_tau = max(tau1, tau2)

    except Exception as e:
        # print(f"Fit failed: {e}")
        return np.nan

    # 6. Plotting (Optional)
    if plot:
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            show = True
        else:
            show = False

        # Plot data points
        ax.plot(x_data, y_data, 'b-', label='ACF Data (+Lags)', alpha=0.6)

        if not np.isnan(estimated_tau):
            y_fit = bi_exp(x_data, *popt)
            ax.plot(x_data, y_fit, 'r--', label=f'Bi-Exp Fit (tau={estimated_tau:.3f}s)')

        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'ACF Fit (Lags {min_lag_s}-{max_lag_s}s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

    return estimated_tau

def estimate_population_tau(signal, fs=30.0):
    """
    Estimates the decay time constant (tau) of a calcium signal using a log-linear fit
    on the autocorrelation function (ACF).

    Methodology:
    1. Preprocesses signal (mean centered, unit variance).
    2. Calculates ACF and normalizes it.
    3. Identifies the window for fitting:
       - Starts from lag 2 (to avoid immediate noise).
       - Ends where ACF drops below 0.1 or at max available lag.
    4. Performs a log-linear fit: ln(ACF) = -t/tau + C.
       - intercept (C) is not forced to 0.
    5. Validates the fit:
       - R^2 > 0.7
       - 0 < tau < 10

    Args:
        signal (np.ndarray): 1D array of calcium signal (e.g., upsampled dff).
        fs (float): Sampling frequency in Hz. Default is 30.0.

    Returns:
        tuple:
            - tau (float): Estimated time constant in seconds. np.nan if invalid.
            - plot_data (tuple): (lags_s, acf, (x_fit, y_fit_linear)) for visualization.
              x_fit and y_fit_linear are arrays for the fitted line in linear space (time vs ACF).
              Returns (None, None, None) if ACF calculation fails.
    """
    # 1. Preprocessing
    if len(signal) == 0:
        return np.nan, (None, None, None)

    signal = np.asarray(signal)
    if np.std(signal) == 0:
        return np.nan, (None, None, None)

    # Standardize: (x - mean) / std
    signal_norm = (signal - np.mean(signal)) / np.std(signal)

    # 2. ACF Calculation
    # Full correlation
    acf = correlate(signal_norm, signal_norm, mode='full')
    # Keep positive lags
    acf = acf[len(acf)//2:]
    # Normalize by length (since signal is strictly standardized, autocorrelation at lag 0 is length of signal)
    # But usually ACF is normalized to 1 at lag 0.
    if acf[0] != 0:
        acf = acf / acf[0]
    else:
        return np.nan, (None, None, None)

    # Lags in seconds
    lags = np.arange(len(acf)) / fs

    # 3. Dynamic Windowing
    # Find index where ACF drops below 0.1
    # We want to fit strictly positive correlation region before it hits noise floor
    drop_indices = np.where(acf < 0.1)[0]
    if len(drop_indices) > 0:
        idx_end = drop_indices[0]
    else:
        # If it never drops below 0.1, we take a reasonable max (e.g., 2 seconds equivalent or full length)
        # 2 seconds = 2 * fs samples
        idx_end = int(2.0 * fs)
        if idx_end > len(acf):
            idx_end = len(acf)

    # Start lag: 2 (skip lag 0 and 1 to avoid peak artifact)
    idx_start = 2

    # Fit data container for visualization
    # We'll return full ACF for plotting context
    plot_data_lags = lags
    plot_data_acf = acf
    plot_data_fit = None

    # Check if we have enough points
    if idx_end <= idx_start + 2: # At least 3 points
        return np.nan, (plot_data_lags, plot_data_acf, None)

    # Extract fit segment
    x_fit = lags[idx_start:idx_end]
    y_fit = acf[idx_start:idx_end]

    # Ensure y_fit is positive for log
    if np.any(y_fit <= 0):
         # truncate at first non-positive
         first_non_pos = np.where(y_fit <= 0)[0][0]
         if first_non_pos < 3:
             return np.nan, (plot_data_lags, plot_data_acf, None)
         x_fit = x_fit[:first_non_pos]
         y_fit = y_fit[:first_non_pos]

    # 4. Log-Linear Fit
    # ln(ACF) = slope * t + intercept
    # tau = -1 / slope
    try:
        y_fit_log = np.log(y_fit)
        slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit_log)

        # 5. Validation
        r_squared = r_value**2

        if slope >= 0: # Decay should have negative slope
            tau = np.nan
        else:
            tau = -1.0 / slope

        # Valid physical bounds and fits
        if r_squared < 0.7 or tau <= 0 or tau > 10.0:
            tau = np.nan

        # Prepare fit line for visualization (in linear space)
        # calculate fitted y values: y = exp(slope*x + intercept)
        y_fit_line = np.exp(slope * x_fit + intercept)
        plot_data_fit = (x_fit, y_fit_line)

    except Exception:
        tau = np.nan
        plot_data_fit = None

    return tau, (plot_data_lags, plot_data_acf, plot_data_fit)

def estimate_manual_tau(signal, fs=30.0, t_start=0.0, t_end=2.0, do_plot=True):
    """
    Estimates tau using a log-linear fit on the ACF with strictly manual thresholds.
    Designed for interactive/manual usage on single neurons.

    Args:
        signal (np.ndarray): The calcium signal.
        fs (float): Sampling rate in Hz.
        t_start (float): Start time (lag) for the fit window in seconds.
        t_end (float): End time (lag) for the fit window in seconds.
        do_plot (bool): If True, plots the semi-log ACF and the fitted line.

    Returns:
        float: Estimated tau (seconds).
    """
    # 1. Preprocessing & ACF
    if len(signal) == 0 or np.std(signal) == 0:
        print("Signal is empty or constant.")
        return np.nan

    # Standardize
    signal_norm = (signal - np.mean(signal)) / np.std(signal)

    # Calculate ACF
    acf = correlate(signal_norm, signal_norm, mode='full')
    acf = acf[len(acf)//2:] # Positive lags
    if acf[0] != 0:
        acf = acf / acf[0]

    lags_s = np.arange(len(acf)) / fs

    # 2. Windowing
    # Convert time thresholds to indices
    idx_start = int(t_start * fs)
    idx_end = int(t_end * fs)

    # Bounds check
    idx_start = max(0, idx_start)
    idx_end = min(len(acf), idx_end)

    if idx_end <= idx_start + 1:
        print(f"Window too small or invalid indices: {idx_start} to {idx_end}")
        return np.nan

    # Extract Fit Data
    x_fit = lags_s[idx_start:idx_end]
    y_fit = acf[idx_start:idx_end]

    # Filter for positive values only (for log)
    valid_mask = y_fit > 0
    x_fit = x_fit[valid_mask]
    y_fit = y_fit[valid_mask]

    if len(x_fit) < 2:
        print("Not enough positive data points in window for log fit.")
        return np.nan

    # 3. Fitting
    try:
        y_fit_log = np.log(y_fit)
        slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit_log)

        if slope >= 0:
            tau = np.inf # Growth or flat
        else:
            tau = -1.0 / slope

    except Exception as e:
        print(f"Fit failed: {e}")
        tau = np.nan
        slope, intercept = 0, 0

    # 4. Plotting
    if do_plot:
        plt.figure(figsize=(8, 5))

        # Plot full ACF in semi-log
        plt.semilogy(lags_s, acf, 'o-', color='navy', markersize=3, label='ACF', alpha=0.4)

        # Plot Fit Window markers
        plt.axvline(t_start, color='green', linestyle='--', label='Start')
        plt.axvline(t_end, color='red', linestyle='--', label='End')

        # Plot Fitted Line (if valid)
        if not np.isnan(tau) and len(x_fit) > 0:
            # y = exp(slope*x + intercept)
            y_line = np.exp(slope * x_fit + intercept)
            plt.semilogy(x_fit, y_line, 'r-', linewidth=2, label=f'Fit (tau={tau:.3f}s)')

        plt.xlim(0, max(t_end * 1.5, 4.0)) # Show context around fit window
        plt.ylim(bottom=1e-3) # Avoid log(0) issues in plot
        plt.title(f"Manual Tau Estimation\nWindow: {t_start}s - {t_end}s | Tau: {tau:.4f}s")
        plt.xlabel("Lag (s)")
        plt.ylabel("Autocorrelation (log)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.show()

    return tau

def estimate_manual_tau_exp(signal, fs=30.0, t_start=0.0, t_end=2.0, do_plot=True):
    """
    Estimates tau using a non-linear least squares fit (curve_fit) of a mono-exponential decay
    on the ACF: f(t) = A * exp(-t/tau) + C.
    Plots the result on a semi-log scale.

    Args:
        signal (np.ndarray): The calcium signal.
        fs (float): Sampling rate in Hz.
        t_start (float): Start time (lag) for the fit window in seconds.
        t_end (float): End time (lag) for the fit window in seconds.
        do_plot (bool): If True, plots the semi-log ACF and the fitted curve.

    Returns:
        float: Estimated tau (seconds).
        tuple: (A, tau, C) fitted parameters.
    """
    # 1. Preprocessing & ACF
    if len(signal) == 0 or np.std(signal) == 0:
        print("Signal is empty or constant.")
        return np.nan, (np.nan, np.nan, np.nan)

    # Standardize
    #signal_norm = (signal - np.mean(signal)) / np.std(signal)
    signal_norm = (signal - np.mean(signal))

    # Calculate ACF
    acf = correlate(signal_norm, signal_norm, mode='full')
    acf = acf[len(acf)//2:] # Positive lags
    if acf[0] != 0:
        acf = acf / acf[0]

    lags_s = np.arange(len(acf)) / fs

    # 2. Windowing
    idx_start = int(t_start * fs)
    idx_end = int(t_end * fs)

    idx_start = max(0, idx_start)
    idx_end = min(len(acf), idx_end)

    if idx_end <= idx_start + 2:
        print(f"Window too small: {idx_start} to {idx_end}")
        return np.nan, (np.nan, np.nan, np.nan)

    # Extract Fit Data
    x_fit = lags_s[idx_start:idx_end]
    y_fit = acf[idx_start:idx_end]

    # 3. Fitting
    # Model: A * exp(-x/tau) + C
    def mono_exp(t, A, tau, C):
        return A * np.exp(-t/tau) + C

    # Initial Guess
    # A ~ 1.0 (since normalized), tau ~ 1.0s, C ~ 0.0
    p0 = [1.0, 1.0, 0.0]
    # Bounds: A>0, tau>0
    bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

    tau_fit = np.nan
    A_fit, C_fit = np.nan, np.nan
    popt = [np.nan, np.nan, np.nan]

    try:
        # We need to handle potential Inf/NaNs in x_fit/y_fit, though should be fine
        if np.isfinite(x_fit).all() and np.isfinite(y_fit).all():
            popt, pcov = curve_fit(mono_exp, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=5000)
            A_fit, tau_fit, C_fit = popt
        else:
             print("Data contains NaNs or Infs.")

    except Exception as e:
        print(f"Curve fit failed: {e}")
        tau_fit = np.nan

    # 4. Plotting
    if do_plot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # --- LEFT PANEL: Semi-Log Scale ---
        ax1 = axes[0]
        # Data
        ax1.semilogy(lags_s, acf, 'o-', color='navy', markersize=3, label='ACF (Data)', alpha=0.4)

        # Window Limits
        ax1.axvline(t_start, color='green', linestyle='--', label='Start')
        ax1.axvline(t_end, color='red', linestyle='--', label='End')

        # Fit Curve
        if not np.isnan(tau_fit):
            y_fit_curve = mono_exp(x_fit, *popt)
            ax1.semilogy(x_fit, y_fit_curve, 'r-', linewidth=2.5, label=f'Exp Fit (tau={tau_fit:.3f}s)')

        ax1.set_xlim(0, max(t_end * 1.5, 4.0))
        ax1.set_ylim(bottom=1e-3)
        ax1.set_title(f"Semi-Log Scale\nTau: {tau_fit:.4f}s ({tau_fit*1000:.2f} ms)")
        ax1.set_xlabel("Lag (s)")
        ax1.set_ylabel("Autocorrelation (log)")
        ax1.legend()
        ax1.grid(True, which="both", alpha=0.3)

        # --- RIGHT PANEL: Linear Scale ---
        ax2 = axes[1]
        # Data
        ax2.plot(lags_s, acf, 'o-', color='navy', markersize=3, label='ACF (Data)', alpha=0.4)

        # Window Limits
        ax2.axvline(t_start, color='green', linestyle='--', label='Start')
        ax2.axvline(t_end, color='red', linestyle='--', label='End')

        # Fit Curve
        if not np.isnan(tau_fit):
            # Recalculate curve for linear plot (same data points)
            y_fit_curve = mono_exp(x_fit, *popt)
            ax2.plot(x_fit, y_fit_curve, 'r-', linewidth=2.5, label=f'Exp Fit')

        ax2.set_xlim(0, max(t_end * 1.5, 4.0))
        # ax2.set_ylim(0, 1.1)
        ax2.set_title(f"Linear Scale\nTau: {tau_fit:.4f}s")
        ax2.set_xlabel("Lag (s)")
        ax2.set_ylabel("Autocorrelation (linear)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def compute_cumsum_correlation_metrics(true_spikes_train, reconstructed_spikes):
    """
    Pearson and Spearman between cumsum(true) and cumsum(reconstructed),
    both before ('raw') and after ('calib') dividing reconstructed by the OLS slope factor.

    Returns dict with keys:
        pearson_cumsum_raw, pearson_cumsum_calib,
        spearman_cumsum_raw, spearman_cumsum_calib
    """
    true_spikes_train    = np.asarray(true_spikes_train, dtype=float)
    reconstructed_spikes = np.asarray(reconstructed_spikes, dtype=float)

    x     = np.cumsum(true_spikes_train)
    y_raw = np.cumsum(reconstructed_spikes)

    nan_result = {
        'pearson_cumsum_raw':   np.nan, 'pearson_cumsum_calib':  np.nan,
        'spearman_cumsum_raw':  np.nan, 'spearman_cumsum_calib': np.nan,
    }

    if len(x) < 2 or np.std(x) == 0 or np.std(y_raw) == 0:
        return nan_result

    # OLS slope forced through origin (same as in analyze_cumsum_calibration)
    model = LinearRegression(fit_intercept=False)
    model.fit(x.reshape(-1, 1), y_raw)
    slope = model.coef_[0]

    if np.isnan(slope) or slope == 0:
        y_calib = y_raw
    else:
        y_calib = np.cumsum(reconstructed_spikes / slope)

    try:
        p_raw,   _ = pearsonr(x, y_raw)
        p_calib, _ = pearsonr(x, y_calib)
        s_raw,   _ = spearmanr(x, y_raw)
        s_calib, _ = spearmanr(x, y_calib)
    except Exception:
        return nan_result

    return {
        'pearson_cumsum_raw':   float(p_raw),
        'pearson_cumsum_calib': float(p_calib),
        'spearman_cumsum_raw':  float(s_raw),
        'spearman_cumsum_calib': float(s_calib),
    }


def compute_signal_correlation_metrics(sig1, sig2):
    """
    Pearson and Spearman between two signals, both raw and after OLS amplitude
    scaling of sig2 to match sig1 (same scaling used for R²/chi²/NMSE).

    Note: both scale variants are mathematically identical for Pearson and Spearman
    because both metrics are invariant to linear transformations; they are included
    for consistency with the rest of the GOF reporting.

    Returns dict with keys:
        pearson_raw, pearson_scaled, spearman_raw, spearman_scaled
    """
    sig1 = np.asarray(sig1, dtype=float)
    sig2 = np.asarray(sig2, dtype=float)

    nan_result = {
        'pearson_raw': np.nan, 'pearson_scaled': np.nan,
        'spearman_raw': np.nan, 'spearman_scaled': np.nan,
    }

    if len(sig1) < 2 or np.std(sig1) == 0 or np.std(sig2) == 0:
        return nan_result

    # OLS scale: minimize ||sig1 - a*sig2||² → a = <sig2, sig1> / <sig2, sig2>
    denom = np.dot(sig2, sig2)
    a_ols = np.dot(sig2, sig1) / denom if denom != 0 else 1.0
    sig2_scaled = a_ols * sig2

    try:
        p_raw,    _ = pearsonr(sig1, sig2)
        p_scaled, _ = pearsonr(sig1, sig2_scaled)
        s_raw,    _ = spearmanr(sig1, sig2)
        s_scaled, _ = spearmanr(sig1, sig2_scaled)
    except Exception:
        return nan_result

    return {
        'pearson_raw':   float(p_raw),
        'pearson_scaled': float(p_scaled),
        'spearman_raw':  float(s_raw),
        'spearman_scaled': float(s_scaled),
    }

    return tau_fit, (A_fit, tau_fit, C_fit)
