"""
Deprecated / retired functions — kept for reference only.

None of these are used by the active analysis pipeline. Tau estimation is now
done via the derivative phase-space method (functions.derivative_method) and the
PSD Lorentzian fit performed inline in the notebooks using the module-level
`lorentzian` helper in functions.metrics.

This module was split out of functions/metrics.py during the repository cleanup
to keep metrics.py focused on the metrics actually in use. Importing this module
pulls in `nitime`, which is otherwise no longer a hard dependency.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.signal import correlate
import nitime.algorithms as tsa


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

    # NOTE: this return was orphaned in the original metrics.py (it had drifted
    # below a different function, so estimate_manual_tau_exp silently returned
    # None). Reattached here during the cleanup.
    return tau_fit, (A_fit, tau_fit, C_fit)
