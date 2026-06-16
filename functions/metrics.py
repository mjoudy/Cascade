import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, zscore, linregress, wilcoxon
from scipy.optimize import minimize_scalar, curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.signal import correlate

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


def compute_binned_rate_correlation_metrics(true_spikes_train, reconstructed_spikes,
                                            bin_ms=500.0, fs=200.0):
    """
    Pearson and Spearman between binned spike counts (no cumsum).
    Bin both traces into bin_ms-wide windows and correlate the per-bin totals.
    This is genuinely different from cumsum metrics: it measures whether the
    reconstruction fires in the right time bins, not just the running total.
    bin_ms=500 ms was selected from a bin-width sweep in active_neurons.ipynb.
    """
    true_spikes_train    = np.asarray(true_spikes_train, dtype=float)
    reconstructed_spikes = np.asarray(reconstructed_spikes, dtype=float)

    nan_result = {
        'pearson_binned_rate':  np.nan,
        'spearman_binned_rate': np.nan,
    }

    bin_frames = max(1, int(round(bin_ms * fs / 1000.0)))
    T      = len(true_spikes_train)
    n_bins = T // bin_frames
    if n_bins < 3:
        return nan_result

    sl          = n_bins * bin_frames
    true_binned = true_spikes_train[:sl].reshape(n_bins, bin_frames).sum(axis=1)
    rec_clipped = np.clip(reconstructed_spikes[:sl], 0, None)
    rec_binned  = rec_clipped.reshape(n_bins, bin_frames).sum(axis=1)

    if np.std(true_binned) == 0 or np.std(rec_binned) == 0:
        return nan_result

    try:
        p, _ = pearsonr(true_binned, rec_binned)
        s, _ = spearmanr(true_binned, rec_binned)
    except Exception:
        return nan_result

    return {
        'pearson_binned_rate':  float(p),
        'spearman_binned_rate': float(s),
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


# ── GOF scaling & metrics ─────────────────────────────────────────────────────

def ols_scale(original, sim):
    """OLS scale factor: minimises ||original - a*sim||²."""
    return np.dot(sim, original) / np.dot(sim, sim)


def lad_scale(original, sim):
    """LAD scale factor: minimises sum|original - a*sim|."""
    result = minimize_scalar(
        lambda a: np.sum(np.abs(original - a * sim)),
        bounds=(0.01, 100), method='bounded'
    )
    return result.x


def compute_gof(original, sim):
    """Return dict with mse, rmse, r_squared for a (original, sim) pair."""
    mse    = np.mean((original - sim) ** 2)
    ss_res = np.sum((original - sim) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2)
    return {
        'mse':       mse,
        'rmse':      np.sqrt(mse),
        'r_squared': 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    }


def add_chi2_nmse(sub_dict, y_true, y_pred):
    """Add chi_squared and nmse keys to an existing GOF sub-dict in-place."""
    eps = 1e-10
    sub_dict['chi_squared'] = float(np.sum((y_true - y_pred) ** 2 / (np.abs(y_pred) + eps)))
    mse      = np.mean((y_true - y_pred) ** 2)
    var_true = np.var(y_true, ddof=1)
    sub_dict['nmse'] = float(mse / var_true) if var_true > 0 else np.nan


# ── PSD ───────────────────────────────────────────────────────────────────────

def lorentzian(f, A, tau, white_noise):
    """Lorentzian (1/f²) power spectrum model for tau estimation."""
    return A / (1 + (2 * np.pi * f * tau) ** 2) + white_noise


# ── Spike / neuron statistics ─────────────────────────────────────────────────

def neuron_spike_metrics(n):
    """Return (firing_rate_hz, mean_isi_s, cv2, burst_index) for one neuron dict."""
    spikes   = np.sort(n['original_neuron']['spikes'])
    time     = n['evenly_spaced_time']
    duration = time[-1] - time[0]
    n_spikes = len(spikes)

    firing_rate = n_spikes / duration if duration > 0 else np.nan

    if n_spikes > 1:
        isis        = np.diff(spikes)
        mean_isi    = np.mean(isis)
        cv2         = np.mean(2 * np.abs(np.diff(isis)) / (isis[:-1] + isis[1:])) if len(isis) > 1 else np.nan
        burst_index = np.mean(isis < mean_isi / 2)
    else:
        mean_isi = cv2 = burst_index = np.nan

    return firing_rate, mean_isi, cv2, burst_index


# ── Tau-sweep metric helpers ──────────────────────────────────────────────────

def calcium_metrics(original, sim):
    """OLS-scale sim onto original; return (nmse, scale_factor)."""
    denom = np.dot(sim, sim)
    if denom == 0:
        return np.nan, np.nan
    a        = np.dot(sim, original) / denom
    var_true = np.var(original, ddof=1)
    nmse     = float(np.mean((original - a * sim) ** 2) / var_true) if var_true > 0 else np.nan
    return nmse, float(a)


def cumsum_metrics(x_true, rec):
    """OLS-scale cumsum(rec) onto x_true; return (nmse, slope)."""
    y_rec = np.cumsum(rec).astype(float)
    denom = np.dot(x_true, x_true)
    if denom == 0:
        return np.nan, np.nan
    slope = np.dot(x_true, y_rec) / denom
    if slope == 0:
        return np.nan, np.nan
    y_pred = y_rec / slope
    var_x  = np.var(x_true, ddof=1)
    nmse   = float(np.mean((x_true - y_pred) ** 2) / var_x) if var_x > 0 else np.nan
    return nmse, float(slope)


def minmax_norm(arr):
    """Normalise array to [0, 1]; returns zeros if range is zero."""
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)


# ── Correlation helpers for tau-sweep figures ─────────────────────────────────

def safe_pearson(a, b):
    """Pearson r; returns nan if too few samples or zero variance."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    try:
        return float(pearsonr(a, b)[0])
    except Exception:
        return np.nan


def safe_spearman(a, b):
    """Spearman r; returns nan if too few samples or zero variance."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    try:
        return float(spearmanr(a, b)[0])
    except Exception:
        return np.nan


def nonsig_mask(mat, alpha=0.05):
    """
    Identify tau values not significantly worse than the best tau.

    Returns (mask, best_col) where mask[t]=True means tau t is NOT
    significantly worse than the best tau (paired Wilcoxon, Bonferroni corrected).
    """
    n_tau   = mat.shape[1]
    medians = np.nanmedian(mat, axis=0)
    best    = int(np.nanargmax(medians))
    mask    = np.ones(n_tau, dtype=bool)
    n_comp  = n_tau - 1

    for t in range(n_tau):
        if t == best:
            continue
        a, b  = mat[:, best], mat[:, t]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 5 or np.all(a[valid] == b[valid]):
            continue
        try:
            _, p    = wilcoxon(a[valid], b[valid], alternative='greater')
            mask[t] = (p * n_comp) > alpha
        except Exception:
            pass

    return mask, best
