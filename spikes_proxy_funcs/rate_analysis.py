import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor
from scipy.signal import lfilter
from scipy.stats import pearsonr

def cut_spikes(spikes, signal, deriv, win_len=5):
    """Removes indices around spike events to isolate the 'decay-only' bulk."""
    bool_check = all(element == 0 or element == 1 for element in spikes)
    if bool_check:
        event_spikes = np.where(spikes > 0.5)[0]
    else:
        event_spikes = spikes.astype(int)

    remove_index = []
    for i in event_spikes:
        remove_index.append(np.arange(i - win_len, i + win_len))
    
    if len(remove_index) == 0:
        return signal, deriv, np.zeros(len(signal), dtype=bool)

    remove_index = np.unique(np.concatenate(remove_index))
    remove_index = remove_index[(remove_index >= 0) & (remove_index < len(signal))]
    removed_mask = np.zeros(len(signal), dtype=bool)
    removed_mask[remove_index] = True

    return np.delete(signal, remove_index), np.delete(deriv, remove_index), removed_mask

def _plot_phase_space(sig_full, der_full, outlier_mask, model, tau_val, idx, method):
    """Plots phase space highlighting RANSAC inliers (blue) vs outliers (red)."""
    plt.figure(figsize=(7, 7))
    plt.scatter(sig_full[~outlier_mask], der_full[~outlier_mask], s=2, color='#3498db', alpha=0.3, label='Bulk (Inliers)')
    plt.scatter(sig_full[outlier_mask], der_full[outlier_mask], s=2, color='#e74c3c', alpha=0.2, label='Spikes (Outliers)')
    x_range = np.array([np.min(sig_full), np.max(sig_full)]).reshape(-1, 1)
    plt.plot(x_range, model.predict(x_range), color='black', linewidth=2, linestyle='--', label=f'τ fit ≈ {tau_val:.1f}')
    plt.title(f"Phase Space ({method}) - Neuron {idx}")
    plt.xlabel("C")
    plt.ylabel("dC/dt")
    plt.legend()
    plt.grid(alpha=0.1)
    plt.show()

def estimate_tau(calcium_data, true_spikes=None, neuron_indices='all', 
                 method='robust', window_len=51, poly_order=3, cut_win=10, 
                 plot=True, neuron_idx=0):
    """
    Estimates tau from Phase Space using RANSAC for robust estimation.
    - 'cut': Uses true_spikes to remove events manually.
    - 'robust': Blind estimation using RANSAC to identify outliers.
    """
    C_mat = savgol_filter(calcium_data, window_length=window_len, polyorder=poly_order, deriv=0, axis=1)
    dC_mat = savgol_filter(calcium_data, window_length=window_len, polyorder=poly_order, deriv=1, axis=1)
    
    if neuron_indices == 'all':
        indices = np.arange(calcium_data.shape[0])
    elif isinstance(neuron_indices, (int, np.integer)):
        indices = [neuron_indices]
    else:
        indices = neuron_indices
        
    estimated_taus = {}
    for i in indices:
        sig_full, der_full = C_mat[i], dC_mat[i]
        ransac = RANSACRegressor()
        
        if method == 'cut' and true_spikes is not None:
            sig_fit, der_fit, outlier_mask = cut_spikes(true_spikes[i], sig_full, der_full, win_len=cut_win)
            x_fit = sig_fit.reshape(-1, 1)
            ransac.fit(x_fit, der_fit)
        else:
            # Robust method uses RANSAC on full data
            x_full = sig_full.reshape(-1, 1)
            ransac.fit(x_full, der_full)
            outlier_mask = ~ransac.inlier_mask_
        
        slope = ransac.estimator_.coef_[0]
        # avoidance of div by zero could be added, but following original logic:
        tau_val = -1.0 / slope if slope < 0 else np.nan
        estimated_taus[i] = tau_val
        
        if plot and i == neuron_idx:
            _plot_phase_space(sig_full, der_full, outlier_mask, ransac, tau_val, i, method)
            
    return estimated_taus

def calculate_cv2(spike_matrix):
    """Calculates CV2 for each neuron individually, handling both binary masks and index lists."""
    # Ensure input is a numpy array
    spike_matrix = np.atleast_1d(spike_matrix)
    
    # NEW: Check if the array is empty to prevent ValueError in np.max()
    if spike_matrix.size == 0:
        return np.nan
    
    # If the input is a 1D list of spike indices/times (values > 1)
    if spike_matrix.ndim == 1 and np.max(spike_matrix) > 1.0:
        spike_times = spike_matrix
    else:
        # If it's a binary matrix (0s and 1s), identify the spike positions
        if spike_matrix.ndim == 1:
            spike_times = np.where(spike_matrix > 0.5)[0]
        else:
            # Handle 2D matrices
            results = []
            for i in range(spike_matrix.shape[0]):
                st = np.where(spike_matrix[i, :] > 0.5)[0]
                if len(st) < 3:
                    results.append(np.nan)
                else:
                    isi = np.diff(st)
                    results.append(np.mean(2 * np.abs(isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1])))
            return np.array(results)

    # Calculation for a single neuron (index list or 1D mask)
    if len(spike_times) < 3:
        return np.nan
        
    isi = np.diff(spike_times)
    return np.mean(2 * np.abs(isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1]))


def reconstruct_calcium(spikes, tau_frames, signal_length):
    """
    Reconstructs calcium signal using the AR(1) model.
    Handles 'spikes' as either a binary vector or a list of indices.
    """
    if np.isnan(tau_frames) or tau_frames <= 0:
        return np.zeros(signal_length)
    
    # 1. Convert to binary vector if input is indices
    if len(spikes) != signal_length:
        binary_spikes = np.zeros(signal_length)
        # Ensure indices are integers and within bounds
        idx = np.round(spikes).astype(int)
        idx = idx[(idx >= 0) & (idx < signal_length)]
        binary_spikes[idx] = 1.0
    else:
        binary_spikes = spikes

    # 2. Apply AR(1) Filter (a = exp(-dt/tau), dt=1 frame)
    a_coeff = np.exp(-1.0 / tau_frames)
    b, a = [1.0], [1.0, -a_coeff]
    
    return lfilter(b, a, binary_spikes)

def calculate_reconstruction_corr(original_signal, spikes, tau_frames):
    """Calculates Pearson correlation between original and reconstructed calcium."""
    sig_len = len(original_signal)
    recon = reconstruct_calcium(spikes, tau_frames, sig_len)
    
    if np.all(recon == 0) or np.any(np.isnan(recon)):
        return np.nan
        
    # Pearson correlation is invariant to scaling/offset
    corr, _ = pearsonr(original_signal, recon)
    return corr

def reconstruct_spikes(signal, tau_frames):
    """Simple deconvolution: S_t = C_t - exp(-1/tau) * C_{t-1}"""
    if np.isnan(tau_frames) or tau_frames <= 0:
        return np.zeros_like(signal)
    a_coeff = np.exp(-1.0 / tau_frames)
    recon_spks = np.zeros_like(signal)
    # The innovation at time t is the current signal minus the decayed previous state
    recon_spks[1:] = signal[1:] - a_coeff * signal[:-1]
    # Remove negative values (rectification) common in noise
    recon_spks = np.maximum(recon_spks, 0)
    return recon_spks

def calculate_spike_correlation(signal, spikes, tau_frames):
    """Pearson correlation between reconstructed spike density and ground truth."""
    sig_len = len(signal)
    recon_spks = reconstruct_spikes(signal, tau_frames)
    
    # Convert spikes to binary vector if they are indices
    if len(spikes) != sig_len:
        binary_spikes = np.zeros(sig_len)
        idx = np.round(spikes).astype(int)
        idx = idx[(idx >= 0) & (idx < sig_len)]
        binary_spikes[idx] = 1.0
    else:
        binary_spikes = spikes
        
    if np.all(recon_spks == 0): return np.nan
    corr, _ = pearsonr(binary_spikes, recon_spks)
    return corr

def calculate_cumsum_slope(spikes, sampling_rate):
    """Calculates the slope of the cumulative sum of spikes (Events/sec)."""
    if len(spikes) == 0: return 0
    y = np.cumsum(spikes)
    x = np.arange(len(y)) / sampling_rate
    slope, _ = np.polyfit(x, y, 1)
    return slope

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def get_reconstruction_metrics(res, window_len=51, poly_order=3, cut_win=10, new_rate=100):
    """Performs calculations and returns a data dictionary + signals."""
    u_sig = res['signal'].copy()
    u_spk = res['spikes']
    
    # --- Standardize Signal (Handle NaNs) ---
    if np.all(np.isnan(u_sig)):
        return None # Entire signal is NaN, cannot process
        
    if np.any(np.isnan(u_sig)):
        mask = np.isnan(u_sig)
        u_sig[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), u_sig[~mask])

    # --- Tau Estimation ---
    sig_smooth = savgol_filter(u_sig, window_length=window_len, polyorder=poly_order, deriv=0)
    der_smooth = savgol_filter(u_sig, window_length=window_len, polyorder=poly_order, deriv=1)
    sig_fit, der_fit, _ = cut_spikes(u_spk, sig_smooth, der_smooth, win_len=cut_win)
    
    try:
        if len(sig_fit) > 2:
            slope_fit, _ = np.polyfit(sig_fit, der_fit, 1)
            tau_f = -1.0 / slope_fit if slope_fit < 0 else np.nan
        else:
            tau_f = np.nan
    except: 
        tau_f = np.nan
        
    # --- Reconstruction ---
    # Using existing functions from rate_analysis.py
    c_corr = calculate_reconstruction_corr(u_sig, u_spk, tau_f)
    recon_calcium = reconstruct_calcium(u_spk, tau_f, len(u_sig))
    s_corr = calculate_spike_correlation(u_sig, u_spk, tau_f)
    recon_spks = reconstruct_spikes(u_sig, tau_f)
    cs_slope = calculate_cumsum_slope(recon_spks, new_rate)
    
    # Calculate spike count correctly
    n_spikes = len(u_spk) if len(u_spk) != len(u_sig) else np.sum(u_spk > 0.5)

    stats = {
        'Dataset': res['dataset'], 
        'Neuron': res['neuron_idx'], 
        'Spikes': int(n_spikes), # This key must exist for sorting
        'Tau_s': round(tau_f/new_rate, 3) if not np.isnan(tau_f) else "N/A",
        'Ca_Corr': round(c_corr, 3) if not np.isnan(c_corr) else "N/A",
        'Spike_Corr': round(s_corr, 3) if not np.isnan(s_corr) else "N/A",
        'Cumsum_Slope': round(cs_slope, 3)
    }
    
    return stats, (u_sig, recon_calcium, recon_spks, u_spk, tau_f)

def plot_reconstruction(stats, signals, new_rate=100, save_path=None, show=True):
    """Independent plotting function."""
    u_sig, recon_calcium, recon_spks, u_spk, tau_f = signals
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 2, 1]})
    time_axis = np.arange(len(u_sig)) / new_rate
    
    # Panel 1: Calcium Signal
    ax1.plot(time_axis, u_sig, color='gray', alpha=0.4, label='Raw Fluorescence')
    if recon_calcium is not None and not np.all(recon_calcium == 0):
        # Normalize and scale for visualization
        rc_norm = (recon_calcium - np.min(recon_calcium)) / (np.max(recon_calcium) - np.min(recon_calcium) + 1e-9)
        rc_scaled = rc_norm * (np.max(u_sig) - np.min(u_sig)) + np.min(u_sig)
        ax1.plot(time_axis, rc_scaled, color='crimson', label=f"Recon (r={stats['Ca_Corr']})")
    ax1.set_title(f"DS: {stats['Dataset']} | N: {stats['Neuron']} | Tau: {stats['Tau_s']}s")
    ax1.legend(loc='upper right')

    # Panel 2: Deconvolution
    ax2.plot(time_axis, recon_spks, color='darkorange', label=f"Deconvolved (r={stats['Spike_Corr']})")
    ax2.legend(loc='upper right')

    # Panel 3: Ground Truth
    spike_times = time_axis[u_spk > 0.5] if len(u_spk) == len(u_sig) else np.array(u_spk) / new_rate
    ax3.eventplot(spike_times, orientation='horizontal', colors='black', label='Ground Truth')
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    if show: plt.show()
    else: plt.close()