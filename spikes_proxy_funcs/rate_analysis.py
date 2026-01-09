import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor

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
    """Calculates CV2 for each neuron individually."""
    if spike_matrix.ndim == 1:
        spike_matrix = spike_matrix.reshape(1, -1)
    
    n_neurons = spike_matrix.shape[0]
    cv2_values = np.zeros(n_neurons)
    
    for i in range(n_neurons):
        # Assuming binary spike train or detecting peaks > 0.5
        spike_times = np.where(spike_matrix[i, :] > 0.5)[0]
        
        if len(spike_times) < 3:
            # Need at least 3 spikes (2 intervals) to calculate CV2 of diffs
            cv2_values[i] = np.nan
            continue
            
        isi = np.diff(spike_times)
        # CV2 formula: mean(2 * |ISI_i+1 - ISI_i| / (ISI_i+1 + ISI_i))
        cv2_values[i] = np.mean(2 * np.abs(isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1]))
        
    return cv2_values
