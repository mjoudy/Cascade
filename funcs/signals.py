import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, savgol_filter
from scipy.stats import pearsonr
from sklearn.linear_model import RANSACRegressor

def simulate_calcium(spikes, tau=100, dt=1, wup_time=1000,
                     noise_intra=0.01, noise_rec=1.0,
                     baseline=0.0, return_df_f=False,
                     seed=None, plot=False, neuron_idx=0, time_limit=None):
    """Simulates calcium fluorescence from a spike train for each neuron."""
    spikes_cropped = spikes[:, wup_time:]
    n_neurons, sim_dur = spikes_cropped.shape
    rng = np.random.default_rng(seed)

    intra_noise_mat = rng.normal(0, noise_intra, (n_neurons, sim_dur))
    a_coeff = np.exp(-dt / tau)
    b, a = [1.0], [1.0, -a_coeff]
    
    calcium = lfilter(b, a, spikes_cropped + intra_noise_mat, axis=1)
    recording_noise = rng.normal(0, noise_rec, (n_neurons, sim_dur))
    f_signal = baseline + calcium + recording_noise

    if return_df_f and baseline != 0:
        output = (f_signal - baseline) / baseline
        label = "ΔF/F₀"
    else:
        output = f_signal
        label = "Fluorescence (A.U.)"

    if plot:
        _plot_trace(output[neuron_idx], spikes_cropped[neuron_idx], dt, label, time_limit)

    return output, spikes_cropped

def reconstruct_spikes(calcium_data, tau=100, window_len=31, poly_order=3):
    """Reconstructs spikes using Savitzky-Golay filtering (neurons x time)."""
    smooth_calcium = savgol_filter(calcium_data, window_length=window_len, 
                                   polyorder=poly_order, deriv=0, axis=1)
    smooth_derivative = savgol_filter(calcium_data, window_length=window_len, 
                                      polyorder=poly_order, deriv=1, axis=1)
    return smooth_derivative + (1/tau) * smooth_calcium

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
            x_full = sig_full.reshape(-1, 1)
            ransac.fit(x_full, der_full)
            outlier_mask = ~ransac.inlier_mask_
        
        slope = ransac.estimator_.coef_[0]
        tau_val = -1.0 / slope if slope < 0 else np.nan
        estimated_taus[i] = tau_val
        
        if plot and i == neuron_idx:
            _plot_phase_space(sig_full, der_full, outlier_mask, ransac, tau_val, i, method)
            
    return estimated_taus

def cumsum_analysis(true_spikes, recon_spikes):
    """Performs individual cumulative sum slope analysis for each neuron."""
    if true_spikes.ndim == 1:
        true_spikes, recon_spikes = true_spikes[np.newaxis, :], recon_spikes[np.newaxis, :]
    n_neurons = true_spikes.shape[0]
    slopes = np.zeros(n_neurons)
    for i in range(n_neurons):
        if np.sum(true_spikes[i, :]) == 0:
            slopes[i] = np.nan; continue
        cum_true, cum_recon = np.cumsum(true_spikes[i, :]), np.cumsum(recon_spikes[i, :])
        slopes[i] = np.polyfit(cum_true, cum_recon, 1)[0]
    return slopes

def analyze_binned_performance(true_spikes, recon_spikes, bin_size=100, neuron_idx=0, plot=True, time_limit=(0, 5000)):
    """Bins spikes and calculates correlation for each neuron with visual delay fix."""
    n_neurons, n_time = true_spikes.shape
    n_bins = n_time // bin_size
    b_true = true_spikes[:, :n_bins*bin_size].reshape(n_neurons, n_bins, bin_size).sum(axis=2)
    b_recon = recon_spikes[:, :n_bins*bin_size].reshape(n_neurons, n_bins, bin_size).sum(axis=2)
    
    correlations = np.zeros(n_neurons)
    for i in range(n_neurons):
        if np.std(b_true[i, :]) == 0 or np.std(b_recon[i, :]) == 0:
            correlations[i] = np.nan
        else:
            correlations[i], _ = pearsonr(b_true[i, :], b_recon[i, :])
            
    if plot:
        t_bins = np.arange(n_bins) * bin_size
        plt.figure(figsize=(15, 5))
        plt.bar(t_bins, b_true[neuron_idx], width=bin_size, alpha=0.3, color='gray', label='True', align='edge')
        # Corrected visual phase shift using half-bin offset and mid alignment
        plt.step(t_bins + (bin_size/2), b_recon[neuron_idx], where='mid', color='#e67e22', label='Recon')
        plt.title(f"Binned | Neuron {neuron_idx} | r = {correlations[neuron_idx]:.3f}")
        plt.xlim(time_limit); plt.legend(); plt.show()
    return correlations

def calculate_cv2(spike_matrix):
    """Calculates CV2 for each neuron individually."""
    n_neurons = spike_matrix.shape[0]
    cv2_values = np.zeros(n_neurons)
    for i in range(n_neurons):
        spike_times = np.where(spike_matrix[i, :] > 0.5)[0]
        if len(spike_times) < 3:
            cv2_values[i] = np.nan; continue
        isi = np.diff(spike_times)
        cv2_values[i] = np.mean(2 * np.abs(isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1]))
    return cv2_values

def _plot_phase_space(sig_full, der_full, outlier_mask, model, tau_val, idx, method):
    """Plots phase space highlighting RANSAC inliers (blue) vs outliers (red)."""
    plt.figure(figsize=(7, 7))
    plt.scatter(sig_full[~outlier_mask], der_full[~outlier_mask], s=2, color='#3498db', alpha=0.3, label='Bulk (Inliers)')
    plt.scatter(sig_full[outlier_mask], der_full[outlier_mask], s=2, color='#e74c3c', alpha=0.2, label='Spikes (Outliers)')
    x_range = np.array([np.min(sig_full), np.max(sig_full)]).reshape(-1, 1)
    plt.plot(x_range, model.predict(x_range), color='black', linewidth=2, linestyle='--', label=f'τ fit ≈ {tau_val:.1f}')
    plt.title(f"Phase Space ({method}) - Neuron {idx}"); plt.xlabel("C"); plt.ylabel("dC/dt"); plt.legend(); plt.grid(alpha=0.1); plt.show()

def _plot_trace(trace, spike_vec, dt, ylabel, time_limit=None):
    time = np.arange(len(trace)) * dt
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, trace, color='#2c3e50', linewidth=1, alpha=0.8)
    data_min, data_range = np.min(trace), np.ptp(trace)
    spike_pos = data_min - (data_range * 0.05); spike_times = np.where(spike_vec > 0.5)[0] * dt
    ax.vlines(spike_times, ymin=spike_pos, ymax=spike_pos + (data_range * 0.1), color='red', alpha=0.5)
    if time_limit: ax.set_xlim(time_limit)
    ax.set_ylabel(ylabel); ax.set_xlabel("Time (ms)"); plt.tight_layout(); plt.show()