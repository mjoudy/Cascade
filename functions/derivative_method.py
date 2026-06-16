import numpy as np
from scipy.signal import savgol_filter

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

def estimate_tau(u_sig, u_spk, window_len=51, poly_order=3, cut_win=10):
    """
    Estimates Tau for a single neuron's signal and spike train.
    Returns the tau value and the data used for fitting.
    """
    sig_smooth = savgol_filter(u_sig, window_length=window_len, polyorder=poly_order, deriv=0)
    der_smooth = savgol_filter(u_sig, window_length=window_len, polyorder=poly_order, deriv=1)
    sig_fit, der_fit, _ = cut_spikes(u_spk, sig_smooth, der_smooth, win_len=cut_win)
    
    try:
        if len(sig_fit) > 2:
            slope_fit, _ = np.polyfit(sig_fit, der_fit, 1)
            tau_f = -1.0 / slope_fit if slope_fit < 0 else np.nan
        else:
            tau_f = np.nan
    except (np.linalg.LinAlgError, ValueError, TypeError):
        tau_f = np.nan
        
    return tau_f, (sig_fit, der_fit)
