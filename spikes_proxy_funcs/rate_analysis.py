
import numpy as np

def calculate_cv2(spike_times):
    """
    Calculates the Coefficient of Variation 2 (CV2) for a spike train.
    CV2 is a local measure of irregularity.
    
    Parameters:
    -----------
    spike_times : np.array
        Array of spike times (or indices).
        
    Returns:
    --------
    cv2 : float
        The mean CV2 value for the spike train. Returns np.nan if fewer than 3 spikes.
    """
    if len(spike_times) < 3:
        return np.nan
        
    isi = np.diff(spike_times)
    # Avoid division by zero
    denominator = isi[1:] + isi[:-1]
    valid_mask = denominator > 0
    
    if not np.any(valid_mask):
        return np.nan
        
    cv2_local = 2 * np.abs(isi[1:] - isi[:-1]) / denominator
    return np.mean(cv2_local[valid_mask])

def calculate_firing_rate_slope(spike_times, duration, dt=None):
    """
    Calculates the average firing rate as the slope of the cumulative spike count over time.
    
    Parameters:
    -----------
    spike_times : np.array
        Array of spike times.
    duration : float
        Total duration of the recording.
    dt : float, optional
        Time step (if spike_times are indices). If None, assumes spike_times are times in seconds.
        
    Returns:
    --------
    slope : float
        The slope of the cumulative spike count (approximate firing rate in Hz).
    """
    if len(spike_times) == 0:
        return 0.0
    
    if dt is not None:
        times = spike_times * dt
    else:
        times = spike_times
        
    # We fit a line to the cumulative count vs time
    # Points: (t_i, i+1) for each spike i
    y_vals = np.arange(1, len(times) + 1)
    
    # To be robust, we essentially want slope of N(t) vs t.
    if len(times) > 1:
        slope = np.polyfit(times, y_vals, 1)[0]
    else:
        slope = len(times) / duration if duration > 0 else 0
        
    return slope
