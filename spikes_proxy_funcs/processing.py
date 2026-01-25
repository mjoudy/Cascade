import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def upsample(times, dff, spikes, new_rate, intp_method='cubic', do_plot=False):
    """
    ORIGINAL VERSION (V1)
    Upsamples fluorescence data and aligns spikes to a high-frequency grid.
    Note: This version introduces a shift via zero-padding and scales spikes to indices.
    """
    time_start = times[0]
    shifted_time = times - time_start
    
    # Ensure shifted_time does not have duplicates (required for interp1d)
    unique_indices = np.unique(shifted_time, return_index=True)[1]
    # Sort indices to preserve order
    unique_indices = np.sort(unique_indices)
    
    shifted_time_unique = shifted_time[unique_indices]
    dff_unique = dff[unique_indices]
    
    intpld_signal_func = interp1d(shifted_time_unique, dff_unique, kind=intp_method)
    evenly_spaced_time = np.linspace(shifted_time[0], shifted_time[-1], int((shifted_time[-1])*new_rate))
    upsampled_signal = intpld_signal_func(evenly_spaced_time)
    time_shift = np.zeros(int(times[0]*new_rate))
    upsampled_signal = np.concatenate((time_shift, upsampled_signal), axis=0)
    upsampled_spikes = new_rate*spikes
    
    if do_plot:
        plt.figure(figsize=(20,5))
        plt.plot(upsampled_signal)
        # Handle cases where dff might be short due to cutting or other preprocessing
        if len(dff) > 4:
            max_dff = np.max(dff[4:])
            min_dff = np.min(dff[4:])
        else:
            max_dff = np.max(dff)
            min_dff = np.min(dff)
            
        plt.eventplot(upsampled_spikes, lineoffsets=min_dff - max_dff/20, linelengths=max_dff/20, color='k')
        plt.title(f"Upsampled Signal (V1 - Rate: {new_rate} Hz)")
        plt.show()

    return upsampled_signal, upsampled_spikes


def upsample_v2(times, dff, spikes, new_rate, intp_method='cubic', do_plot=False):
    # --- 1. Fix the Duplicate Timestamp Error (The ValueError fix) ---
    _, unique_indices = np.unique(times, return_index=True)
    unique_indices = np.sort(unique_indices)
    times_clean = times[unique_indices]
    dff_clean = dff[unique_indices]

    # --- 2. V2 Logic: Absolute Time Grid ---
    t_start, t_end = times_clean[0], times_clean[-1]
    num_samples = int((t_end - t_start) * new_rate)
    evenly_spaced_time = np.linspace(t_start, t_end, num_samples)
    
    intpld_signal_func = interp1d(times_clean, dff_clean, kind=intp_method, fill_value="extrapolate")
    upsampled_signal = intpld_signal_func(evenly_spaced_time)
    
    # Keep spikes in seconds
    upsampled_spikes = spikes[(spikes >= t_start) & (spikes <= t_end)]

    # --- 3. RESTORED V1 PLOTTING STYLE ---
    if do_plot:
        plt.figure(figsize=(20, 5))
        
        # Plotting without 'evenly_spaced_time' to match V1's index-based x-axis
        plt.plot(upsampled_signal) 
        
        # We must convert seconds back to indices JUST for the plot
        # to match the index-based x-axis of the signal plot
        spike_indices = (upsampled_spikes - t_start) * new_rate
        
        # Logic for offsets taken directly from your V1 code
        if len(dff_clean) > 4:
            max_dff, min_dff = np.max(dff_clean[4:]), np.min(dff_clean[4:])
        else:
            max_dff, min_dff = np.max(dff_clean), np.min(dff_clean)
            
        plt.eventplot(spike_indices, 
                      lineoffsets=min_dff - max_dff/20, 
                      linelengths=max_dff/20, 
                      color='k')
        
        plt.title(f"Upsampled Signal (V2 - Logic fixed, Plotting restored)")
        plt.show()

    return upsampled_signal, upsampled_spikes, evenly_spaced_time