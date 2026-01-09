
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def upsample(times, dff, spikes, new_rate, intp_method='cubic', do_plot=False):
    """
    Upsamples fluorescence data and aligns spikes to a high-frequency grid.
    
    Parameters:
    -----------
    times : np.array
        Original time vector.
    dff : np.array
        Fluorescence trace (dF/F).
    spikes : np.array
        Spike train.
    new_rate : float
        New sampling rate (Hz).
    intp_method : str, optional
        Interpolation method for dff (default: 'cubic').
    do_plot : bool, optional
        If True, plots the upsampled signal and spikes.
        
    Returns:
    --------
    upsampled_signal : np.array
        The interpolated fluorescence trace.
    upsampled_spikes : np.array
        The spikes aligned to the new time grid.
    """
    time_start = times[0]
    shifted_time = times - time_start
    
    # Ensure shifted_time does not have duplicates (required for interp1d)
    unique_indices = np.unique(shifted_time, return_index=True)[1]
    # Sort indices to preserve order (though unique usually does returns sorted values for sorted input)
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
        plt.title(f"Upsampled Signal (Rate: {new_rate} Hz)")
        plt.show()

    return upsampled_signal, upsampled_spikes
