import numpy as np
from scipy.signal import savgol_filter, lfilter

def reconstruct_spikes(calcium_data, tau=100, window_len=31, poly_order=3):
    """
    Reconstructs spikes using Savitzky-Golay filtering.
    
    Args:
        calcium_data (np.ndarray): 1D or 2D array of calcium signals.
                                   If 2D, shape should be (neurons, time).
        tau (float or np.ndarray): Decay time constant(s).
        window_len (int): Window length for Savitzky-Golay filter.
        poly_order (int): Polynomial order for Savitzky-Golay filter.
    
    Returns:
        np.ndarray: Reconstructed spikes (derivative + signal/tau).
    """
    
    # Check dimensionality to determine axis
    if np.ndim(calcium_data) == 1:
        axis = 0
    else:
        axis = 1 # Assumes (neurons, time) structure for 2D
        
        # Ensure tau is broadcastable if it's an array
        if isinstance(tau, np.ndarray) and tau.ndim == 1:
            tau = tau[:, np.newaxis]

    smooth_calcium = savgol_filter(calcium_data, window_length=window_len, 
                                   polyorder=poly_order, deriv=0, axis=axis)
    smooth_derivative = savgol_filter(calcium_data, window_length=window_len, 
                                      polyorder=poly_order, deriv=1, axis=axis)
    
    return smooth_derivative + (1/tau) * smooth_calcium

def simulate_calcium_from_true_spikes(true_spikes_sec, time_grid, tau_frames, sampling_rate, 
                                      noise_std=0.0):
    """
    Simulates calcium fluorescence from discrete true spike times.
    
    Args:
        true_spikes_sec (np.ndarray): Spike timestamps in seconds.
        time_grid (np.ndarray): Array of timestamps for the output signal (seconds).
        tau_frames (float): Decay time constant in FRAMES.
        sampling_rate (float): Sampling rate in Hz.
        noise_std (float): Standard deviation of Gaussian noise to add.
        
    Returns:
        np.ndarray: Simulated calcium signal on the time_grid.
    """
    # 1. Create binary spike train on the time grid
    spike_train = np.zeros_like(time_grid)
    
    # Simple binning: find closest index for each spike
    # Assuming time_grid is evenly spaced and sorted
    t_start = time_grid[0]
    dt = 1.0 / sampling_rate
    
    spike_indices = ((true_spikes_sec - t_start) * sampling_rate).astype(int)
    # Filter indices within bounds
    spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(time_grid))]
    
    # Assign spikes directly (amplitude 1 per event)
    # Note: If multiple spikes fall in one bin, we might want to add them.
    # Here we just set to 1 or add 1
    np.add.at(spike_train, spike_indices, 1.0)
    
    # 2. Setup exponential filter
    # a_coeff = exp(-1/tau) since dt=1 frame in discrete time
    decay_factor = np.exp(-1.0 / tau_frames)
    b = [1.0]
    a = [1.0, -decay_factor]
    
    # 3. Convolve
    # Default axis=-1, works for 1D
    calcium = lfilter(b, a, spike_train)
    
    # 4. Add Noise if requested
    if noise_std > 0:
        rng = np.random.default_rng()
        noise = rng.normal(0, noise_std, len(calcium))
        calcium = calcium + noise
        
    return calcium

def simulate_calcium_from_reconstructed_spikes(rec_spikes, tau_frames, noise_std=0.0):
    """
    Simulates (or recovers) calcium fluorescence from reconstructed spike trace.
    
    Args:
        rec_spikes (np.ndarray): Reconstructed continuous spike signal.
        tau_frames (float): Decay time constant in FRAMES.
        noise_std (float): Standard deviation of Gaussian noise to add.
        
    Returns:
        np.ndarray: Simulated calcium signal.
    """
    # 1. Setup exponential filter
    # a_coeff = exp(-1/tau)
    decay_factor = np.exp(-1.0 / tau_frames)
    b = [1.0]
    a = [1.0, -decay_factor]
    
    # 2. Convolve
    calcium = lfilter(b, a, rec_spikes)
    
    # 3. Add Noise if requested
    if noise_std > 0:
        rng = np.random.default_rng()
        noise = rng.normal(0, noise_std, len(calcium))
        calcium = calcium + noise
        
    return calcium

def get_spikes_train(upsampled_spikes, evenly_spaced_time):
    """
    Generates a binary spike train on the evenly spaced time grid.
    
    Args:
        upsampled_spikes (np.ndarray or list): Spike times in seconds.
        evenly_spaced_time (np.ndarray): The time grid.
        
    Returns:
        np.ndarray: Binary array (0s and 1s) corresponding to spike events on the grid.
    """
    if len(evenly_spaced_time) < 2:
        return np.zeros_like(evenly_spaced_time)
        
    # Ensure inputs are numpy arrays
    upsampled_spikes = np.asarray(upsampled_spikes)
    evenly_spaced_time = np.asarray(evenly_spaced_time)

    dt = evenly_spaced_time[1] - evenly_spaced_time[0]
    t_start = evenly_spaced_time[0]
    
    spike_train = np.zeros(len(evenly_spaced_time))
    
    # INDICES calculation: (time - start) / dt
    # ROUND to nearest index followed by int casting
    # Adding a small epsilon to handle floating point issues if specific need arises, but round is usually robust for midpoints
    spike_indices = np.round((upsampled_spikes - t_start) / dt).astype(int)
    
    # Filter valid indices
    valid_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(spike_train))]
    
    # Set spikes to 1
    spike_train[valid_indices] = 1
    
    return spike_train

# -----------------------------------------------------------------------------
# Legacy / Iterative Simulation Functions
# These functions implement iterative integration which allows separating
# process noise (intra-cellular) from recording noise.
# -----------------------------------------------------------------------------

def simulate_calcium_iterative_true_spikes(true_spikes_sec, time_grid, tau_frames, sampling_rate,
                                           noise_intra=0.01, noise_rec=1.0):
    """
    User's exact iterative logic applied to true spike timestamps.
    """
    sim_dur = len(time_grid)
    t_start = time_grid[0]
    
    # 1. Convert spikes to "current" grid (re-binning)
    spikes = np.zeros(sim_dur)
    spike_indices = ((true_spikes_sec - t_start) * sampling_rate).astype(int)
    valid_indices = spike_indices[(spike_indices >= 0) & (spike_indices < sim_dur)]
    np.add.at(spikes, valid_indices, 1.0)
    
    # 2. Add Process Noise (Intra-cellular)
    rng = np.random.default_rng()
    noise_intra_arr = rng.normal(0, noise_intra, sim_dur)
    spikes_noisy = spikes + noise_intra_arr
    
    # 3. Iterative Integration (User's Logic)
    calcium = np.zeros(sim_dur) # "Clean" spike integration (as ref)
    calcium_nsp = np.zeros(sim_dur) # "Noisy" spike integration
    
    # Using dt=1 frame as per user's snippet logic
    # const_A = exp(-1/tau)
    const_A = np.exp(-1.0 / tau_frames)
    
    calcium[0] = spikes[0]
    calcium_nsp[0] = spikes_noisy[0] # Note: user snippet used spikes[0] here, but noisy flow makes sense
    
    for t in range(1, sim_dur):
        calcium[t] = const_A * calcium[t-1] + spikes[t]
        calcium_nsp[t] = const_A * calcium_nsp[t-1] + spikes_noisy[t]
        
    # 4. Add Recording Noise
    noise_recording = rng.normal(0, noise_rec, sim_dur)
    # output = clean_integral + noise_rec (typically what we want to compare against "true")
    # or output = noisy_integral + noise_rec (full simulation)
    
    # Returning the fully noisy version as it represents realistic data generation
    return calcium_nsp + noise_recording

def simulate_calcium_iterative_rec_spikes(rec_spikes, tau_frames, noise_intra=0.01, noise_rec=1.0):
    """
    User's exact iterative logic applied to reconstructed continuous spikes.
    """
    sim_dur = len(rec_spikes)
    spikes = rec_spikes # Continuous input
    
    # 1. Add Process Noise
    rng = np.random.default_rng()
    noise_intra_arr = rng.normal(0, noise_intra, sim_dur)
    spikes_noisy = spikes + noise_intra_arr
    
    # 2. Iterative Integration
    calcium_nsp = np.zeros(sim_dur)
    const_A = np.exp(-1.0 / tau_frames)
    
    calcium_nsp[0] = spikes_noisy[0]
    
    for t in range(1, sim_dur):
        calcium_nsp[t] = const_A * calcium_nsp[t-1] + spikes_noisy[t]
        
    # 3. Add Recording Noise
    noise_recording = rng.normal(0, noise_rec, sim_dur)
    
    return calcium_nsp + noise_recording
