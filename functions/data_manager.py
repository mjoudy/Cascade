import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import re
from cascade2p import utils as cascade_utils

# -----------------------------------------------------------------------------
# Data Loading & Filtering (From datasets.py)
# -----------------------------------------------------------------------------

def load_selected_data(ground_truth_folder='Ground_truth', num_datasets=None, num_neurons_per_dataset=None):
    """
    Loads ground truth data and filters it to a specified number of datasets and neurons.

    Parameters:
    -----------
    ground_truth_folder : str
        Path to the folder containing dataset files.
    num_datasets : int, optional
        Number of datasets to load. If None, loads all available datasets.
    num_neurons_per_dataset : int, optional
        Number of neurons (recordings) to keep per dataset. If None, keeps all neurons.

    Returns:
    --------
    filtered_datasets : dict
        A dictionary where keys are dataset names and values are lists of recording dictionaries.
    """
    try:
        # Load all available data
        all_datasets = cascade_utils.load_all_ground_truth(ground_truth_folder=ground_truth_folder)
        
        dataset_names = sorted(list(all_datasets.keys()))
        
        # Filter datasets
        if num_datasets is not None and num_datasets != 'all':
            selected_names = dataset_names[:num_datasets]
        else:
            selected_names = dataset_names
            
        filtered_datasets = {}
        
        for name in selected_names:
            recordings = all_datasets[name]
            
            # Filter neurons within dataset
            if num_neurons_per_dataset is not None and num_neurons_per_dataset != 'all':
                selected_recordings = recordings[:num_neurons_per_dataset]
            else:
                selected_recordings = recordings
                
            filtered_datasets[name] = selected_recordings
            
        return filtered_datasets

    except FileNotFoundError:
        print(f"Error: Folder '{ground_truth_folder}' not found.")
        return {}
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return {}

def flatten_datasets(data_dict):
    """
    Flattens a dictionary of datasets into a single list of neurons.
    
    Parameters:
    -----------
    data_dict : dict
        The dictionary returned by load_selected_data.
        
    Returns:
    --------
    flat_data : list
        A list of dictionaries, where each dictionary represents a neuron
        and includes 'global_id', 'dataset_name', 'original_index' plus all original fields.
    """
    flat_data = []
    global_idx = 0
    # Process datasets in sorted order to ensure deterministic ID assignment
    for ds_name in sorted(data_dict.keys()):
        for original_idx, neuron in enumerate(data_dict[ds_name]):
            # Create a new entry merging metadata and the original neuron data
            neuron_entry = {
                'global_id': global_idx,
                'dataset_name': ds_name,
                'original_index': original_idx,
                **neuron
            }
            flat_data.append(neuron_entry)
            global_idx += 1
    return flat_data

# -----------------------------------------------------------------------------
# Data Processing (Upsampling) (From processing.py)
# -----------------------------------------------------------------------------

def upsample_v2(times, dff, spikes, new_rate, intp_method='cubic', do_plot=False):
    """
    Upsamples the given signal to a new sampling rate using interpolation.
    
    Args:
        times (np.ndarray): Original time array.
        dff (np.ndarray): Fluorescence signal (dF/F).
        spikes (np.ndarray): Spike times in seconds.
        new_rate (float): Target sampling rate in Hz.
        intp_method (str): Interpolation method ('cubic', 'linear', etc.).
        do_plot (bool): Whether to plot the result for verification.
        
    Returns:
        tuple: (upsampled_signal, upsampled_spikes, evenly_spaced_time)
    """

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
    
    # Keep spikes in seconds, just filter to window
    upsampled_spikes = spikes[(spikes >= t_start) & (spikes <= t_end)]

    # --- 3. Plotting for verification ---
    if do_plot:
        plt.figure(figsize=(20, 5))
        
        # Plotting without 'evenly_spaced_time' to match V1's index-based x-axis
        plt.plot(upsampled_signal) 
        
        # We must convert seconds back to indices JUST for the plot
        # to match the index-based x-axis of the signal plot
        spike_indices = (upsampled_spikes - t_start) * new_rate
        
        # Logic for offsets
        if len(dff_clean) > 4:
            max_dff, min_dff = np.max(dff_clean[4:]), np.min(dff_clean[4:])
        else:
            max_dff, min_dff = np.max(dff_clean), np.min(dff_clean)
            
        plt.eventplot(spike_indices, 
                      lineoffsets=min_dff - max_dff/20, 
                      linelengths=max_dff/20, 
                      color='k')
        
        plt.title(f"Upsampled Signal (V2)")
        plt.show()

    return upsampled_signal, upsampled_spikes, evenly_spaced_time

# -----------------------------------------------------------------------------
# Visualization (From datasets.py)
# -----------------------------------------------------------------------------

def plot_neuron_trace(data_dict, recording_index=0, ax=None):
    """
    Plots the fluorescence trace and spikes for a single neuron recording.

    Parameters:
    -----------
    data_dict : dict
        A dictionary containing 't', 'dff', 'spikes', and 'frame_rate'.
    recording_index : int
        Index of the recording (used only for title/labeling if provided)
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 4))
    
    t = data_dict['t']
    dff = data_dict['dff']
    spikes = data_dict['spikes']
    
    # Plot fluorescence
    ax.plot(t, dff, color='tab:blue', label='dF/F', linewidth=1)
    
    # Plot spikes
    if len(dff) > 0:
        max_dff = np.max(dff)
        min_dff = np.min(dff)
        range_dff = max_dff - min_dff
        spike_offset = min_dff - range_dff * 0.1
        spike_length = range_dff * 0.1
    else:
        spike_offset = 0
        spike_length = 1

    ax.eventplot(spikes, lineoffsets=spike_offset, linelengths=spike_length, color='k', label='Spikes')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('dF/F')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
