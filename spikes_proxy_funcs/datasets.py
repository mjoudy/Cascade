
import numpy as np
import matplotlib.pyplot as plt
from cascade2p import utils as cascade_utils

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
        if num_datasets is not None:
            selected_names = dataset_names[:num_datasets]
        else:
            selected_names = dataset_names
            
        filtered_datasets = {}
        
        for name in selected_names:
            recordings = all_datasets[name]
            
            # Filter neurons within dataset
            if num_neurons_per_dataset is not None:
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
