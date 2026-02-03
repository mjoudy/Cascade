# Spikes Proxy Analysis Project

This project focuses on the analysis of active neurons using a new method for estimating the decay time constant ($\tau$) of calcium signals. The current repository is a fork of the Cascade project, enhanced with custom analysis tools and workflows.

## Project Overview

The primary goal of this project is to analyze "active neurons" (neurons with significant spiking activity) from calcium imaging datasets. The core of this analysis involves a novel method for estimating the decay time constant ($\tau$) by analyzing the phase space of the signal and its derivative, specifically isolating the decay phase.

## New Analysis Method

This project implements a custom analysis pipeline developed and verified on synthetic data. The method is now applied to real experimental data within this repository.

### Key Components:

1.  **Preprocessing & Spike Cutting (`cut_spikes`):**
    The method identifies spike events and effectively "cuts out" the rising phase and the immediate peak of the spike. This isolates the "decay-only" portion of the calcium signal, which is governed by the system's time constant.

2.  **Robust Tau Estimation (`estimate_tau`):**
    *   **Smoothing:** The signal and its derivative are smoothed using Savitzky-Golay filters.
    *   **Phase Space Analysis:** The method plots the signal against its derivative. For a simple exponential decay ($C(t) = C_0 e^{-t/\tau}$), this relationship is linear: $\frac{dC}{dt} = -\frac{1}{\tau} C(t)$.
    *   **Fitting:** A linear fit is applied to the isolated decay phases in this phase space. The negative inverse of the slope yields the estimated $\tau$.

3.  **Reconstruction & Validation:**
    *   **Reconstruction:** Using the estimated $\tau$, the underlying spike train is reconstructed from the calcium signal, effectively deconvolving the dynamics.
    *   **Calibration:** The method includes metrics to calibrate the magnitude of the reconstructed spikes against the original data, ensuring accurate quantitative analysis.

## Repository Structure

*   **`active_neurons.ipynb`**: The main analysis notebook. It performs the following steps:
    *   Loads datasets using `datasets.load_selected_data`.
    *   Filters for active neurons (> 100 spikes).
    *   Upsamples signals to 100Hz for high-resolution analysis.
    *   Applies the `estimate_tau` method to filtered neurons.
    *   Visualizes original vs. upsampled data and analysis results.

*   **`functions/`**: A package containing the core implementations of the new method:
    *   `derivative_method.py`: Contains `estimate_tau` and `cut_spikes` functions.
    *   `reconstruction.py`: Functions for spike reconstruction and signal simulation.
    *   `metrics.py`: Metrics for evaluating reconstruction quality, including cumulative sum slope analysis and binned comparisons.
    *   `data_manager.py`: (Implied) Handles data loading and dataset management.

## Integration with Forked Data

This repository utilizes the `cascade` codebase structure but introduces a simplified and targeted data pipeline. 
*   Data is loaded into a dictionary structure (`data_dict`).
*   It is then flattened and filtered (`data_active_neurons`).
*   The `processing.upsample_v2` function ensures all data is on a consistent time grid (100Hz) before the new Tau estimation method is applied.

## Usage Example

To run the analysis on active neurons:

1.  **Open `active_neurons.ipynb`**.
2.  **Load Data:** Run the initial cells to load `data_dict` via `datasets.load_selected_data`.
3.  **Preprocessing:** Execute the flattening and filtering steps to generate `data_active_neurons`.
4.  **Upsampling:** Run the upsampling loop to prepare `upsampled_results`.
5.  **Analysis:** The notebook applies `estimate_tau` to the upsampled signals:
    ```python
    from functions.derivative_method import estimate_tau
    
    # Example call
    tau_val, fit_data = estimate_tau(u_sig, u_spk, window_len=51, poly_order=3, cut_win=10)
    ```
6.  **Visualize:** Use `plot_neuron_comparison` to inspect individual neuron results.

## Original Project
This repository is a fork of [Cascade](https://github.com/mjoudy/spikes-proxy). Please refer to `README_original.md` for information regarding the original Cascade implementation.
