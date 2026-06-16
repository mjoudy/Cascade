"""
End-to-end spike-proxy analysis pipeline.

This module encapsulates the per-neuron processing that used to live as a long
sequence of cells in active_neurons_clean.ipynb. Each stage is a small function
that mutates the list of neuron dicts in place; `run_analysis` chains them in
order, and `build_measurements_dataframe` collapses the result into the flat
scalar-metric table that gets exported to CSV.

Typical use from a notebook:

    from functions.pipeline import run_analysis, build_measurements_dataframe

    neurons, optimal_bin_ms = run_analysis()           # full build
    df, df_measurements     = build_measurements_dataframe(neurons)

The numerical logic is a faithful extraction of the original notebook cells;
default parameters reproduce the published results.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit

from functions.data_manager import load_selected_data, flatten_datasets, upsample_v2
from functions.derivative_method import estimate_tau
from functions.reconstruction import (
    get_spikes_train,
    reconstruct_spikes,
    simulate_calcium_from_true_spikes,
    simulate_calcium_from_reconstructed_spikes,
)
from functions.metrics import (
    lorentzian,
    compute_gof, ols_scale, lad_scale, add_chi2_nmse,
    analyze_cumsum_calibration,
    compute_cumsum_correlation_metrics,
    compute_binned_rate_correlation_metrics,
    compute_signal_correlation_metrics,
    safe_spearman,
    neuron_spike_metrics,
)
from functions.groups import (
    build_group_indices,
    GROUPS_BY_INDICATOR, GROUPS_ORDER_BY_INDICATOR, DS_TO_GROUP_BY_INDICATOR,
)


# ── Stage 1: load + filter active neurons ─────────────────────────────────────

def load_active_neurons(ground_truth_folder='Ground_truth', min_spikes=100,
                        num_datasets='all', num_neurons_per_dataset='all', verbose=True):
    """Load ground truth, flatten, and keep neurons with > min_spikes spikes."""
    data_dict = load_selected_data(ground_truth_folder=ground_truth_folder,
                                   num_datasets=num_datasets,
                                   num_neurons_per_dataset=num_neurons_per_dataset)
    flat_data = flatten_datasets(data_dict)
    active = [n for n in flat_data if len(n['spikes']) > min_spikes]
    if verbose:
        print(f"Total neurons:  {len(flat_data)}")
        print(f"Active neurons: {len(active)}")
    return active


# ── Stage 2: resample to a common grid ────────────────────────────────────────

def upsample_neurons(data_active_neurons, new_rate=200, verbose=True):
    """Upsample each neuron to new_rate Hz; return the list of neuron dicts."""
    neurons = []
    for neuron in data_active_neurons:
        upsampled_signal, upsampled_spikes, evenly_spaced_time = upsample_v2(
            times=neuron['t'], dff=neuron['dff'], spikes=neuron['spikes'],
            new_rate=new_rate, do_plot=False,
        )
        neurons.append({
            'original_neuron':   neuron,
            'upsampled_signal':  upsampled_signal,
            'upsampled_spikes':  upsampled_spikes,
            'evenly_spaced_time': evenly_spaced_time,
        })
    if verbose:
        print(f"Upsampled {len(neurons)} neurons.")
    return neurons


# ── Stage 3: τ via derivative / phase-space method ────────────────────────────

def add_tau_derivative(neurons, window_len=51, poly_order=3, cut_win=10, verbose=True):
    """Estimate τ from the signal/derivative phase space; store tau_frames/tau_ms."""
    for item in neurons:
        u_sig       = item['upsampled_signal']
        u_spk_times = item['upsampled_spikes']
        u_time      = item['evenly_spaced_time']

        fs      = 1.0 / (u_time[1] - u_time[0]) if len(u_time) > 1 else 30.0
        t_start = u_time[0]

        u_spk_indices = ((u_spk_times - t_start) * fs).astype(int)
        u_spk_indices = u_spk_indices[(u_spk_indices >= 0) & (u_spk_indices < len(u_sig))]

        try:
            tau_val, fit_data = estimate_tau(u_sig, u_spk_indices,
                                             window_len=window_len, poly_order=poly_order, cut_win=cut_win)
        except Exception:
            tau_val, fit_data = np.nan, (None, None)

        item['tau_frames'] = tau_val
        item['tau_ms']     = (tau_val / fs) * 1000
        item['fit_data']   = fit_data
    if verbose:
        print(f"Initialised {len(neurons)} neurons with tau_frames / tau_ms / fit_data.")
    return neurons


# ── Stage 4: binary spike train on the upsampled grid ─────────────────────────

def add_spike_train(neurons, verbose=True):
    for n in neurons:
        n['spikes_train'] = get_spikes_train(n['upsampled_spikes'], n['evenly_spaced_time'])
    if verbose:
        print(f"Added 'spikes_train' to {len(neurons)} neurons.")
    return neurons


# ── Stage 5: reconstructed spikes (derivative τ) ──────────────────────────────

def add_reconstructed_deriv(neurons, verbose=True):
    for n in neurons:
        tau = n['tau_frames']
        if not np.isnan(tau) and tau > 0:
            n['reconstructed_spikes'] = reconstruct_spikes(n['upsampled_signal'], tau=tau)
        else:
            n['reconstructed_spikes'] = np.zeros_like(n['upsampled_signal'])
    if verbose:
        print("Added 'reconstructed_spikes' to all neurons.")
    return neurons


# ── Stage 6: simulated calcium (derivative τ) ─────────────────────────────────

def add_simulated_deriv(neurons, verbose=True):
    for n in neurons:
        tau       = n.get('tau_frames')
        time_grid = n['evenly_spaced_time']
        fs        = 1.0 / (time_grid[1] - time_grid[0]) if len(time_grid) > 1 else 100.0
        rec_spikes = n['reconstructed_spikes']

        if tau is not None and not np.isnan(tau) and tau > 0:
            n['sim_calcium_from_true'] = simulate_calcium_from_true_spikes(
                true_spikes_sec=n['original_neuron']['spikes'], time_grid=time_grid,
                tau_frames=tau, sampling_rate=fs, noise_std=0.0,
            )
            n['sim_calcium_from_reconstructed'] = simulate_calcium_from_reconstructed_spikes(
                rec_spikes=rec_spikes, tau_frames=tau, noise_std=0.0,
            )
        else:
            zeros = np.zeros_like(n['upsampled_signal'])
            n['sim_calcium_from_true']          = zeros
            n['sim_calcium_from_reconstructed'] = zeros
    if verbose:
        print("Added 'sim_calcium_from_true' and 'sim_calcium_from_reconstructed'.")
    return neurons


# ── Stage 7: blind deconvolution — PSD Lorentzian fit ─────────────────────────

def add_tau_psd(neurons, f_starts=(0.5, 1.0, 1.5), f_end=30.0, verbose=True):
    """Estimate τ by fitting a Lorentzian to the multitaper PSD (nitime)."""
    import nitime.algorithms as tsa   # lazy: nitime is only needed for this stage

    for n in tqdm(neurons, desc="PSD Lorentzian fit", disable=not verbose):
        time_vector = n['evenly_spaced_time']
        fs     = 1.0 / (time_vector[1] - time_vector[0]) if len(time_vector) > 1 else 200.0
        signal = n['upsampled_signal']

        try:
            freq_tsa, psd_tsa, _ = tsa.multi_taper_psd(signal, Fs=fs)
        except Exception as e:
            n['tau_psd_frames'] = np.nan
            n['tau_psd_ms']     = np.nan
            n['psd_fit_info']   = {'status': f'PSD failed: {e}'}
            continue

        fit_success = False
        for f_min in f_starts:
            mask  = (freq_tsa >= f_min) & (freq_tsa <= f_end)
            f_fit = freq_tsa[mask]
            p_fit = psd_tsa[mask]

            finite = np.isfinite(f_fit) & np.isfinite(p_fit)
            f_fit, p_fit = f_fit[finite], p_fit[finite]
            if len(f_fit) < 4:
                continue

            try:
                popt, pcov = curve_fit(
                    lorentzian, f_fit, p_fit,
                    p0=[p_fit[0], 0.1, np.min(p_fit)],
                    bounds=([0, 1e-5, 0], [np.inf, 10.0, np.inf]),
                    maxfev=5000,
                )
                A_fit, tau_sec, C_fit = popt
                tau_frames = tau_sec * fs
                psd_pred   = lorentzian(f_fit, *popt)
                ss_res     = np.sum((p_fit - psd_pred) ** 2)
                ss_tot     = np.sum((p_fit - np.mean(p_fit)) ** 2)
                r_squared  = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                perr       = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan] * 3

                n['tau_psd_frames'] = tau_frames
                n['tau_psd_ms']     = tau_sec * 1000
                n['psd_fit_info']   = {
                    'status': 'ok', 'f_min': f_min, 'f_max': f_end,
                    'A': A_fit, 'tau_sec': tau_sec, 'tau_frames': tau_frames,
                    'tau_ms': tau_sec * 1000, 'white_noise': C_fit,
                    'r_squared': r_squared,
                    'std_A': perr[0], 'std_tau': perr[1], 'std_wn': perr[2],
                }
                fit_success = True
                break
            except RuntimeError:
                continue

        if not fit_success:
            n['tau_psd_frames'] = np.nan
            n['tau_psd_ms']     = np.nan
            n['psd_fit_info']   = {'status': 'fit_failed'}

    if verbose:
        n_ok  = sum(1 for n in neurons if n.get('psd_fit_info', {}).get('status') == 'ok')
        print(f"PSD fit: {n_ok} OK / {len(neurons) - n_ok} failed or error")
    return neurons


# ── Stage 8: reconstructed spikes (PSD τ) ─────────────────────────────────────

def add_reconstructed_psd(neurons, verbose=True):
    for n in neurons:
        tau_psd = n.get('tau_psd_frames')
        if tau_psd is not None and not np.isnan(tau_psd) and tau_psd > 0:
            n['reconstructed_spikes_psd'] = reconstruct_spikes(n['upsampled_signal'], tau=tau_psd)
        else:
            n['reconstructed_spikes_psd'] = np.zeros_like(n['upsampled_signal'])
    if verbose:
        print("Added 'reconstructed_spikes_psd'.")
    return neurons


# ── Stage 9: simulated calcium (PSD τ) ────────────────────────────────────────

def add_simulated_psd(neurons, verbose=True):
    for n in neurons:
        tau_psd   = n.get('tau_psd_frames')
        time_grid = n['evenly_spaced_time']
        fs        = 1.0 / (time_grid[1] - time_grid[0]) if len(time_grid) > 1 else 100.0
        rec_psd   = n.get('reconstructed_spikes_psd')

        if tau_psd is not None and not np.isnan(tau_psd) and tau_psd > 0:
            n['sim_calcium_from_rec_psd'] = simulate_calcium_from_reconstructed_spikes(
                rec_spikes=rec_psd, tau_frames=tau_psd,
            )
            n['sim_calcium_from_true_psd'] = simulate_calcium_from_true_spikes(
                true_spikes_sec=n['original_neuron']['spikes'], time_grid=time_grid,
                tau_frames=tau_psd, sampling_rate=fs, noise_std=0.0,
            )
        else:
            zeros = np.zeros_like(n['upsampled_signal'])
            n['sim_calcium_from_rec_psd']  = None
            n['sim_calcium_from_true_psd'] = zeros
    if verbose:
        print("Added 'sim_calcium_from_rec_psd' and 'sim_calcium_from_true_psd'.")
    return neurons


# ── Stage 10: goodness-of-fit (no-scale / OLS / LAD) + χ² + NMSE ───────────────

def add_gof_metrics(neurons, verbose=True):
    for n in neurons:
        original = n['upsampled_signal']
        for sim_key, gof_key in [
            ('sim_calcium_from_true',          'gof_tau_true'),
            ('sim_calcium_from_reconstructed', 'gof_tau'),
            ('sim_calcium_from_true_psd',      'gof_tau_psd_true'),
            ('sim_calcium_from_rec_psd',       'gof_tau_psd'),
        ]:
            sim = n.get(sim_key)
            if sim is None or np.all(np.asarray(sim) == 0) or np.std(sim) == 0:
                n[gof_key] = None
                continue

            sim = np.asarray(sim)
            gof = {}
            gof['no_scale']                 = compute_gof(original, sim)
            gof['no_scale']['scale_factor'] = 1.0

            a_ols                    = ols_scale(original, sim)
            gof['ols']               = compute_gof(original, a_ols * sim)
            gof['ols']['scale_factor'] = a_ols

            a_lad                    = lad_scale(original, sim)
            gof['lad']               = compute_gof(original, a_lad * sim)
            gof['lad']['scale_factor'] = a_lad

            for scale_key in ('no_scale', 'ols', 'lad'):
                sf = gof[scale_key]['scale_factor']
                add_chi2_nmse(gof[scale_key], original, sf * sim)

            n[gof_key] = gof
    if verbose:
        print("Added gof_tau / gof_tau_psd / gof_tau_true / gof_tau_psd_true to all neurons.")
    return neurons


# ── Stage 11: cumsum calibration (derivative + PSD τ) ─────────────────────────

def add_cumsum_calibration(neurons, verbose=True):
    for n in neurons:
        time_vector       = n['evenly_spaced_time']
        true_spikes_train = get_spikes_train(n['original_neuron']['spikes'], time_vector)

        cal = analyze_cumsum_calibration(true_spikes_train, n['reconstructed_spikes'])
        n['cumsum_slope_raw']   = cal['slope_raw']
        n['cumsum_r2_calib']    = cal['r2_calib']
        n['cumsum_slope_calib'] = cal['slope_calib']
        n['calibrated_spikes']  = cal['calibrated_trace']

        cal_psd = analyze_cumsum_calibration(true_spikes_train, n['reconstructed_spikes_psd'])
        n['cumsum_slope_raw_psd']   = cal_psd['slope_raw']
        n['cumsum_r2_calib_psd']    = cal_psd['r2_calib']
        n['cumsum_slope_calib_psd'] = cal_psd['slope_calib']
        n['calibrated_spikes_psd']  = cal_psd['calibrated_trace']
    if verbose:
        print("Added cumsum calibration fields (deriv + PSD τ).")
    return neurons


# ── Stage 12: cross-comparisons — PSD vs True / PSD vs Deriv ───────────────────

def add_cumsum_cross_comparisons(neurons, verbose=True):
    for n in neurons:
        rec_spikes        = n['reconstructed_spikes']
        rec_spikes_psd    = n.get('reconstructed_spikes_psd')
        true_spikes_train = get_spikes_train(n['original_neuron']['spikes'], n['evenly_spaced_time'])

        if rec_spikes_psd is None:
            for key in ['cumsum_psd_vs_true_slope_raw',  'cumsum_psd_vs_true_r2_calib',
                        'cumsum_psd_vs_true_slope_calib', 'cumsum_psd_vs_deriv_slope_raw',
                        'cumsum_psd_vs_deriv_r2_calib',   'cumsum_psd_vs_deriv_slope_calib']:
                n[key] = np.nan
            continue

        cal_psd_vs_true  = analyze_cumsum_calibration(true_spikes_train, rec_spikes_psd)
        cal_psd_vs_deriv = analyze_cumsum_calibration(rec_spikes, rec_spikes_psd)

        n['cumsum_psd_vs_true_slope_raw']    = cal_psd_vs_true['slope_raw']
        n['cumsum_psd_vs_true_r2_calib']     = cal_psd_vs_true['r2_calib']
        n['cumsum_psd_vs_true_slope_calib']  = cal_psd_vs_true['slope_calib']
        n['cumsum_psd_vs_deriv_slope_raw']   = cal_psd_vs_deriv['slope_raw']
        n['cumsum_psd_vs_deriv_r2_calib']    = cal_psd_vs_deriv['r2_calib']
        n['cumsum_psd_vs_deriv_slope_calib'] = cal_psd_vs_deriv['slope_calib']
    if verbose:
        print("Added PSD-vs-True and PSD-vs-Deriv cumsum comparisons.")
    return neurons


# ── Stage 13: Pearson & Spearman — cumsum ─────────────────────────────────────

def add_cumsum_correlations(neurons, verbose=True):
    for n in neurons:
        true_spikes_train = get_spikes_train(n['original_neuron']['spikes'], n['evenly_spaced_time'])

        corr = compute_cumsum_correlation_metrics(true_spikes_train, n['reconstructed_spikes'])
        n['pearson_cumsum_raw']    = corr['pearson_cumsum_raw']
        n['pearson_cumsum_calib']  = corr['pearson_cumsum_calib']
        n['spearman_cumsum_raw']   = corr['spearman_cumsum_raw']
        n['spearman_cumsum_calib'] = corr['spearman_cumsum_calib']

        rec_psd = n.get('reconstructed_spikes_psd')
        if rec_psd is None:
            n.update({k: np.nan for k in ['pearson_cumsum_raw_psd', 'pearson_cumsum_calib_psd',
                                          'spearman_cumsum_raw_psd', 'spearman_cumsum_calib_psd']})
        else:
            corr_psd = compute_cumsum_correlation_metrics(true_spikes_train, rec_psd)
            n['pearson_cumsum_raw_psd']    = corr_psd['pearson_cumsum_raw']
            n['pearson_cumsum_calib_psd']  = corr_psd['pearson_cumsum_calib']
            n['spearman_cumsum_raw_psd']   = corr_psd['spearman_cumsum_raw']
            n['spearman_cumsum_calib_psd'] = corr_psd['spearman_cumsum_calib']
    if verbose:
        print("Added Pearson / Spearman cumsum metrics.")
    return neurons


# ── Stage 14: per-group optimal bin width + binned-rate correlations ──────────

def sweep_optimal_bin_ms(neurons, bin_widths_s=None, verbose=True):
    """Find, per by-indicator group, the bin width maximising median Spearman ρ
    of binned reconstructed vs true rate."""
    if bin_widths_s is None:
        bin_widths_s = np.logspace(np.log10(0.05), np.log10(3.0), 12)   # 50 ms → 3 s
    group_indices_binned = build_group_indices(neurons, DS_TO_GROUP_BY_INDICATOR)
    optimal_bin_ms = {}

    for grp in tqdm(GROUPS_ORDER_BY_INDICATOR, desc="Bin sweep", disable=not verbose):
        indices = group_indices_binned[grp]
        if not indices:
            continue

        rho_mat = np.full((len(indices), len(bin_widths_s)), np.nan)
        for row_i, ni in enumerate(indices):
            nd       = neurons[ni]
            original = nd.get('upsampled_signal')
            time_vec = nd.get('evenly_spaced_time')
            tau_f    = nd.get('tau_frames')
            if original is None or time_vec is None or tau_f is None or np.isnan(tau_f):
                continue

            T  = len(time_vec)
            fs = 1.0 / (time_vec[1] - time_vec[0]) if T > 1 else 200.0
            true_train = get_spikes_train(nd['original_neuron']['spikes'], time_vec).astype(float)

            try:
                rec = np.clip(reconstruct_spikes(original, tau=tau_f), 0, None)
            except Exception:
                continue

            for bi, bw in enumerate(bin_widths_s):
                bin_frames = max(1, int(round(bw * fs)))
                n_bins     = T // bin_frames
                if n_bins < 3:
                    continue
                sl     = n_bins * bin_frames
                true_b = true_train[:sl].reshape(n_bins, bin_frames).sum(axis=1)
                rec_b  = rec[:sl].reshape(n_bins, bin_frames).sum(axis=1)
                rho_mat[row_i, bi] = safe_spearman(true_b, rec_b)

        med_rho = np.nanmedian(rho_mat, axis=0)
        best_bi = int(np.nanargmax(med_rho))
        optimal_bin_ms[grp] = float(bin_widths_s[best_bi] * 1000)
        if verbose:
            print(f"{grp:38s}  best bin = {optimal_bin_ms[grp]:6.0f} ms   ρ = {med_rho[best_bi]:.3f}")
    return optimal_bin_ms


def add_binned_rate_correlations(neurons, optimal_bin_ms, default_bin_ms=500.0, verbose=True):
    ds_to_optimal_bin = {
        ds: optimal_bin_ms.get(grp, default_bin_ms)
        for grp, dss in GROUPS_BY_INDICATOR.items()
        for ds in dss
    }

    for n in neurons:
        time_vector       = n['evenly_spaced_time']
        fs                = 1.0 / (time_vector[1] - time_vector[0]) if len(time_vector) > 1 else 200.0
        true_spikes_train = get_spikes_train(n['original_neuron']['spikes'], time_vector)
        ds                = n['original_neuron'].get('dataset_name', '')
        bin_ms            = ds_to_optimal_bin.get(ds, default_bin_ms)

        n['binned_rate_bin_ms'] = bin_ms

        corr = compute_binned_rate_correlation_metrics(
            true_spikes_train, n['reconstructed_spikes'], bin_ms=bin_ms, fs=fs)
        n['pearson_binned_rate_raw']  = corr['pearson_binned_rate']
        n['spearman_binned_rate_raw'] = corr['spearman_binned_rate']

        rec_psd = n.get('reconstructed_spikes_psd')
        if rec_psd is None:
            n['pearson_binned_rate_raw_psd']  = np.nan
            n['spearman_binned_rate_raw_psd'] = np.nan
        else:
            corr_psd = compute_binned_rate_correlation_metrics(
                true_spikes_train, rec_psd, bin_ms=bin_ms, fs=fs)
            n['pearson_binned_rate_raw_psd']  = corr_psd['pearson_binned_rate']
            n['spearman_binned_rate_raw_psd'] = corr_psd['spearman_binned_rate']
    if verbose:
        print("Added binned rate metrics with group-optimal bin widths.")
    return neurons


# ── Stage 15: Pearson & Spearman — calcium signal ─────────────────────────────

def add_signal_correlations(neurons, verbose=True):
    for n in neurons:
        original = n['upsampled_signal']
        for sim_key, prefix in [
            ('sim_calcium_from_reconstructed', 'gof_tau'),
            ('sim_calcium_from_rec_psd',       'gof_tau_psd'),
        ]:
            sim     = n.get(sim_key)
            sim_arr = np.asarray(sim) if sim is not None else None

            if sim_arr is None or np.all(sim_arr == 0) or np.std(sim_arr) == 0:
                for sfx in ['pearson_raw', 'pearson_scaled', 'spearman_raw', 'spearman_scaled']:
                    n[f'{prefix}_{sfx}'] = np.nan
                continue

            corr = compute_signal_correlation_metrics(original, sim_arr)
            for sfx in ['pearson_raw', 'pearson_scaled', 'spearman_raw', 'spearman_scaled']:
                n[f'{prefix}_{sfx}'] = corr[sfx]
    if verbose:
        print("Added Pearson / Spearman calcium metrics.")
    return neurons


# ── Stage 16: NMSE — cumulative spike count vs identity ───────────────────────

def add_cumsum_nmse(neurons, verbose=True):
    for n in neurons:
        true_spikes_train = get_spikes_train(n['original_neuron']['spikes'], n['evenly_spaced_time'])
        x_true = np.cumsum(true_spikes_train).astype(float)
        var_x  = np.var(x_true, ddof=1)

        for cal_key, out_key in [
            ('calibrated_spikes',     'nmse_cumsum'),
            ('calibrated_spikes_psd', 'nmse_cumsum_psd'),
        ]:
            cal = n.get(cal_key)
            if cal is not None and var_x > 0:
                y_pred = np.cumsum(np.asarray(cal)).astype(float)
                n[out_key] = float(np.mean((x_true - y_pred) ** 2) / var_x)
            else:
                n[out_key] = np.nan
    if verbose:
        print("Added 'nmse_cumsum' and 'nmse_cumsum_psd'.")
    return neurons


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_analysis(ground_truth_folder='Ground_truth', min_spikes=100, new_rate=200,
                 num_datasets='all', num_neurons_per_dataset='all',
                 bin_widths_s=None, verbose=True):
    """Run the full per-neuron pipeline end to end.

    Returns
    -------
    neurons : list of dict
        Per-neuron dicts populated with signals, τ estimates, reconstructions,
        simulations and all goodness-of-fit / correlation metrics.
    optimal_bin_ms : dict
        Per by-indicator group, the bin width (ms) that maximised median
        Spearman ρ in the bin-width sweep.
    """
    active = load_active_neurons(ground_truth_folder, min_spikes,
                                 num_datasets, num_neurons_per_dataset, verbose)
    neurons = upsample_neurons(active, new_rate, verbose)

    add_tau_derivative(neurons, verbose=verbose)
    add_spike_train(neurons, verbose=verbose)
    add_reconstructed_deriv(neurons, verbose=verbose)
    add_simulated_deriv(neurons, verbose=verbose)

    add_tau_psd(neurons, verbose=verbose)
    add_reconstructed_psd(neurons, verbose=verbose)
    add_simulated_psd(neurons, verbose=verbose)

    add_gof_metrics(neurons, verbose=verbose)
    add_cumsum_calibration(neurons, verbose=verbose)
    add_cumsum_cross_comparisons(neurons, verbose=verbose)
    add_cumsum_correlations(neurons, verbose=verbose)

    optimal_bin_ms = sweep_optimal_bin_ms(neurons, bin_widths_s=bin_widths_s, verbose=verbose)
    add_binned_rate_correlations(neurons, optimal_bin_ms, verbose=verbose)

    add_signal_correlations(neurons, verbose=verbose)
    add_cumsum_nmse(neurons, verbose=verbose)

    return neurons, optimal_bin_ms


# ── DataFrame assembly ────────────────────────────────────────────────────────

# Array/object columns dropped when collapsing to the scalar-metrics table.
SIGNAL_COLS = [
    'upsampled_signal', 'upsampled_spikes', 'evenly_spaced_time', 'fit_data',
    'spikes_train', 'reconstructed_spikes', 'reconstructed_spikes_psd',
    'sim_calcium_from_true', 'sim_calcium_from_reconstructed',
    'sim_calcium_from_rec_psd', 'sim_calcium_from_true_psd',
    'calibrated_spikes', 'calibrated_spikes_psd',
    'gof_tau_true', 'gof_tau_psd_true',
    'original_neuron',
]


def build_measurements_dataframe(neurons, verbose=True):
    """Collapse the neuron dicts into (df, df_measurements).

    df              : full DataFrame including array/object columns.
    df_measurements : scalar-only table (arrays dropped, nested GOF/PSD metrics
                      flattened) — this is what gets exported to CSV.
    """
    df = pd.DataFrame(neurons)

    firing_rates, mean_isis, cv2s, burst_indices = zip(*[neuron_spike_metrics(n) for n in neurons])
    df['firing_rate_hz'] = firing_rates
    df['mean_isi_s']     = mean_isis
    df['cv2']            = cv2s
    df['burst_index']    = burst_indices

    df['global_id']    = df['original_neuron'].apply(lambda x: x.get('global_id'))
    df['dataset_name'] = df['original_neuron'].apply(lambda x: x.get('dataset_name'))
    df['frame_rate']   = df['original_neuron'].apply(lambda x: x.get('frame_rate'))

    df_measurements = df.copy()

    def _safe_get(gof_dict, scale, metric):
        try:
            return gof_dict[scale][metric]
        except (TypeError, KeyError):
            return np.nan

    def _col_apply(frame, col, func):
        if col not in frame.columns:
            return pd.Series(np.nan, index=frame.index)
        return frame[col].apply(func)

    df_measurements['gof_tau_r2']     = _col_apply(df, 'gof_tau',     lambda x: _safe_get(x, 'ols', 'r_squared'))
    df_measurements['gof_tau_psd_r2'] = _col_apply(df, 'gof_tau_psd', lambda x: _safe_get(x, 'ols', 'r_squared'))

    df_measurements['gof_tau_chi2']     = _col_apply(df, 'gof_tau',     lambda x: _safe_get(x, 'ols', 'chi_squared'))
    df_measurements['gof_tau_nmse']     = _col_apply(df, 'gof_tau',     lambda x: _safe_get(x, 'ols', 'nmse'))
    df_measurements['gof_tau_psd_chi2'] = _col_apply(df, 'gof_tau_psd', lambda x: _safe_get(x, 'ols', 'chi_squared'))
    df_measurements['gof_tau_psd_nmse'] = _col_apply(df, 'gof_tau_psd', lambda x: _safe_get(x, 'ols', 'nmse'))

    df_measurements['psd_fit_r2']      = _col_apply(df, 'psd_fit_info', lambda x: x.get('r_squared', np.nan) if isinstance(x, dict) else np.nan)
    df_measurements['psd_fit_std_tau'] = _col_apply(df, 'psd_fit_info', lambda x: x.get('std_tau', np.nan)   if isinstance(x, dict) else np.nan)

    df_measurements = df_measurements.drop(
        columns=SIGNAL_COLS + ['gof_tau', 'gof_tau_psd', 'psd_fit_info'],
        errors='ignore',
    )

    if verbose:
        print(f"df shape: {df.shape}")
        print(f"df_measurements shape: {df_measurements.shape}")
    return df, df_measurements
