"""
Plotting helpers for the spike-proxy analysis.

Collects every figure-producing routine in one place:

  * correlation histograms (linear / log) and ECDF helpers,
  * single-neuron trace / comparison / full-summary panels,
  * binned-spike comparison and the cumsum calibration dashboard.

Pure-computation functions live in functions.metrics / functions.reconstruction
and are imported here only where a plot needs to recompute a quantity.

Split out of notebook_utils.py, functions/metrics.py and functions/data_manager.py
during the repository cleanup.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

from functions.reconstruction import get_spikes_train
from functions.metrics import safe_spearman

# ── Correlation histogram constants & helpers ────────────────────────────────
#
# CORR_COLORS: [deriv-τ colour, PSD-τ colour] — used consistently across all
#              correlation figures so the same τ method always gets the same hue.
#
# Usage in notebook cells:
#   plot_linear_corr_hist(ax, data, label, color)   — for spread-out distributions
#   plot_log_corr_hist(ax, data, label, color)       — for distributions near 1

CORR_COLORS = ['#4C72B0', '#DD8452']   # [deriv-τ: blue, PSD-τ: orange]

CORR_RCPARAMS = {
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 11,        'axes.labelsize': 10,
    'xtick.labelsize': 8.5,      'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,          'ps.fonttype': 42,
}

_LOG_RHO_TICKS  = [-1.0, -0.5, 0.0, 0.5, 0.9, 0.99, 0.999]
LOG_CORR_LEFT   = np.log10(1e-3)   # ρ = 0.999  (best fit, left of axis)
LOG_CORR_RIGHT  = np.log10(2.0)    # ρ = -1.0   (worst fit, right of axis)
LOG_CORR_XTICKS  = [np.log10(1 - r) for r in _LOG_RHO_TICKS]
LOG_CORR_XLABELS = [str(r) for r in _LOG_RHO_TICKS]
LOG_CORR_BINS    = np.linspace(LOG_CORR_LEFT,  LOG_CORR_RIGHT, 50)
LINEAR_CORR_BINS = np.linspace(-1, 1, 50)


def to_log_corr(s):
    """log10(1 − ρ), clipped to avoid −∞ at ρ = 1."""
    return np.log10(1 - np.clip(s, None, 1 - 1e-4))


def plot_linear_corr_hist(ax, data, label, color):
    """Standard linear histogram for a correlation Series."""
    median_r = float(np.nanmedian(data))
    ax.hist(data, bins=LINEAR_CORR_BINS, color=color, alpha=0.80,
            edgecolor='white', linewidth=0.4, zorder=2)
    ax.axvline(median_r, color='black', linestyle='--', linewidth=1.2, zorder=3,
               label=f'Median = {median_r:.3f}')
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(label, labelpad=4)
    ax.set_ylabel('Count', labelpad=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.03, 0.95, f'$n={len(data)}$', transform=ax.transAxes,
            fontsize=8.5, va='top', ha='left', color='#444444')
    ax.legend(fontsize=8, frameon=False)


def plot_log_corr_hist(ax, data, label, color, inset=True):
    """log10(1−ρ) histogram with an optional linear inset in the top-right."""
    median_r = float(np.nanmedian(data))
    ax.hist(to_log_corr(data), bins=LOG_CORR_BINS, color=color, alpha=0.80,
            edgecolor='white', linewidth=0.4, zorder=2)
    ax.axvline(to_log_corr(median_r), color='black', linestyle='--',
               linewidth=1.2, zorder=3, label=f'Median = {median_r:.3f}')
    ax.set_xticks(LOG_CORR_XTICKS)
    ax.set_xticklabels(LOG_CORR_XLABELS, rotation=35, ha='right')
    ax.set_xlim(LOG_CORR_LEFT, LOG_CORR_RIGHT)
    ax.set_xlabel(label, labelpad=4)
    ax.set_ylabel('Count', labelpad=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.03, 0.95, f'$n={len(data)}$', transform=ax.transAxes,
            fontsize=8, va='top', ha='left', color='#444444')
    ax.legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(0.03, 0.88))
    if inset:
        ax_in = ax.inset_axes([0.55, 0.50, 0.42, 0.46])
        ax_in.hist(data, bins=np.linspace(-1, 1, 40), color=color,
                   alpha=0.75, edgecolor='white', linewidth=0.3)
        ax_in.axvline(median_r, color='black', linestyle='--', linewidth=0.9)
        ax_in.set_xlim(-1, 1)
        ax_in.set_xticks([-1, 0, 1])
        ax_in.tick_params(labelsize=6.5)
        ax_in.spines['top'].set_visible(False)
        ax_in.spines['right'].set_visible(False)
        ax_in.set_title('full range', fontsize=6.5, color='#666666', pad=2)


# ── Single-neuron / dataset plots ─────────────────────────────────────────────

def plot_ecdf(ax, vals, label, color):
    sorted_vals = np.sort(vals.dropna())
    y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, y, label=label, color=color, linewidth=1.8)


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


def plot_neuron_comparison(index, original_data_list, upsampled_data_list):
    """Overlay original vs upsampled signal and spike times for one neuron."""
    if index >= len(original_data_list) or index >= len(upsampled_data_list):
        print(f"Index {index} is out of bounds.")
        return

    orig = original_data_list[index]
    ups  = upsampled_data_list[index]

    plt.figure(figsize=(15, 6))
    plt.plot(orig['t'],  orig['dff'], 'o-', color='gray',    alpha=0.5, label='Original Signal',  markersize=4)
    plt.plot(ups['evenly_spaced_time'], ups['upsampled_signal'], '-', color='tab:blue', label='Upsampled Signal', linewidth=2)

    max_val   = max(np.max(orig['dff']), np.max(ups['upsampled_signal']))
    range_val = max_val - min(np.min(orig['dff']), np.min(ups['upsampled_signal']))

    plt.plot(orig['spikes'], [max_val + range_val * 0.05] * len(orig['spikes']),
             '|', color='gray',    label='Original Spikes',  markersize=15, markeredgewidth=2)
    plt.plot(ups['upsampled_spikes'], [max_val + range_val * 0.10] * len(ups['upsampled_spikes']),
             '|', color='tab:red', label='Upsampled Spikes', markersize=15, markeredgewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Signal (DF/F)')
    plt.title(f'Neuron {index}: Original vs Upsampled Signal & Spikes')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_neuron_summary(neuron_idx, neurons, df_measurements, zoom_s=5.0, zoom_start=None,
                        outdir='outputs'):
    """Full analysis summary panel for one neuron (saved to PDF).

    zoom_s      : width of the zoom window in seconds (default 5).
    zoom_start  : start time of the zoom window in seconds.
                  If None, the highest-variance window is chosen automatically.
    outdir      : directory the summary PDF is written to (created if needed).
    """
    n           = neurons[neuron_idx]
    time        = n['evenly_spaced_time']
    calcium     = n['upsampled_signal']
    spike_times = n['original_neuron']['spikes']
    spike_train = n.get('spikes_train', get_spikes_train(spike_times, time))
    rec         = n.get('reconstructed_spikes')
    rec_psd     = n.get('reconstructed_spikes_psd')
    sim_rec     = n.get('sim_calcium_from_reconstructed')
    sim_rec_psd = n.get('sim_calcium_from_rec_psd')
    dataset     = n['original_neuron'].get('dataset_name', '')
    tau_ms      = n.get('tau_ms', np.nan)
    tau_psd_ms  = n.get('tau_psd_ms', np.nan)

    row   = df_measurements.loc[neuron_idx]
    p_tau = row.get('gof_tau_pearson_raw')
    s_tau = row.get('gof_tau_spearman_raw')
    p_psd = row.get('gof_tau_psd_pearson_raw')
    s_psd = row.get('gof_tau_psd_spearman_raw')

    def ols_rescale(original, sim):
        if sim is None: return None
        denom = np.dot(sim, sim)
        return (np.dot(sim, original) / denom) * sim if denom != 0 else sim

    sim_rec_sc     = ols_rescale(calcium, sim_rec)
    sim_rec_psd_sc = ols_rescale(calcium, sim_rec_psd)

    cs_true = np.cumsum(spike_train).astype(float)

    def calib_cs(signal, slope_key):
        if signal is None: return None
        cs    = np.cumsum(signal).astype(float)
        slope = n.get(slope_key)
        return cs / slope if slope else cs

    cs_rec_calib = calib_cs(rec,     'cumsum_slope_raw')
    cs_psd_calib = calib_cs(rec_psd, 'cumsum_slope_raw_psd')

    T  = len(time)
    fs = 1.0 / (time[1] - time[0]) if T > 1 else 200.0

    # Binned rate vectors — use group-optimal bin stored on neuron, fallback 500 ms
    BIN_MS     = n.get('binned_rate_bin_ms', 500.0)
    bin_frames = max(1, int(round(BIN_MS * fs / 1000.0)))
    n_bins     = T // bin_frames
    sl         = n_bins * bin_frames
    true_train  = get_spikes_train(spike_times, time).astype(float)
    true_binned = true_train[:sl].reshape(n_bins, bin_frames).sum(axis=1)
    rec_binned     = np.clip(rec,     0, None)[:sl].reshape(n_bins, bin_frames).sum(axis=1) if rec     is not None else None
    rec_psd_binned = np.clip(rec_psd, 0, None)[:sl].reshape(n_bins, bin_frames).sum(axis=1) if rec_psd is not None else None

    # Zoom window
    zwin = min(int(zoom_s * fs), T)
    if zoom_start is not None:
        z0 = int((zoom_start - time[0]) * fs)
        z0 = max(0, min(z0, T - zwin))
    else:
        step   = max(1, zwin // 4)
        starts = range(0, T - zwin, step)
        best_i = int(np.argmax([np.var(calcium[i:i + zwin]) for i in starts]))
        z0     = list(starts)[best_i]
    zoom_sl  = slice(z0, z0 + zwin)
    zt, zc   = time[zoom_sl], calcium[zoom_sl]

    def _clean(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _fmt(v):
        return f'{v:.3f}' if (v is not None and np.isfinite(v)) else 'N/A'

    fig     = plt.figure(figsize=(14, 11))
    gs_main = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.3, 1.3, 1.6], hspace=0.55)

    panels = [
        (sim_rec_sc,     p_tau, s_tau, tau_ms,     'deriv', rec),
        (sim_rec_psd_sc, p_psd, s_psd, tau_psd_ms, 'PSD',   rec_psd),
    ]
    for ri, (sim_sc, pv, sv, tv, label, rec_) in enumerate(panels):
        gs_row = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[ri],
                                                  height_ratios=[3, 1], hspace=0.08)
        ax_ov  = fig.add_subplot(gs_row[0])
        ax_res = fig.add_subplot(gs_row[1], sharex=ax_ov)

        # Full-signal overlay — True is thicker so overlap is visible even when buried
        ax_ov.plot(time, calcium, color='#4C72B0', lw=2.5, alpha=0.9, label='Calcium')
        if sim_sc is not None:
            ax_ov.plot(time, sim_sc, color='#C44E52', lw=0.9, alpha=0.6, label='Simulated (OLS)')
            res = calcium - sim_sc
            ax_res.plot(time, res, color='#888888', lw=0.5)
            ax_res.axhline(0, color='black', lw=0.7, linestyle='--', alpha=0.5)
            ax_res.fill_between(time, res, 0, alpha=0.2, color='#888888')
            ax_res.set_ylabel('Resid.', fontsize=7)

        # Spike rasters — two bands below the calcium trace
        _ymin, _ymax = calcium.min(), calcium.max()
        _rng     = _ymax - _ymin if _ymax != _ymin else 1.0
        _tick_h  = 0.06 * _rng   # true spike tick height
        _rec_h   = 0.18 * _rng   # reconstructed trace band height
        _gap     = 0.03 * _rng
        # Band positions (from top of calcium downward)
        _tick_top = _ymin - _gap                          # top of true-spike band
        _rec_top  = _tick_top - _tick_h - _gap            # top of rec band
        _rec_base = _rec_top - _rec_h                     # bottom of rec band

        # True spikes: vertical ticks
        _sp_in = spike_times[(spike_times >= time[0]) & (spike_times <= time[-1])]
        ax_ov.vlines(_sp_in, _tick_top - _tick_h, _tick_top,
                     color='#2c7bb6', lw=0.6, alpha=0.8, label='True spikes')

        # Reconstructed: filled trace scaled to fill its own band
        if rec_ is not None:
            _rec_cl = np.clip(rec_, 0, None)
            _mx = _rec_cl.max()
            if _mx > 0:
                _rec_y = _rec_base + _rec_h * (_rec_cl / _mx)
                ax_ov.fill_between(time, _rec_base, _rec_y,
                                   color='#e05c3a', alpha=0.55, label='Rec spikes')
                ax_ov.plot(time, _rec_y, color='#c0392b', lw=0.6, alpha=0.7)

        ax_ov.set_ylim(_rec_base - _gap, _ymax + 0.05 * _rng)

        # Highlight zoom window
        ax_ov.axvspan(time[zoom_sl.start], time[zoom_sl.stop - 1],
                      alpha=0.12, color='green', zorder=0, label='zoom')
        ax_ov.set_title(f'τ = {tv:.0f} ms ({label})   r = {_fmt(pv)}  |  ρ = {_fmt(sv)}', fontsize=9)
        ax_ov.set_ylabel('dF/F')
        ax_ov.legend(fontsize=7, frameon=False, ncol=4)
        ax_res.set_xlabel('Time (s)')
        plt.setp(ax_ov.get_xticklabels(), visible=False)
        _clean(ax_ov); _clean(ax_res)

    # Bottom row: 2×2 scatters (left) + single combined zoom (right)
    gs2     = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[2],
                                               width_ratios=[3, 2], wspace=0.35)
    gs_scat = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs2[0],
                                               hspace=0.55, wspace=0.45)
    ax_scat_tau = fig.add_subplot(gs_scat[0, 0])
    ax_scat_psd = fig.add_subplot(gs_scat[0, 1])
    ax_cs_tau   = fig.add_subplot(gs_scat[1, 0])
    ax_cs_psd   = fig.add_subplot(gs_scat[1, 1])

    # Single zoom panel — both simulations overlaid on the same calcium trace
    ax_zm = fig.add_subplot(gs2[1])
    ax_zm.plot(zt, zc, color='#4C72B0', lw=3.0, alpha=0.9, label='Calcium')
    if sim_rec_sc is not None:
        ax_zm.plot(zt, sim_rec_sc[zoom_sl], color='#C44E52', lw=1.4, alpha=0.8,
                   label='Sim (deriv)')
    if sim_rec_psd_sc is not None:
        ax_zm.plot(zt, sim_rec_psd_sc[zoom_sl], color='#DD8452', lw=1.4, alpha=0.8,
                   linestyle='--', label='Sim (PSD)')
    _zymin, _zymax = zc.min(), zc.max()
    _zrng      = _zymax - _zymin if _zymax != _zymin else 1.0
    _zgap      = 0.03 * _zrng
    _ztick_h   = 0.06 * _zrng
    _ztick_top = _zymin - _zgap
    _sp_zoom = spike_times[(spike_times >= zt[0]) & (spike_times <= zt[-1])]
    ax_zm.vlines(_sp_zoom, _ztick_top - _ztick_h, _ztick_top,
                 color='#2c7bb6', lw=1.2, alpha=0.85, label='True spikes')
    ax_zm.set_ylim(_ztick_top - _ztick_h - _zgap, _zymax + 0.05 * _zrng)
    ax_zm.set_xlabel('Time (s)', fontsize=8)
    ax_zm.set_ylabel('dF/F', fontsize=8)
    ax_zm.set_title(f'Zoom  ({zoom_s:.0f} s window)', fontsize=9)
    ax_zm.legend(fontsize=7, frameon=False)
    _clean(ax_zm)

    def _gof_scatter(ax, sim_sc, sv, title):
        if sim_sc is None:
            ax.set_title(title + '\n[no data]', fontsize=9); return
        ax.scatter(calcium, sim_sc, s=0.5, alpha=0.2, color='#4C72B0', rasterized=True)
        lo, hi = min(calcium.min(), sim_sc.min()), max(calcium.max(), sim_sc.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f'{title}\nρ = {_fmt(sv)}', fontsize=9)
        ax.set_xlabel('True dF/F', fontsize=7)
        ax.set_ylabel('Sim. dF/F', fontsize=7)
        ax.tick_params(labelsize=7)
        _clean(ax)

    _gof_scatter(ax_scat_tau, sim_rec_sc,     s_tau, f'Scatter  τ={tau_ms:.0f} ms (deriv)')
    _gof_scatter(ax_scat_psd, sim_rec_psd_sc, s_psd, f'Scatter  τ={tau_psd_ms:.0f} ms (PSD)')

    def _binned_rate_scatter(ax, true_b, rec_b, title):
        if rec_b is None:
            ax.set_title(title + '\n[no data]', fontsize=9); return
        sv = safe_spearman(true_b, rec_b)
        # Scale rec to match true range for display only (Spearman is rank-invariant)
        rec_mean = rec_b.mean()
        rec_disp = rec_b * (true_b.mean() / rec_mean) if rec_mean > 0 else rec_b
        ax.scatter(true_b, rec_disp, s=6, alpha=0.4, color='#4C72B0',
                   edgecolors='none', rasterized=True)
        hi = max(true_b.max(), rec_disp.max())
        ax.plot([0, hi], [0, hi], 'k--', lw=1, label='y = x')
        ax.set_title(f'{title}  ({BIN_MS:.0f} ms bins)\nρ = {_fmt(sv)}', fontsize=9)
        ax.set_xlabel('True spike count / bin', fontsize=7)
        ax.set_ylabel('Rec (scaled) / bin', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False)
        _clean(ax)

    _binned_rate_scatter(ax_cs_tau, true_binned, rec_binned,     f'Binned rate  (deriv τ)')
    _binned_rate_scatter(ax_cs_psd, true_binned, rec_psd_binned, f'Binned rate  (PSD τ)')

    fig.suptitle(
        f'Full Analysis Summary — Neuron {neuron_idx}  —  {dataset}\n'
        f'τ = {tau_ms:.0f} ms (deriv)    τ = {tau_psd_ms:.0f} ms (PSD)',
        fontsize=13, fontweight='bold', y=1.02)
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f'neuron_{neuron_idx}_summary.pdf'),
                bbox_inches='tight', dpi=300)
    plt.show()


# ── Spike-rate comparison & calibration dashboard ─────────────────────────────

def plot_binned_comparison(true_spikes_train, reconstructed_spikes, bin_size=50, sampling_rate=None, plot=True, ax=None):
    """
    Bins the true and reconstructed spike trains and plots their superimposed summation.
    Calculates Pearson correlation between the binned traces.

    Args:
        true_spikes_train (np.ndarray): Binary spike train (1s and 0s).
        reconstructed_spikes (np.ndarray): Reconstructed continuous spike signal.
        bin_size (int): Size of the bin in samples (data points).
        sampling_rate (float, optional): Sampling rate in Hz. If provided, x-axis will be in seconds.
        plot (bool): Whether to generate the plot.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None and plot is True, creates new figure.

    Returns:
        float: Pearson correlation coefficient.
    """

    # Ensure bin_size is at least 1
    bin_width = int(bin_size)
    if bin_width < 1:
        bin_width = 1

    # Determine the length that is a multiple of bin_width
    n_bins = len(true_spikes_train) // bin_width
    limit = n_bins * bin_width

    # Truncate to limit
    t_spikes_trunc = true_spikes_train[:limit]
    r_spikes_trunc = reconstructed_spikes[:limit]

    # Reshape and sum
    # Shape: (n_bins, bin_width) -> sum along axis 1
    binned_true = t_spikes_trunc.reshape(n_bins, bin_width).sum(axis=1)
    binned_rec = r_spikes_trunc.reshape(n_bins, bin_width).sum(axis=1)

    # Calculate Correlation
    if np.std(binned_true) == 0 or np.std(binned_rec) == 0:
        correlation = np.nan
    else:
        correlation, _ = pearsonr(binned_true, binned_rec)

    if plot:
        if ax is None:
            plt.figure(figsize=(15, 6))
            ax = plt.gca()
            show = True
        else:
            show = False

        # Calculate X-axis
        # If sampling_rate is provided, convert to seconds
        if sampling_rate:
            # Time per bin in seconds
            dt_bin = bin_width / sampling_rate
            time_axis = np.arange(n_bins) * dt_bin
            xlabel = 'Time (s)'
            bin_label_width = dt_bin
            step_offset = dt_bin / 2
        else:
            # Use samples/indices
            time_axis = np.arange(n_bins) * bin_width
            xlabel = 'Time (samples)'
            bin_label_width = bin_width
            step_offset = bin_width / 2

        # Plot True Spikes (Bar)
        ax.bar(time_axis, binned_true, width=bin_label_width, alpha=0.3, color='gray', label='True Spikes', align='edge')

        # Plot Reconstructed Spikes (Step) with visual shift correction
        ax.step(time_axis + step_offset, binned_rec, where='mid', color='#e67e22', label='Reconstructed Spikes')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Summed Activity per Bin')
        ax.set_title(f'Binned Spike Comparison (Bin Size: {bin_width}) | r = {correlation:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

    return correlation


def plot_calibration_dashboard(true_spikes_train, reconstructed_spikes, calibration_results, bin_size=50, sampling_rate=None):
    """
    Plots a 4-panel dashboard using pre-calculated calibration results.

    Args:
        true_spikes_train (np.ndarray): Binary spike train.
        reconstructed_spikes (np.ndarray): Raw reconstructed trace.
        calibration_results (dict): Output from analyze_cumsum_calibration.
        bin_size (int): Bin size in samples.
        sampling_rate (float, optional): Sampling rate in Hz.
    """

    # Extract metrics
    slope_raw = calibration_results['slope_raw']
    r2_raw = calibration_results['r2_raw']
    slope_calib = calibration_results['slope_calib']
    r2_calib = calibration_results['r2_calib']
    calibrated_spikes = calibration_results['calibrated_trace']

    # Prepare Plot Data
    x_cumsum = np.cumsum(true_spikes_train)
    y_cumsum_raw = np.cumsum(reconstructed_spikes)
    y_cumsum_calib = np.cumsum(calibrated_spikes)

    if sampling_rate:
        time_axis = np.arange(len(true_spikes_train)) / sampling_rate
        xlabel_trace = 'Time (s)'
    else:
        time_axis = np.arange(len(true_spikes_train))
        xlabel_trace = 'Time (samples)'

    true_spike_times = time_axis[np.where(true_spikes_train > 0)[0]]

    # Plotting
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    # Panel 1: Binned Comparison (Top)
    ax1 = fig.add_subplot(gs[0, :])
    plot_binned_comparison(
        true_spikes_train,
        calibrated_spikes,
        bin_size=bin_size,
        sampling_rate=sampling_rate,
        plot=True,
        ax=ax1
    )
    ax1.set_title(f"1. Binned Spikes Comparison (Calibrated | Bin Size: {bin_size})")

    # Panel 2: Trace Comparison (Middle)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time_axis, calibrated_spikes, color='black', alpha=0.8, label='Calibrated Rec Trace')

    ymin, ymax = np.min(calibrated_spikes), np.max(calibrated_spikes)
    range_y = ymax - ymin if ymax != ymin else 1.0
    ax2.vlines(true_spike_times, ymin=ymax, ymax=ymax + 0.1*range_y, color='red', alpha=0.6, label='True Spikes')

    ax2.set_xlabel(xlabel_trace)
    ax2.set_ylabel('Amplitude (Calibrated)')
    ax2.set_title("2. Reconstructed Trace vs True Spike Events")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: CumSum Raw (Bottom Left)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(x_cumsum, y_cumsum_raw, color='C0', label='Data')
    if not np.isnan(slope_raw):
        ax3.plot(x_cumsum, slope_raw * x_cumsum, 'k--', alpha=0.7, label=f'Fit (Slope={slope_raw:.4f})')

    ax3.set_title(f"3. Raw: CumSum(Rec) vs CumSum(True)\nR2={r2_raw:.4f}, Slope={slope_raw:.4f}")
    ax3.set_xlabel('CumSum(True Spikes)')
    ax3.set_ylabel('CumSum(Raw Rec)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: CumSum Calibrated (Bottom Right)
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(x_cumsum, y_cumsum_calib, color='C1', label='Data')
    if not np.isnan(slope_calib):
        ax4.plot(x_cumsum, slope_calib * x_cumsum, 'k--', alpha=0.7, label=f'Fit (Slope={slope_calib:.4f})')

    ax4.set_title(f"4. Calibrated: CumSum(Rec) vs CumSum(True)\nR2={r2_calib:.4f}, Slope={slope_calib:.4f}")
    ax4.set_xlabel('CumSum(True Spikes)')
    ax4.set_ylabel('CumSum(Calibrated Rec)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
