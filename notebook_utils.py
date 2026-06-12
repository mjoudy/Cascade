"""
Notebook-specific helpers for active_neurons_clean.ipynb.

Pure-computation functions (ols_scale, compute_gof, calcium_metrics, etc.)
live in functions/metrics.py and are re-exported here via the wildcard import.

Import with: from notebook_utils import *
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from functions.data_manager import *
from functions.derivative_method import *
from functions.metrics import *
from functions.reconstruction import *

# ── Dataset group definitions ─────────────────────────────────────────────────
#
# Two grouping schemes are provided:
#
#   GROUPS          — original scheme: primary axis is indicator × species.
#                     Use when species context matters or for comparison with
#                     the CASCADE paper's own figures.
#
#   GROUPS_BY_INDICATOR — revised scheme: primary axis is indicator, split by
#                     excitatory vs inhibitory cell type within indicators that
#                     have both. Preferred for τ / kinetics comparisons because
#                     it keeps the grouping axis consistent.
#
# Derived helpers (GROUPS_ORDER, DS_TO_GROUP) are provided for both schemes
# with the suffix _V1 (original) and _V2 (by-indicator).

# ── Scheme 1: indicator × species (original) ─────────────────────────────────

GROUPS = {
    "OGB-1 mouse":             ["DS01-OGB1-m-V1", "DS02-OGB1-2-m-V1"],
    "Synthetic dye zebrafish": ["DS04-OGB1-zf-pDp", "DS05-Cal520-zf-pDp"],
    "GCaMP6f zebrafish":       ["DS06-GCaMP6f-zf-aDp", "DS07-GCaMP6f-zf-dD", "DS08-GCaMP6f-zf-OB"],
    "GCaMP6f mouse":           ["DS09-GCaMP6f-m-V1", "DS10-GCaMP6f-m-V1-neuropil-corrected",
                                "DS11-GCaMP6f-m-V1-neuropil-corrected", "X-DS09-GCaMP6f-m-V1",
                                "X-DS10-GCaMP6f-m-V1"],
    "GCaMP6s mouse":           ["DS12-GCaMP6s-m-V1-neuropil-corrected", "DS13-GCaMP6s-m-V1-neuropil-corrected",
                                "DS14-GCaMP6s-m-V1", "DS15-GCaMP6s-m-V1", "DS16-GCaMP6s-m-V1",
                                "X-DS11-GCaMP6s-m-V1", "X-DS12-GCaMP6s-m-V1"],
    "GCaMP5k mouse":           ["DS17-GCaMP5k-m-V1"],
    "Red indicators mouse":    ["DS18-R-CaMP-m-CA3", "DS19-R-CaMP-m-S1",
                                "DS20-jRCaMP1a-m-V1", "DS21-jGECO1a-m-V1"],
    "SST interneurons":        ["DS22-OGB1-m-SST-V1", "DS25-GCaMP6f-m-SST-V1"],
    "PV interneurons":         ["DS23-OGB1-m-PV-V1", "DS24-GCaMP6f-m-PV-V1", "DS27-GCaMP6f-m-PV-vivo-V1"],
    "VIP interneurons":        ["DS26-GCaMP6f-m-VIP-V1"],
    "NAOMi simulated":         ["X-NAOMi-GCaMP6f-simulated"],
}

GROUP_COLORS = {
    "OGB-1 mouse":             "#4C72B0",
    "Synthetic dye zebrafish": "#64B5CD",
    "GCaMP6f zebrafish":       "#55A868",
    "GCaMP6f mouse":           "#27AE60",
    "GCaMP6s mouse":           "#DD8452",
    "GCaMP5k mouse":           "#E59866",
    "Red indicators mouse":    "#C44E52",
    "SST interneurons":        "#8172B2",
    "PV interneurons":         "#AA4499",
    "VIP interneurons":        "#937860",
    "NAOMi simulated":         "#AAAAAA",
}

GROUPS_ORDER = list(GROUPS.keys())
DS_TO_GROUP  = {ds: grp for grp, dss in GROUPS.items() for ds in dss}

# ── Scheme 2: indicator × cell type (revised) ────────────────────────────────

GROUPS_BY_INDICATOR = {
    "OGB-1 excitatory":   ["DS01-OGB1-m-V1", "DS02-OGB1-2-m-V1", "DS04-OGB1-zf-pDp"],
    "OGB-1 inhibitory":   ["DS22-OGB1-m-SST-V1", "DS23-OGB1-m-PV-V1"],
    "Cal-520":            ["DS05-Cal520-zf-pDp"],
    "GCaMP6f excitatory": ["DS06-GCaMP6f-zf-aDp", "DS07-GCaMP6f-zf-dD", "DS08-GCaMP6f-zf-OB",
                           "DS09-GCaMP6f-m-V1", "DS10-GCaMP6f-m-V1-neuropil-corrected",
                           "DS11-GCaMP6f-m-V1-neuropil-corrected",
                           "X-DS09-GCaMP6f-m-V1", "X-DS10-GCaMP6f-m-V1"],
    "GCaMP6f inhibitory": ["DS24-GCaMP6f-m-PV-V1", "DS25-GCaMP6f-m-SST-V1",
                           "DS26-GCaMP6f-m-VIP-V1", "DS27-GCaMP6f-m-PV-vivo-V1"],
    "GCaMP6s":            ["DS12-GCaMP6s-m-V1-neuropil-corrected", "DS13-GCaMP6s-m-V1-neuropil-corrected",
                           "DS14-GCaMP6s-m-V1", "DS15-GCaMP6s-m-V1", "DS16-GCaMP6s-m-V1",
                           "X-DS11-GCaMP6s-m-V1", "X-DS12-GCaMP6s-m-V1"],
    "GCaMP5k":            ["DS17-GCaMP5k-m-V1"],
    "R-CaMP1.07":         ["DS18-R-CaMP-m-CA3", "DS19-R-CaMP-m-S1"],
    "jRCaMP1a":           ["DS20-jRCaMP1a-m-V1"],
    "jGECO1a":            ["DS21-jGECO1a-m-V1"],
    "NAOMi simulated":    ["X-NAOMi-GCaMP6f-simulated"],
}

GROUP_COLORS_BY_INDICATOR = {
    "OGB-1 excitatory":   "#4C72B0",
    "OGB-1 inhibitory":   "#9BB8D4",
    "Cal-520":            "#64B5CD",
    "GCaMP6f excitatory": "#27AE60",
    "GCaMP6f inhibitory": "#82C9A0",
    "GCaMP6s":            "#DD8452",
    "GCaMP5k":            "#E59866",
    "R-CaMP1.07":         "#C44E52",
    "jRCaMP1a":           "#E07B7B",
    "jGECO1a":            "#A93226",
    "NAOMi simulated":    "#AAAAAA",
}

GROUPS_ORDER_BY_INDICATOR = list(GROUPS_BY_INDICATOR.keys())
DS_TO_GROUP_BY_INDICATOR  = {ds: grp for grp, dss in GROUPS_BY_INDICATOR.items() for ds in dss}

# ── Dataset helpers ───────────────────────────────────────────────────────────

def get_dataset_name(n):
    name = n.get('dataset_name')
    if not name and 'original_neuron' in n:
        name = n['original_neuron'].get('dataset_name')
    return name or 'Unknown'


def build_group_indices(neurons, ds_to_group=None):
    """Return dict: group name → list of indices into `neurons`.

    ds_to_group defaults to DS_TO_GROUP (scheme 1).
    Pass DS_TO_GROUP_BY_INDICATOR for scheme 2.
    """
    if ds_to_group is None:
        ds_to_group = DS_TO_GROUP
    group_indices = defaultdict(list)
    for i, n in enumerate(neurons):
        grp = ds_to_group.get(get_dataset_name(n), 'Unknown')
        group_indices[grp].append(i)
    return group_indices

# ── Correlation histogram constants & helpers ────────────────────────────────
#
# CORR_COLORS: [deriv-τ colour, PSD-τ colour] — used consistently across all
#              correlation figures so the same τ method always gets the same hue.
#
# Usage in notebook cells:
#   plot_linear_corr_hist(ax, data, label, color)   — for spread-out distributions
#   plot_log_corr_hist(ax, data, label, color)       — for distributions near 1

CORR_COLORS = ['#8172B2', '#937860']   # [deriv-τ: purple, PSD-τ: brown]

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


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_ecdf(ax, vals, label, color):
    sorted_vals = np.sort(vals.dropna())
    y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, y, label=label, color=color, linewidth=1.8)


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


def plot_neuron_summary(neuron_idx, neurons, df_measurements, zoom_s=5.0, zoom_start=None):
    """Full analysis summary panel for one neuron (saved to PDF).

    zoom_s      : width of the zoom window in seconds (default 5).
    zoom_start  : start time of the zoom window in seconds.
                  If None, the highest-variance window is chosen automatically.
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

    fig     = plt.figure(figsize=(14, 9))
    gs_main = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.4, 1.4, 1.1], hspace=0.5)

    panels = [
        (sim_rec_sc,     p_tau, s_tau, tau_ms,     'deriv'),
        (sim_rec_psd_sc, p_psd, s_psd, tau_psd_ms, 'PSD'),
    ]
    for ri, (sim_sc, pv, sv, tv, label) in enumerate(panels):
        gs_row  = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[ri],
                                                   width_ratios=[3, 2], wspace=0.18)
        gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_row[0],
                                                   height_ratios=[3, 1], hspace=0.08)
        ax_ov  = fig.add_subplot(gs_left[0])
        ax_res = fig.add_subplot(gs_left[1], sharex=ax_ov)
        ax_zm  = fig.add_subplot(gs_row[1])

        # Full-signal overlay — True is thicker so overlap is visible even when buried
        ax_ov.plot(time, calcium, color='#4C72B0', lw=1.4, alpha=0.9, label='Calcium')
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
        _rec_raw = rec if ri == 0 else rec_psd
        if _rec_raw is not None:
            _rec_cl = np.clip(_rec_raw, 0, None)
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

        # Zoomed panel — same thickness hierarchy + spike rasters
        ax_zm.plot(zt, zc, color='#4C72B0', lw=2.0, alpha=0.9, label='Calcium')
        if sim_sc is not None:
            ax_zm.plot(zt, sim_sc[zoom_sl], color='#C44E52', lw=1.3, alpha=0.7, label='Simulated')
            ax_zm.fill_between(zt, zc, sim_sc[zoom_sl], alpha=0.15, color='#888888')
        _zymin, _zymax = zc.min(), zc.max()
        _zrng    = _zymax - _zymin if _zymax != _zymin else 1.0
        _ztick_h = 0.06 * _zrng
        _zrec_h  = 0.18 * _zrng
        _zgap    = 0.03 * _zrng
        _ztick_top = _zymin - _zgap
        _zrec_top  = _ztick_top - _ztick_h - _zgap
        _zrec_base = _zrec_top - _zrec_h

        _sp_zoom = spike_times[(spike_times >= zt[0]) & (spike_times <= zt[-1])]
        ax_zm.vlines(_sp_zoom, _ztick_top - _ztick_h, _ztick_top,
                     color='#2c7bb6', lw=1.2, alpha=0.85)

        _rec_zoom = (rec if ri == 0 else rec_psd)
        if _rec_zoom is not None:
            _rz = np.clip(_rec_zoom[zoom_sl], 0, None)
            _mx = _rz.max()
            if _mx > 0:
                _rz_y = _zrec_base + _zrec_h * (_rz / _mx)
                ax_zm.fill_between(zt, _zrec_base, _rz_y,
                                   color='#e05c3a', alpha=0.55)
                ax_zm.plot(zt, _rz_y, color='#c0392b', lw=0.8, alpha=0.7)

        ax_zm.set_ylim(_zrec_base - _zgap, _zymax + 0.05 * _zrng)
        ax_zm.set_xlabel('Time (s)', fontsize=8)
        ax_zm.set_ylabel('dF/F', fontsize=8)
        ax_zm.set_title(f'Zoom  ({zoom_s:.0f} s, highest-variance window)', fontsize=9)
        ax_zm.legend(fontsize=7, frameon=False)
        _clean(ax_zm)

    # Bottom row: scatter + cumsum
    gs2         = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[2], wspace=0.45)
    ax_scat_tau = fig.add_subplot(gs2[0])
    ax_scat_psd = fig.add_subplot(gs2[1])
    ax_cs_tau   = fig.add_subplot(gs2[2])
    ax_cs_psd   = fig.add_subplot(gs2[3])

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
    plt.savefig(f'neuron_{neuron_idx}_summary.pdf', bbox_inches='tight', dpi=300)
    plt.show()
