"""
Dataset grouping schemes and helpers.

Two grouping schemes are provided:

  GROUPS              — original scheme: primary axis is indicator × species.
                        Use when species context matters or for comparison with
                        the CASCADE paper's own figures.

  GROUPS_BY_INDICATOR — revised scheme: primary axis is indicator, split by
                        excitatory vs inhibitory cell type within indicators that
                        have both. Preferred for τ / kinetics comparisons because
                        it keeps the grouping axis consistent.

Derived helpers (GROUPS_ORDER, DS_TO_GROUP) are provided for both schemes,
with the by-indicator variants suffixed _BY_INDICATOR.

Split out of notebook_utils.py during the repository cleanup.
"""

from collections import defaultdict

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
