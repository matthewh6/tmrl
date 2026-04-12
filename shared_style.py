"""
shared_style.py — single source of truth for all result figures.

Import at the top of every plotting script:

    import shared_style
    import matplotlib
    matplotlib.rcParams.update(shared_style.RC)

Then look up colors with:
    color = shared_style.COLORS['TMRL']        # full saturation (RL curves)
    color = shared_style.COLORS_MUTED['CSP']   # desaturated (Success@K)
    colors = shared_style.CSP_SWEEP            # 5-step gradient for sweep figure

Method name display labels:
    label = shared_style.LABELS['TMRL']        # -> 'TMRL'
    label = shared_style.LABELS['BC']          # -> 'BC'
"""

# ---------------------------------------------------------------------------
# Primary palette — full saturation, used in RL learning curve figures
# Derived from: sns.husl_palette(n_colors=6, l=.75, h=.35)
# PostBC+DSRL uses gray in the RL figure (hardcoded, not from HUSL palette)
# BC and CSP are aliases — same hue as DSRL and TMRL respectively
# ---------------------------------------------------------------------------
COLORS = {
    'TMRL':        '#4ed03b',
    'RLPD':        '#40cac2',
    'DSRL':        '#85bbf7',
    'SPiRL':       '#f295f7',
    'RND':         '#f99da7',
    'PostBC':      '#d8b33c',   # amber — PostBC pre-training alone
    'PostBC+DSRL': '#909090',   # gray  — as shown in sim_results figure
    'BC':          '#85bbf7',   # same hue as DSRL
    'CSP':         '#4ed03b',   # same hue as TMRL
}

# ---------------------------------------------------------------------------
# Muted palette — 50% desaturated, 85% value
# Used in pre-training figures (Success@K) — same hues, quieter tone
# PostBC in Success@K = muted version of PostBC+DSRL gray (it IS PostBC+DSRL
# pre-training, evaluated before RL fine-tuning)
# ---------------------------------------------------------------------------
COLORS_MUTED = {
    'TMRL':        '#7ab171',
    'RLPD':        '#71aca8',
    'DSRL':        '#a2b8d2',
    'SPiRL':       '#d0a8d2',
    'RND':         '#d4adb1',
    'PostBC':      '#b0b0b0',   # keep amber for standalone PostBC
    'PostBC+DSRL': '#b0b0b0',   # muted gray — matches RL figure baseline
    'BC':          '#a2b8d2',   # same hue as DSRL muted
    'CSP':         '#7ab171',   # same hue as TMRL muted
}

# ---------------------------------------------------------------------------
# CSP context-noise sweep gradient
# 5 steps, same hue as TMRL/CSP, muted, light -> dark as sigma increases
# ---------------------------------------------------------------------------
CSP_SWEEP = [
    '#a7bfa4',   # sigma=0.00 — lightest
    '#8fb789',   # sigma=0.25
    '#75ab6d',   # sigma=0.50 — anchored to COLORS_MUTED['CSP'] region
    '#599650',   # sigma=0.75
    '#3e7d35',   # sigma=1.00 — darkest
]

# ---------------------------------------------------------------------------
# Linestyles — consistent across all figures
# ---------------------------------------------------------------------------
LINESTYLES = {
    'TMRL':        '-',
    'CSP':         '-',
    'DSRL':        '--',
    'BC':          '--',
    'PostBC':      '-.',
    'PostBC': '-.',
    'RLPD':        ':',
    'RND':         (0, (3, 1, 1, 1)),
    'SPiRL':       (0, (5, 2)),
}

# ---------------------------------------------------------------------------
# Markers — used in Success@K and other point-annotated figures
# ---------------------------------------------------------------------------
MARKERS = {
    'TMRL':        'D',
    'CSP':         'D',
    'DSRL':        'o',
    'BC':          'o',
    'PostBC':      's',
    'PostBC+DSRL': 's',
    'RLPD':        '^',
    'RND':         'v',
    'SPiRL':       'P',
}

# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------
LABELS = {
    'TMRL':        'TMRL',
    'CSP':         'CSP',
    'DSRL':        'DSRL',
    'BC':          'BC',
    'PostBC':      'PostBC',
    'PostBC+DSRL': 'PostBC+DSRL',
    'RLPD':        'RLPD',
    'RND':         'RND',
    'SPiRL':       'SPiRL',
}

# ---------------------------------------------------------------------------
# Task display names — consistent short names for subplot titles and captions
# ---------------------------------------------------------------------------
TASK_LABELS = {
    'pointmaze-giant-navigate-v0': 'pointmaze',
    'pointmaze_giant':             'pointmaze',
    'cube-single-play-v0':         'cube-single',
    'cube_single':                 'cube-single',
    'libero_goal':                 'libero-goal',
    'libero_90':                   'libero-90',
    'libero-goal-swap':            'libero-goal',
    'dexterous-grasping':          'dex-grasp',
}

# ---------------------------------------------------------------------------
# Figure sizing — standard widths for consistent layout in two-column paper
# Single subplot:          3.2 x 2.4 in
# Two subplots side by side: 6.5 x 2.4 in
# Four subplots (2x2):     6.5 x 4.8 in
# Full-width single:       6.5 x 2.8 in
# ---------------------------------------------------------------------------
FIGSIZE_SINGLE     = (3.2, 2.4)
FIGSIZE_TWO        = (6.5, 2.4)
FIGSIZE_FOUR       = (6.5, 4.8)
FIGSIZE_FULL       = (6.5, 2.8)

# ---------------------------------------------------------------------------
# rcParams — apply to every figure with:
#   import matplotlib; matplotlib.rcParams.update(shared_style.RC)
# ---------------------------------------------------------------------------
RC = {
    # Font
    "font.family":        "palatino",
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    # Legend
    "legend.fontsize":    8,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#dddddd",
    "legend.borderpad":   0.4,
    "legend.labelspacing":0.25,
    "legend.handlelength":1.6,
    "legend.handletextpad":0.4,
    # Axes
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    # Grid
    "axes.grid":          True,
    "grid.alpha":         0.18,
    "grid.linewidth":     0.5,
    # Lines
    "lines.linewidth":    1.8,
    "lines.markersize":   5,
    # Output
    "pdf.fonttype":       42,   # embed fonts for camera-ready
    "ps.fonttype":        42,
}