#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trend feature flags for Path A DAMI ablations.

Stored in pickle trend_list (per turn): [pol_t, slope_3, slope_5, slope_7, volatility_5]
— see trends.TREND_FEATURE_NAMES.

Training flag (--trend_features) selects which columns are fed into input_x3.
"""

from path_a.core.trends import NUM_TREND_FEATURES, TREND_FEATURE_NAMES

# Index in trend_list [..., 5]
FEATURE_INDEX = {name: i for i, name in enumerate(TREND_FEATURE_NAMES)}

# Aliases for CLI / config (all lowercased after resolve)
MODE_ALIASES = {
    'volatility_5_only': 'volatility5_only',
    'vol_only': 'volatility5_only',
    'slope_3_only': 'slope3_only',
    'slope_7_only': 'slope7_only',
    'trend_full': 'full',
    'bundle': 'full',
}

# Each mode -> feature names taken from trend_list (baseline uses senti_list instead)
MODE_FEATURE_NAMES = {
    'baseline': (),
    'none': (),
    'pol_only': ('pol_t',),
    'slope3_only': ('slope_3',),
    'slope5_only': ('slope_5',),
    'slope7_only': ('slope_7',),
    'volatility5_only': ('volatility_5',),
    # Full Path A bundle: polarity + 5-turn slope + volatility (not slope_3/7)
    'full': ('pol_t', 'slope_5', 'volatility_5'),
}

TREND_MODES = tuple(MODE_FEATURE_NAMES.keys())

# Suggested checkpoint suffix per mode (for run_path_a_ablation / training)
DEFAULT_SUFFIX_BY_MODE = {
    'baseline': '.128',
    'pol_only': '.128.pol',
    'slope3_only': '.128.slope3',
    'slope5_only': '.128.slope5',
    'slope7_only': '.128.slope7',
    'volatility5_only': '.128.vol5',
    'full': '.128.trend',
}


def resolve_trend_mode(trend_features):
    mode = (trend_features or 'baseline').lower().strip()
    mode = MODE_ALIASES.get(mode, mode)
    if mode in ('none',):
        return 'baseline'
    if mode not in MODE_FEATURE_NAMES:
        raise ValueError(
            'trend_features must be one of: {}'.format(', '.join(sorted(TREND_MODES))))
    return mode


def feature_indices_for_mode(trend_features):
    mode = resolve_trend_mode(trend_features)
    return [FEATURE_INDEX[name] for name in MODE_FEATURE_NAMES[mode]]


def sentiment_input_dim(trend_features):
    mode = resolve_trend_mode(trend_features)
    if mode == 'baseline':
        return 1
    return len(MODE_FEATURE_NAMES[mode])


def select_sentiment_batch(senti_batch, trend_batch, trend_features):
    """Return [B, D, sentiment_dim] for DAMI input_x3."""
    import numpy as np

    mode = resolve_trend_mode(trend_features)
    if mode == 'baseline':
        senti = np.asarray(senti_batch, dtype=np.float32)
        if senti.ndim == 2:
            return senti[..., np.newaxis]
        return senti

    if trend_batch is None:
        raise ValueError(
            'trend_list missing in pickle. Run: python build_trend_features.py --data_name <name>')

    trend = np.asarray(trend_batch, dtype=np.float32)
    cols = feature_indices_for_mode(mode)
    return trend[..., cols]


def list_trend_modes():
    """Human-readable summary of flags (for docs / --help)."""
    lines = ['Available --trend_features flags:']
    for mode in sorted(TREND_MODES):
        if mode in ('none',):
            continue
        names = MODE_FEATURE_NAMES[mode]
        dim = 1 if mode == 'baseline' else len(names)
        feat = 'senti (baseline)' if mode == 'baseline' else ', '.join(names)
        suffix = DEFAULT_SUFFIX_BY_MODE.get(mode, '')
        lines.append('  {:20s} dim={}  features=[{}]  suffix={}'.format(
            mode, dim, feat, suffix))
    return '\n'.join(lines)
