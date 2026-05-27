#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sentiment-trend features for Path A (user turns only, no leakage)."""

from snownlp import SnowNLP
import numpy as np

TREND_FEATURE_NAMES = ('pol_t', 'slope_3', 'slope_5', 'slope_7', 'volatility_5')
NUM_TREND_FEATURES = len(TREND_FEATURE_NAMES)


def utterance_polarity(text: str) -> float:
    """SnowNLP polarity mapped to [-1, 1] (DAMI convention)."""
    if not text or not str(text).strip():
        return 0.0
    return float(2 * SnowNLP(str(text)).sentiments - 1)


def sentiment_slope(user_polarities, k: int = 5) -> float:
    """Linear slope over last k user polarities; negative => deteriorating mood."""
    pols = list(user_polarities[-k:])
    if len(pols) < 2:
        return 0.0
    x = np.arange(len(pols), dtype=np.float64)
    slope, _ = np.polyfit(x, pols, 1)
    return float(slope)


def sentiment_volatility(user_polarities, k: int = 5) -> float:
    pols = list(user_polarities[-k:])
    if len(pols) < 2:
        return 0.0
    return float(np.std(pols))


def trend_vector_from_user_history(user_polarities) -> np.ndarray:
    """Per-turn vector stored in pickle: pol_t, slope_3/5/7, volatility_5 (5 cols)."""
    if not user_polarities:
        return np.zeros(NUM_TREND_FEATURES, dtype=np.float32)
    pol_t = float(user_polarities[-1])
    return np.array([
        pol_t,
        sentiment_slope(user_polarities, k=3),
        sentiment_slope(user_polarities, k=5),
        sentiment_slope(user_polarities, k=7),
        sentiment_volatility(user_polarities, k=5),
    ], dtype=np.float32)


def compute_dialogue_trends_from_turns(turns):
    """
    Compute per-turn trend features for a dialogue.

    Each turn dict supports:
      - role: 'c2b' (user) or 'b2c' (agent), or numeric 0=user / 1=agent
      - content: raw text (preferred)
      - snow_sentiment: optional precomputed SnowNLP score in [0, 1]

    User-only history is updated on user turns; agent turns reuse the last
    user-side trend vector (no future leakage).
    """
    user_polarities = []
    per_turn = []

    for turn in turns:
        role = turn.get('role')
        if role in (0, 'c2b', 'user'):
            is_user = True
        elif role in (1, 'b2c', 'agent'):
            is_user = False
        else:
            is_user = str(role).lower() in ('c2b', 'user', '0')

        if is_user:
            if 'content' in turn and turn['content']:
                pol = utterance_polarity(turn['content'])
            elif 'snow_sentiment' in turn:
                pol = float(2 * turn['snow_sentiment'] - 1)
            else:
                pol = 0.0
            user_polarities.append(pol)
            feats = trend_vector_from_user_history(user_polarities)
        else:
            if user_polarities:
                feats = trend_vector_from_user_history(user_polarities)
            else:
                feats = np.zeros(NUM_TREND_FEATURES, dtype=np.float32)
        per_turn.append(feats)

    return per_turn


def compute_dialogue_trends_from_roles_and_senti(roles, snow_sentiments):
    """
    Fallback when raw text is unavailable: use per-turn SnowNLP scores in [0, 1]
    on user (c2b) turns only.
    """
    turns = []
    for role, senti in zip(roles, snow_sentiments):
        turns.append({'role': role, 'snow_sentiment': float(senti)})
    return compute_dialogue_trends_from_turns(turns)
