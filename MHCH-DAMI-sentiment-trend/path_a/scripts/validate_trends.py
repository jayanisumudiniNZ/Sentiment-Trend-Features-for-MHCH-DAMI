#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate trend_list in linked MHCH-DAMI/data pickles."""

import argparse
import os
import pickle as pkl
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT)

from path_a.bootstrap import get_data_dir, get_results_dir, init_environment
from path_a.core.trends import TREND_FEATURE_NAMES

init_environment()


def load_pkl_fields(pkl_path):
    with open(pkl_path, 'rb') as fin:
        fields = []
        while True:
            try:
                fields.append(pkl.load(fin))
            except EOFError:
                break
    keys = [
        'dialogues_ids_list', 'role_list', 'tf_list', 'pos_list', 'senti_list',
        'dialogues_sent_len_list', 'dialogues_len_list', 'label_list',
    ]
    data = dict(zip(keys, fields[:8]))
    data['trend_list'] = fields[8] if len(fields) > 8 else None
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='clothing', choices=['clothing', 'makeup'])
    parser.add_argument('--split', default='train', choices=['train', 'eval', 'test'])
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    path = os.path.join(get_data_dir(), args.data_name, args.split + '.pkl')
    data = load_pkl_fields(path)
    trends = data.get('trend_list')
    senti = np.asarray(data['senti_list'])
    dia_lens = data['dialogues_len_list']
    senti_ok = any(float(senti[i][t]) != 0.0 for i, dl in enumerate(dia_lens) for t in range(int(dl)))
    if not senti_ok:
        print('WARNING: senti_list is all zeros in linked MHCH-DAMI data.')
        print('Add JSON under {}/{} and run upstream data_prepare.py'.format(
            get_data_dir(), args.data_name))
    if trends is None:
        raise SystemExit(
            'No trend_list in {}. Run: python -m path_a.scripts.build_trend_features --data_name {}'.format(
                path, args.data_name))

    flat = trends.reshape(-1, trends.shape[-1])
    mask = np.any(flat != 0, axis=1)
    active = flat[mask]
    print('File:', path)
    print('Shape:', trends.shape)
    print('Feature names:', TREND_FEATURE_NAMES)
    for i, name in enumerate(TREND_FEATURE_NAMES):
        col = active[:, i]
        print('  {}: min={:.4f} max={:.4f} mean={:.4f} std={:.4f}'.format(
            name, col.min(), col.max(), col.mean(), col.std()))
    print('Validation OK.')


if __name__ == '__main__':
    main()
