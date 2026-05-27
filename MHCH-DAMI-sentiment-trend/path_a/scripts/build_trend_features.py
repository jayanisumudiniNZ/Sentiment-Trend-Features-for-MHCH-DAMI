#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Augment MHCH-DAMI/data pickles with trend_list (9th field)."""

import argparse
import json
import os
import pickle as pkl
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from path_a.bootstrap import get_data_dir, init_environment
from path_a.core.trends import (
    NUM_TREND_FEATURES,
    compute_dialogue_trends_from_roles_and_senti,
    compute_dialogue_trends_from_turns,
)

init_environment()


def load_pkl_split(path):
    with open(path, 'rb') as fin:
        objects = []
        while True:
            try:
                objects.append(pkl.load(fin))
            except EOFError:
                break
    keys = [
        'dialogues_ids_list', 'role_list', 'tf_list', 'pos_list', 'senti_list',
        'dialogues_sent_len_list', 'dialogues_len_list', 'label_list',
    ]
    data = dict(zip(keys, objects[:8]))
    data['trend_list'] = objects[8] if len(objects) > 8 else None
    return data


def save_pkl_split(path, data):
    with open(path, 'wb') as fout:
        for key in [
            'dialogues_ids_list', 'role_list', 'tf_list', 'pos_list', 'senti_list',
            'dialogues_sent_len_list', 'dialogues_len_list', 'label_list', 'trend_list',
        ]:
            pkl.dump(data[key], fout)


def load_dialogues_from_json(json_path):
    dialogues = []
    with open(json_path, 'r', encoding='utf-8', errors='ignore') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            try:
                obj = json.loads(line)
                dialogues.append(obj['session'])
            except (json.JSONDecodeError, KeyError):
                continue
    return dialogues


def trends_from_json_dialogues(dialogues):
    trend_list = []
    for dialogue in dialogues:
        turns = [{'role': t['role'], 'content': t['content']} for t in dialogue]
        trend_list.append(compute_dialogue_trends_from_turns(turns))
    return trend_list


def pad_and_scale_trends(trend_list, dialogues_len_list, maxlen=30):
    padded = []
    masks = []
    for trends, dia_len in zip(trend_list, dialogues_len_list):
        row = np.zeros((maxlen, NUM_TREND_FEATURES), dtype=np.float32)
        n = min(int(dia_len), maxlen)
        for t in range(n):
            row[t] = trends[t]
        padded.append(row)
        mask = np.zeros(maxlen, dtype=bool)
        mask[:n] = True
        masks.append(mask)
    stacked = np.stack(padded, axis=0)
    mask_flat = np.stack(masks, axis=0).reshape(-1)
    flat = stacked.reshape(-1, NUM_TREND_FEATURES)
    scaler = StandardScaler()
    scaler.fit(flat[mask_flat])
    scaled_flat = flat.copy()
    scaled_flat[mask_flat] = scaler.transform(flat[mask_flat])
    return scaled_flat.reshape(stacked.shape).astype(np.float32)


def senti_list_is_degenerate(senti_list, dialogues_len_list):
    for sentis, dia_len in zip(senti_list, dialogues_len_list):
        for t in range(int(dia_len)):
            if float(sentis[t]) != 0.0:
                return False
    return True


def build_for_dataset(data_name, use_json=True):
    base = os.path.join(get_data_dir(), data_name)
    split_paths = {
        'train': os.path.join(base, 'train.pkl'),
        'eval': os.path.join(base, 'eval.pkl'),
        'test': os.path.join(base, 'test.pkl'),
    }
    json_paths = {
        'train': os.path.join(base, 'mhch_cloth_train.json') if data_name == 'clothing' else os.path.join(base, 'mhch_makeup_train.json'),
        'eval': os.path.join(base, 'mhch_cloth_eval.json') if data_name == 'clothing' else os.path.join(base, 'mhch_makeup_eval.json'),
        'test': os.path.join(base, 'mhch_cloth_test.json') if data_name == 'clothing' else os.path.join(base, 'mhch_makeup_test.json'),
    }

    for split, pkl_path in split_paths.items():
        print('Processing {} / {} ...'.format(data_name, split))
        data = load_pkl_split(pkl_path)

        if use_json and os.path.exists(json_paths[split]):
            dialogues = load_dialogues_from_json(json_paths[split])
            trend_list = trends_from_json_dialogues(dialogues)
            print('  trends from raw JSON ({})'.format(json_paths[split]))
        else:
            if senti_list_is_degenerate(data['senti_list'], data['dialogues_len_list']):
                print('  WARNING: senti_list is all zeros in {}'.format(pkl_path))
                print('  Add MHCH JSON under {} and re-run.'.format(base))
            trend_list = []
            for roles, sentis, dia_len in zip(data['role_list'], data['senti_list'], data['dialogues_len_list']):
                dl = int(dia_len)
                raw_roles = [int(roles[t]) for t in range(dl)]
                raw_sentis = [float(sentis[t]) for t in range(dl)]
                trend_list.append(
                    compute_dialogue_trends_from_roles_and_senti(raw_roles, raw_sentis))
            print('  trends from pickle role + sentiment')

        data['trend_list'] = pad_and_scale_trends(trend_list, data['dialogues_len_list'])
        save_pkl_split(pkl_path, data)
        print('  wrote {}'.format(pkl_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='clothing', choices=['clothing', 'makeup'])
    parser.add_argument('--no_json', action='store_true')
    args = parser.parse_args()
    build_for_dataset(args.data_name, use_json=not args.no_json)


if __name__ == '__main__':
    main()
