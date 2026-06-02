#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch-evaluate Path A checkpoints; writes results/path_a_ablation_*.md"""

import argparse
import json
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT)

from path_a.core.trend_features import DEFAULT_SUFFIX_BY_MODE, list_trend_modes

PATH_A_VARIANTS = [
    ('baseline', DEFAULT_SUFFIX_BY_MODE['baseline']),
    ('full', DEFAULT_SUFFIX_BY_MODE['full']),
    ('full_slope3', DEFAULT_SUFFIX_BY_MODE['full_slope3']),
    ('pol_only', DEFAULT_SUFFIX_BY_MODE['pol_only']),
    ('slope3_only', DEFAULT_SUFFIX_BY_MODE['slope3_only']),
    ('slope5_only', DEFAULT_SUFFIX_BY_MODE['slope5_only']),
    ('slope7_only', DEFAULT_SUFFIX_BY_MODE['slope7_only']),
    ('volatility5_only', DEFAULT_SUFFIX_BY_MODE['volatility5_only']),
]

PUBLISHED_CLOTHING = {'baseline': {'f1': 67.3, 'gt1': 70.3, 'gt2': 79.1, 'gt3': 83.9}}


def weight_dir(data_name, suffix):
    return os.path.join(_ROOT, 'weights', data_name, 'dami' + suffix + 'train')


def has_checkpoint(data_name, suffix, tag='best'):
    d = weight_dir(data_name, suffix)
    return os.path.isdir(d) and os.path.isfile(os.path.join(d, 'checkpoint'))


def run_evaluate(data_name, suffix, trend_features, split, checkpoint):
    cmd = [
        sys.executable, os.path.join(_ROOT, 'evaluate.py'),
        '--data_name', data_name, '--split', split, '--suffix', suffix,
        '--checkpoint', checkpoint, '--trend_features', trend_features,
    ]
    print('>>', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=_ROOT)


def load_metrics(data_name, trend_features, split):
    path = os.path.join(_ROOT, 'results', 'path_a', data_name, trend_features, split,
                        'metrics_{}.json'.format(split))
    if not os.path.isfile(path):
        return None
    with open(path) as fin:
        return json.load(fin)


def write_report(data_name, split, rows, out_path):
    from path_a.core.trend_features import MODE_FEATURE_NAMES, resolve_trend_mode

    lines = [
        '# Path A ablation — {}'.format(data_name),
        '', 'Split: **{}**'.format(split), '',
        '| Variant | Features | F1 | Macro-F1 | GT-I | GT-II | GT-III | checkpoint |',
        '| ------- | -------- | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for row in rows:
        mode = row['variant']
        feat = 'senti' if mode == 'baseline' else ', '.join(MODE_FEATURE_NAMES[resolve_trend_mode(mode)])
        if row.get('metrics'):
            m = row['metrics']
            lines.append('| {} | {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {} |'.format(
                mode, feat, m['f1'] * 100, m['macro_f1'] * 100, m['gt1'] * 100,
                m['gt2'] * 100, m['gt3'] * 100, row.get('checkpoint', '—')))
        else:
            lines.append('| {} | {} | — | — | — | — | — | not trained |'.format(mode, feat))
    if data_name == 'clothing':
        p = PUBLISHED_CLOTHING['baseline']
        lines.append('\n**Published DAMI (Clothing):** F1={f1}, GT-I={gt1}, GT-II={gt2}, GT-III={gt3}'.format(**p))
    lines.append('')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fout:
        fout.write('\n'.join(lines))
    print('Wrote', out_path)


def main():
    parser = argparse.ArgumentParser(epilog=list_trend_modes())
    parser.add_argument('--data_name', default='clothing', choices=['clothing', 'makeup'])
    parser.add_argument('--split', default='test')
    parser.add_argument('--checkpoint', default='best')
    parser.add_argument('--variants', nargs='*')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--list_modes', action='store_true')
    args = parser.parse_args()
    if args.list_modes:
        print(list_trend_modes())
        return

    variants = PATH_A_VARIANTS
    if args.variants:
        allowed = set(args.variants)
        variants = [v for v in PATH_A_VARIANTS if v[0] in allowed]

    rows = []
    for trend_features, suffix in variants:
        row = {'variant': trend_features, 'suffix': suffix}
        if not has_checkpoint(args.data_name, suffix, args.checkpoint):
            print('SKIP:', weight_dir(args.data_name, suffix))
            rows.append(row)
            continue
        row['checkpoint'] = args.checkpoint
        if not args.dry_run:
            run_evaluate(args.data_name, suffix, trend_features, args.split, args.checkpoint)
            row['metrics'] = load_metrics(args.data_name, trend_features, args.split)
        rows.append(row)

    out = os.path.join(_ROOT, 'results', 'path_a_ablation_{}_{}.md'.format(args.data_name, args.split))
    write_report(args.data_name, args.split, rows, out)


if __name__ == '__main__':
    main()
