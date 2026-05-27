#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run full ablation: train + evaluate BERT models across all trend feature variants.

Usage:
  python run_ablation.py --data_name clothing --split test
  python run_ablation.py --data_name makeup --split test --bert_only
  python run_ablation.py --data_name clothing --split test --flags_only
"""

import argparse
import json
import os
import subprocess
import sys


TREND_MODES = {
    'pol_only': '.bert.pol',
    'slope3_only': '.bert.slope3',
    'slope5_only': '.bert.slope5',
    'slope7_only': '.bert.slope7',
    'volatility5_only': '.bert.vol5',
    'full': '.bert.full',
}


def run_command(cmd, description):
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"  [FAILED] {description}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser('BERT Ablation Runner')
    parser.add_argument('--data_name', default='clothing', choices=['clothing', 'makeup'])
    parser.add_argument('--split', default='test', choices=['train', 'eval', 'test'])
    parser.add_argument('--bert_only', action='store_true', help='Only run BERT standalone')
    parser.add_argument('--flags_only', action='store_true', help='Only run BERT+flags variants')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, only evaluate')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    python = sys.executable
    results_summary = []

    # 1. BERT standalone
    if not args.flags_only:
        suffix = '.bert'
        if not args.eval_only:
            run_command([
                python, 'train.py',
                '--data_name', args.data_name,
                '--model_type', 'bert_only',
                '--suffix', suffix,
                '--epochs', str(args.epochs),
                '--batch_size', str(args.batch_size),
            ] + (['--device', args.device] if args.device else []),
                f"Training BERT-only on {args.data_name}"
            )

        run_command([
            python, 'evaluate.py',
            '--data_name', args.data_name,
            '--model_type', 'bert_only',
            '--suffix', suffix,
            '--split', args.split,
        ] + (['--device', args.device] if args.device else []),
            f"Evaluating BERT-only on {args.data_name}/{args.split}"
        )

    # 2. BERT + flag features (all variants)
    if not args.bert_only:
        for trend_mode, suffix in TREND_MODES.items():
            if not args.eval_only:
                run_command([
                    python, 'train.py',
                    '--data_name', args.data_name,
                    '--model_type', 'bert_flags',
                    '--trend_features', trend_mode,
                    '--suffix', suffix,
                    '--epochs', str(args.epochs),
                    '--batch_size', str(args.batch_size),
                ] + (['--device', args.device] if args.device else []),
                    f"Training BERT+flags ({trend_mode}) on {args.data_name}"
                )

            run_command([
                python, 'evaluate.py',
                '--data_name', args.data_name,
                '--model_type', 'bert_flags',
                '--trend_features', trend_mode,
                '--suffix', suffix,
                '--split', args.split,
            ] + (['--device', args.device] if args.device else []),
                f"Evaluating BERT+flags ({trend_mode}) on {args.data_name}/{args.split}"
            )

    # 3. Collect results summary
    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY — {args.data_name} / {args.split}")
    print(f"{'='*70}")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', args.data_name)

    header = f"{'Model':<30} {'F1':>6} {'Macro-F1':>9} {'AUC':>6} {'GT-I':>6} {'GT-II':>7} {'GT-III':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    # BERT only
    bert_only_json = os.path.join(results_dir, 'bert_only', 'baseline', args.split, f'metrics_{args.split}.json')
    if os.path.exists(bert_only_json):
        with open(bert_only_json) as f:
            m = json.load(f)
        row = f"{'BERT-only':<30} {m['f1']*100:>6.2f} {m['macro_f1']*100:>9.2f} {m['auc']*100:>6.2f} {m.get('gt1',0)*100:>6.2f} {m.get('gt2',0)*100:>7.2f} {m.get('gt3',0)*100:>7.2f}"
        print(row)
        results_summary.append({'model': 'bert_only', 'trend': 'baseline', **m})

    # BERT + flags
    for trend_mode in TREND_MODES:
        json_path = os.path.join(results_dir, 'bert_flags', trend_mode, args.split, f'metrics_{args.split}.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                m = json.load(f)
            row = f"{'BERT+flags(' + trend_mode + ')':<30} {m['f1']*100:>6.2f} {m['macro_f1']*100:>9.2f} {m['auc']*100:>6.2f} {m.get('gt1',0)*100:>6.2f} {m.get('gt2',0)*100:>7.2f} {m.get('gt3',0)*100:>7.2f}"
            print(row)
            results_summary.append({'model': 'bert_flags', 'trend': trend_mode, **m})

    summary_path = os.path.join(results_dir, f'ablation_summary_{args.split}.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == '__main__':
    main()
