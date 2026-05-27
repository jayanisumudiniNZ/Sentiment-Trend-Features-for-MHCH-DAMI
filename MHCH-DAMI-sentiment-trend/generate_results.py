#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parse DAMI Path A training log and generate training_history + training_curves.

Usage:
    python generate_results.py --data_name clothing --trend_features pol_only --suffix .128.pol
    python generate_results.py --log_path networks/logs/dami.clothing.train.128.pol.train.dami.log
"""

import os
import re
import json
import glob
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def find_latest_log(data_name, suffix, log_dir='networks/logs'):
    suffix_clean = suffix.lstrip('.')
    pattern = os.path.join(log_dir, f'dami.{data_name}.train.{suffix_clean}.train.dami*.log')
    logs = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not logs:
        pattern2 = os.path.join(log_dir, f'dami.{data_name}.train*{suffix_clean}*.log')
        logs = sorted(glob.glob(pattern2), key=os.path.getmtime)
    if not logs:
        raise FileNotFoundError(
            f'No log files matching pattern in {log_dir} for data={data_name}, suffix={suffix}')
    return logs[-1]


def parse_log(log_path):
    epoch_re = re.compile(
        r'Training the model for epoch (\d+) with batch size (\d+)')
    metrics_re = re.compile(
        r'Handoff (\w+): Loss:([\d.]+)\tAcc:([\d.]+)\tF1Score:([\d.]+)'
        r'\tMacro_F1Score:([\d.]+)\tAUC:([\d.]+)'
        r'\tGT-I:([\d.]+)\tGT-II:([\d.]+)\tGT-III:([\d.]+)')
    test_line_re = re.compile(
        r'Metrics test\t([\d.]+)\t([\d.]+)\t([\d.]+)\t([\d.]+)\t([\d.]+)\t([\d.]+)')
    lambda_re = re.compile(r'Lambda=([\d.\-]+)\t([\d.]+)\t([\d.]+)\t([\d.]+)')
    confusion_re = re.compile(
        r'\[\[\s*(\d+)\s+(\d+)\]\s*\n?\s*\[\s*(\d+)\s+(\d+)\]\]')

    train_history = []
    eval_history = []
    test_metrics = None
    lambda_results = []
    confusion_matrix = None
    classification_report_lines = []
    current_epoch = -1

    with open(log_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    for line in lines:
        m = epoch_re.search(line)
        if m:
            current_epoch = int(m.group(1))

        m = metrics_re.search(line)
        if m:
            phase = m.group(1)
            entry = {
                'epoch': current_epoch,
                'loss': float(m.group(2)),
                'acc': float(m.group(3)),
                'f1': float(m.group(4)),
                'macro_f1': float(m.group(5)),
                'auc': float(m.group(6)),
                'gt1': float(m.group(7)),
                'gt2': float(m.group(8)),
                'gt3': float(m.group(9)),
            }
            if phase == 'train':
                train_history.append(entry)
            elif phase == 'eval':
                eval_history.append(entry)
            elif phase == 'test':
                test_metrics = entry

        m = test_line_re.search(line)
        if m:
            pass  # captured via Handoff test above

        m = lambda_re.search(line)
        if m:
            lambda_results.append({
                'lambda': float(m.group(1)),
                'gt1': float(m.group(2)),
                'gt2': float(m.group(3)),
                'gt3': float(m.group(4)),
            })

    # Extract classification report
    in_report = False
    for line in lines:
        stripped = line.split(' - ', 1)[-1] if ' - ' in line else line
        if 'precision    recall  f1-score' in stripped:
            in_report = True
            classification_report_lines = [stripped]
        elif in_report:
            if stripped.strip() == '' and len(classification_report_lines) > 5:
                in_report = False
            else:
                classification_report_lines.append(stripped)

    # Extract confusion matrix
    cm_idx = content.rfind('[[')
    if cm_idx >= 0:
        cm_snippet = content[cm_idx:cm_idx + 80]
        cm_match = re.search(
            r'\[\[\s*(\d+)\s+(\d+)\]\s*\n?\s*\[\s*(\d+)\s+(\d+)\]\]', cm_snippet)
        if cm_match:
            confusion_matrix = np.array([
                [int(cm_match.group(1)), int(cm_match.group(2))],
                [int(cm_match.group(3)), int(cm_match.group(4))]
            ])

    return {
        'train_history': train_history,
        'eval_history': eval_history,
        'test_metrics': test_metrics,
        'classification_report': '\n'.join(classification_report_lines),
        'lambda_results': lambda_results,
        'confusion_matrix': confusion_matrix,
    }


def save_training_history_csv(train_history, eval_history, output_path):
    with open(output_path, 'w') as f:
        f.write('epoch,phase,loss,acc,f1,macro_f1,auc,gt1,gt2,gt3\n')
        for e in train_history:
            f.write(f"{e['epoch']},train,{e['loss']:.4f},{e['acc']:.4f},{e['f1']:.4f},"
                    f"{e['macro_f1']:.4f},{e['auc']:.4f},{e['gt1']:.4f},{e['gt2']:.4f},{e['gt3']:.4f}\n")
        for e in eval_history:
            f.write(f"{e['epoch']},eval,{e['loss']:.4f},{e['acc']:.4f},{e['f1']:.4f},"
                    f"{e['macro_f1']:.4f},{e['auc']:.4f},{e['gt1']:.4f},{e['gt2']:.4f},{e['gt3']:.4f}\n")


def plot_training_curves(train_history, eval_history, title_suffix, output_path):
    if not HAS_MPL:
        print('matplotlib not available, skipping training curves')
        return

    epochs = [e['epoch'] for e in train_history]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f'DAMI Path A — Training Curves ({title_suffix})', fontsize=15, fontweight='bold')

    metric_pairs = [
        ('loss', 'Loss'),
        ('f1', 'F1-Score'),
        ('macro_f1', 'Macro-F1'),
        ('auc', 'AUC'),
        ('gt1', 'GT-I'),
        ('gt2', 'GT-II'),
    ]

    for ax, (key, title) in zip(axes.flat, metric_pairs):
        train_vals = [e[key] for e in train_history]
        eval_vals = [e[key] for e in eval_history]
        ax.plot(epochs[:len(train_vals)], train_vals, 'b-o', markersize=3, label='Train', linewidth=1.5)
        ax.plot(epochs[:len(eval_vals)], eval_vals, 'r-s', markersize=3, label='Eval', linewidth=1.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_lambda_analysis(lambda_results, output_path):
    if not HAS_MPL or not lambda_results:
        return

    lambdas = [r['lambda'] for r in lambda_results]
    gt1 = [r['gt1'] * 100 if r['gt1'] <= 1.0 else r['gt1'] for r in lambda_results]
    gt2 = [r['gt2'] * 100 if r['gt2'] <= 1.0 else r['gt2'] for r in lambda_results]
    gt3 = [r['gt3'] * 100 if r['gt3'] <= 1.0 else r['gt3'] for r in lambda_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, gt1, 'o-', label='GT-I', linewidth=2, markersize=6)
    ax.plot(lambdas, gt2, 's-', label='GT-II', linewidth=2, markersize=6)
    ax.plot(lambdas, gt3, '^-', label='GT-III', linewidth=2, markersize=6)
    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('GT Scores vs Lambda — Test Set', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate training history and curves from DAMI Path A log')
    parser.add_argument('--data_name', default='clothing')
    parser.add_argument('--trend_features', default='full',
                        help='Feature set name (pol_only, full, slope3_only, etc.)')
    parser.add_argument('--suffix', default='.128.trend',
                        help='Model suffix used during training')
    parser.add_argument('--log_path', default=None,
                        help='Explicit log path; if omitted, searches for latest')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    if args.log_path:
        log_path = args.log_path
    else:
        log_path = find_latest_log(args.data_name, args.suffix)
    print(f'Parsing log: {log_path}')

    output_dir = args.output_dir or os.path.join(
        'results', 'path_a', args.data_name, args.trend_features, 'test')
    os.makedirs(output_dir, exist_ok=True)

    parsed = parse_log(log_path)

    if not parsed['train_history']:
        print('WARNING: No training history found in log.')
        print('Make sure the Tensorflow logger is writing to the log file.')
        print('(The fix in train.py adds tf_logger handlers — retrain to capture history.)')
        return

    print(f'  Found {len(parsed["train_history"])} train epochs, '
          f'{len(parsed["eval_history"])} eval epochs')

    # 1. training_history.csv
    csv_path = os.path.join(output_dir, 'training_history.csv')
    save_training_history_csv(parsed['train_history'], parsed['eval_history'], csv_path)
    print(f'  Saved: {csv_path}')

    # 2. training_curves.png
    title_suffix = args.trend_features
    curves_path = os.path.join(output_dir, 'training_curves.png')
    plot_training_curves(parsed['train_history'], parsed['eval_history'],
                         title_suffix, curves_path)
    print(f'  Saved: {curves_path}')

    # 3. lambda_analysis.png (from log, if present)
    if parsed['lambda_results']:
        lambda_path = os.path.join(output_dir, 'lambda_analysis.png')
        plot_lambda_analysis(parsed['lambda_results'], lambda_path)
        print(f'  Saved: {lambda_path}')

        lambda_json = os.path.join(output_dir, 'lambda_analysis.json')
        with open(lambda_json, 'w') as f:
            json.dump(parsed['lambda_results'], f, indent=2)
        print(f'  Saved: {lambda_json}')

    # 4. test_metrics summary (if found in log)
    if parsed['test_metrics']:
        tm = parsed['test_metrics']
        print(f"\n=== Test Results (from log) ===")
        print(f"  F1:       {tm['f1']*100:.2f}%")
        print(f"  Macro-F1: {tm['macro_f1']*100:.2f}%")
        print(f"  AUC:      {tm['auc']*100:.2f}%")
        print(f"  GT-I:     {tm['gt1']*100:.2f}%")
        print(f"  GT-II:    {tm['gt2']*100:.2f}%")
        print(f"  GT-III:   {tm['gt3']*100:.2f}%")

    print(f'\nAll results saved to: {output_dir}/')
    print('\nTIP: For classification_report, confusion_matrix, roc_curve, and metrics,')
    print('     run evaluate.py which loads the checkpoint and generates those artefacts.')


if __name__ == '__main__':
    main()
