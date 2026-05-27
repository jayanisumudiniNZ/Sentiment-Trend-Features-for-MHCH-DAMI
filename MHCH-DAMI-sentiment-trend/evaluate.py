#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate Path A DAMI checkpoints and generate all result artefacts."""

import argparse
import json
import logging
import os
import pickle as pkl
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc, classification_report, confusion_matrix, roc_curve
)

warnings.filterwarnings('ignore', category=FutureWarning)

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from path_a.bootstrap import (
    get_data_dir,
    get_extension_config_dir,
    get_mhch_config_dir,
    get_results_dir,
    init_environment,
    patch_network_artifact_dirs,
)
from path_a.integrations.dami import DAMI
from path_a.integrations.data_loader import Data_loader

init_environment()

import tf_compat as tf
from utility import get_a_p_r_f_sara, get_gtt_score

CLASS_NAMES = ['Normal', 'Transferable']


def collect_predictions(network, data_generator, data_name, mode, batch_size, nb_classes):
    total_labels, total_predictions = [], []
    total_labels_flat = np.array([])
    total_predictions_flat = np.array([])
    total_scores_flat = np.array([])

    for x1, x2, x3, y, sent_len, dia_len, tfs, pos_list in data_generator(
            data_name=data_name, mode=mode, batch_size=batch_size,
            nb_classes=nb_classes, shuffle=False, epoch=0):
        feed_dict = {
            network.input_x1: x1, network.input_x2: x2, network.input_x3: x3,
            network.tfs: tfs, network.pos_list: pos_list, network.input_y: y,
            network.sent_len: sent_len, network.dia_len: dia_len,
            network.dropout_keep_prob: 1.0,
        }
        sequence, scores = network.sess.run([network.output, network.proba], feed_dict)
        y = np.argmax(y, -1)
        for batch_id in range(len(dia_len)):
            labels = y[batch_id, :dia_len[batch_id]]
            preds = sequence[batch_id, :dia_len[batch_id]]
            probas = scores[batch_id, :dia_len[batch_id], 1]
            total_labels.append(labels)
            total_predictions.append(preds)
            if total_labels_flat.size == 0:
                total_labels_flat, total_predictions_flat, total_scores_flat = labels, preds, probas
            else:
                total_labels_flat = np.concatenate([total_labels_flat, labels])
                total_predictions_flat = np.concatenate([total_predictions_flat, preds])
                total_scores_flat = np.concatenate([total_scores_flat, probas])

    return total_labels, total_predictions, total_labels_flat, total_predictions_flat, total_scores_flat


def save_metrics_json(summary, output_path):
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def save_metrics_txt(summary, output_path):
    lines = ['Metric\tValue']
    lines.append(f"Accuracy\t{summary['accuracy']*100:.2f}")
    lines.append(f"Precision\t{summary['precision']*100:.2f}")
    lines.append(f"Recall\t{summary['recall']*100:.2f}")
    lines.append(f"F1\t{summary['f1']*100:.2f}")
    lines.append(f"Macro-F1\t{summary['macro_f1']*100:.2f}")
    lines.append(f"AUC\t{summary['auc']*100:.2f}")
    lines.append(f"GT-I\t{summary['gt1']*100:.2f}")
    lines.append(f"GT-II\t{summary['gt2']*100:.2f}")
    lines.append(f"GT-III\t{summary['gt3']*100:.2f}")
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def save_classification_report(labels_flat, preds_flat, cm, output_path):
    report = classification_report(
        labels_flat, preds_flat,
        target_names=CLASS_NAMES, digits=4
    )
    with open(output_path, 'w') as f:
        f.write(report)
        f.write('\n\nConfusion matrix:\n')
        f.write(str(cm) + '\n')


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticklabels(CLASS_NAMES, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=16, fontweight='bold', color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(labels_flat, scores_flat, auc_score, output_path):
    fpr_arr, tpr_arr, _ = roc_curve(labels_flat, scores_flat, pos_label=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_arr, tpr_arr, 'b-', linewidth=2.5,
            label=f'DAMI (AUC = {auc_score*100:.2f}%)')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    ax.fill_between(fpr_arr, tpr_arr, alpha=0.15, color='blue')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_lambda_analysis(labels_seq, preds_seq, output_path):
    lambdas = [0.99, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -0.99]
    gt1_vals, gt2_vals, gt3_vals = [], [], []

    for lam in lambdas:
        g1, g2, g3 = get_gtt_score(labels_seq, preds_seq, lamb=lam)
        gt1_vals.append(g1 * 100)
        gt2_vals.append(g2 * 100)
        gt3_vals.append(g3 * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, gt1_vals, 'o-', label='GT-I', linewidth=2, markersize=6)
    ax.plot(lambdas, gt2_vals, 's-', label='GT-II', linewidth=2, markersize=6)
    ax.plot(lambdas, gt3_vals, '^-', label='GT-III', linewidth=2, markersize=6)
    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('GT Scores vs Lambda — Test Set', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return [{'lambda': l, 'gt1': g1, 'gt2': g2, 'gt3': g3}
            for l, g1, g2, g3 in zip(lambdas, gt1_vals, gt2_vals, gt3_vals)]


def main():
    parser = argparse.ArgumentParser('Path A evaluate — full artefact generation')
    parser.add_argument('--data_name', default='clothing')
    parser.add_argument('--model_name', default='dami')
    parser.add_argument('--suffix', default='.128.trend')
    parser.add_argument('--split', default='test', choices=['train', 'eval', 'test'])
    parser.add_argument('--checkpoint', default='best', choices=['best', 'last'])
    parser.add_argument('--memory', default='0')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--trend_features', default='full')
    args = parser.parse_args()

    tf.set_random_seed(7)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    data_loader = Data_loader(data_name=args.data_name)
    data_config = data_loader.load_config(
        os.path.join(get_mhch_config_dir(), 'data', 'config.' + args.data_name + '.json'))
    model_path = os.path.join(get_extension_config_dir(), 'model', 'config.' + args.model_name + '.json')
    if not os.path.isfile(model_path):
        model_path = os.path.join(get_mhch_config_dir(), 'model', 'config.' + args.model_name + '.json')
    model_config = data_loader.load_config(model_path)
    if args.trend_features:
        model_config['trend_features'] = args.trend_features
    data_loader.trend_features = model_config.get('trend_features', 'full')

    with open(os.path.join(get_data_dir(), args.data_name, 'vocab.pkl'), 'rb') as fp:
        vocab = pkl.load(fp)

    trend_mode = model_config.get('trend_features', 'full')
    base_out = args.output_dir or get_results_dir()
    output_dir = os.path.join(base_out, 'path_a', args.data_name, trend_mode, args.split)
    os.makedirs(output_dir, exist_ok=True)

    network = DAMI(memory=float(args.memory), vocab=vocab, config_dict=model_config)
    network.set_nb_words(min(vocab.size(), data_config['nb_words']) + 1)
    network.set_data_name(args.data_name)
    network.set_name(args.model_name + args.suffix + 'train')
    network.set_from_model_config(model_config)
    network.set_from_data_config(data_config)
    network.build_graph()
    patch_network_artifact_dirs(network)
    network.restore(network.save_dir, network.model_name + '.' + args.checkpoint)

    labels_seq, preds_seq, labels_flat, preds_flat, scores_flat = collect_predictions(
        network, data_loader.data_generator_m, args.data_name, args.split,
        model_config['batch_size'], data_config['nb_classes'])

    # --- Compute metrics ---
    accuracy, precision, recall, f1, macro_f1, _, _ = get_a_p_r_f_sara(labels_flat, preds_flat, category=1)
    fpr_arr, tpr_arr, _ = roc_curve(labels_flat, scores_flat, pos_label=1)
    auc_score = auc(fpr_arr, tpr_arr)
    gt1, gt2, gt3 = get_gtt_score(labels_seq, preds_seq)
    cm = confusion_matrix(labels_flat, preds_flat)

    summary = {
        'accuracy': float(accuracy), 'precision': float(precision), 'recall': float(recall),
        'f1': float(f1), 'macro_f1': float(macro_f1), 'auc': float(auc_score),
        'gt1': float(gt1), 'gt2': float(gt2), 'gt3': float(gt3),
    }

    # --- Save all artefacts ---
    print(f'\n=== {args.split.upper()} Results ({trend_mode}) ===')
    print(f"  F1={f1*100:.2f}  Macro-F1={macro_f1*100:.2f}  AUC={auc_score*100:.2f}")
    print(f"  GT-I={gt1*100:.2f}  GT-II={gt2*100:.2f}  GT-III={gt3*100:.2f}")

    # 1. metrics_test.json
    json_path = os.path.join(output_dir, f'metrics_{args.split}.json')
    save_metrics_json(summary, json_path)
    print(f'  Saved: {json_path}')

    # 2. metrics_test.txt
    txt_path = os.path.join(output_dir, f'metrics_{args.split}.txt')
    save_metrics_txt(summary, txt_path)
    print(f'  Saved: {txt_path}')

    # 3. classification_report_test.txt
    report_path = os.path.join(output_dir, f'classification_report_{args.split}.txt')
    save_classification_report(labels_flat, preds_flat, cm, report_path)
    print(f'  Saved: {report_path}')

    # 4. confusion_matrix_test.png
    cm_path = os.path.join(output_dir, f'confusion_matrix_{args.split}.png')
    plot_confusion_matrix(cm, cm_path)
    print(f'  Saved: {cm_path}')

    # 5. roc_curve_test.png
    roc_path = os.path.join(output_dir, f'roc_curve_{args.split}.png')
    plot_roc_curve(labels_flat, scores_flat, auc_score, roc_path)
    print(f'  Saved: {roc_path}')

    # 6. lambda_analysis.png
    lambda_path = os.path.join(output_dir, 'lambda_analysis.png')
    lambda_results = plot_lambda_analysis(labels_seq, preds_seq, lambda_path)
    print(f'  Saved: {lambda_path}')

    # Save lambda results as JSON too
    lambda_json_path = os.path.join(output_dir, 'lambda_analysis.json')
    with open(lambda_json_path, 'w') as f:
        json.dump(lambda_results, f, indent=2)

    print(f'\nAll artefacts saved to: {output_dir}/')


if __name__ == '__main__':
    main()
