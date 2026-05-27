#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate BERT models for chat handover and generate result artefacts.

Produces: metrics JSON/TXT, classification report, confusion matrix,
ROC curve, and lambda analysis (GT scores).

Examples:
  # Evaluate standalone BERT
  python evaluate.py --data_name clothing --model_type bert_only --suffix .bert --split test

  # Evaluate BERT + flags
  python evaluate.py --data_name clothing --model_type bert_flags --trend_features full --suffix .bert.full --split test
"""

import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.bert_handover import BertHandover
from models.bert_handover_with_flags import BertHandoverWithFlags
from utils.data_loader import (
    create_dataloader, DialogueSequenceDataset, TREND_MODE_INDICES, load_raw_dialogues
)
from utils.metrics import compute_all_metrics, compute_sequence_metrics, get_gtt_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BERT-Evaluate')

CLASS_NAMES = ['Normal', 'Transferable']


def parse_args():
    parser = argparse.ArgumentParser('MHCH-DAMI BERT — evaluate')
    parser.add_argument('--data_name', default='clothing', choices=['clothing', 'makeup'])
    parser.add_argument('--model_type', default='bert_only', choices=['bert_only', 'bert_flags'])
    parser.add_argument('--trend_features', default='full')
    parser.add_argument('--suffix', default='.bert')
    parser.add_argument('--split', default='test', choices=['train', 'eval', 'test'])
    parser.add_argument('--checkpoint', default='best', choices=['best', 'last'])
    parser.add_argument('--bert_model', default='bert-base-chinese')
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_context_turns', type=int, default=5)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(args, device):
    """Load trained model from checkpoint."""
    trend_features = args.trend_features if args.model_type == 'bert_flags' else 'baseline'
    trend_dim = len(TREND_MODE_INDICES.get(trend_features, []))

    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'weights',
        args.data_name, f'{args.model_type}{args.suffix}'
    )

    checkpoint_file = 'best_model.pt' if args.checkpoint == 'best' else 'last_model.pt'
    checkpoint_path = os.path.join(save_dir, checkpoint_file)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})
    bert_model_name = saved_args.get('bert_model', args.bert_model)

    if args.model_type == 'bert_only':
        model = BertHandover(
            bert_model_name=bert_model_name,
            nb_classes=2,
            dropout=0.0,
        )
    else:
        model = BertHandoverWithFlags(
            bert_model_name=bert_model_name,
            nb_classes=2,
            dropout=0.0,
            trend_feature_dim=trend_dim if trend_dim > 0 else 5,
            flag_hidden_dim=saved_args.get('flag_hidden_dim', 32) if 'flag_hidden_dim' in saved_args else 32,
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    logger.info(f"Val metrics at save: {checkpoint.get('val_metrics', {})}")
    return model


def collect_turn_predictions(model, dataloader, device):
    """Collect per-turn predictions for flat metrics."""
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label']

            kwargs = {}
            if 'trend_features' in batch:
                kwargs['trend_features'] = batch['trend_features'].to(device)

            logits = model(input_ids, attention_mask, token_type_ids, **kwargs)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_scores)


def collect_sequence_predictions(model, data_name, split, tokenizer, max_seq_len, max_context_turns, trend_features, device):
    """Collect predictions organized by dialogue for GT scoring."""
    from utils.data_loader import load_raw_dialogues, TREND_MODE_INDICES
    from transformers import BertTokenizer

    raw = load_raw_dialogues(data_name, split)
    trend_indices = TREND_MODE_INDICES.get(trend_features, [])

    labels_seq, preds_seq = [], []

    with torch.no_grad():
        for dia_idx in tqdm(range(len(raw['dia_lens'])), desc="Sequence predictions"):
            dia_len = raw['dia_lens'][dia_idx]
            labels = raw['labels'][dia_idx][:dia_len]
            roles = raw['roles'][dia_idx][:dia_len]
            word_ids = raw['dialogues_ids'][dia_idx][:dia_len]
            trends = raw['trends'][dia_idx][:dia_len] if raw['trends'] is not None else None

            dia_labels = []
            dia_preds = []

            for turn_idx in range(dia_len):
                context_start = max(0, turn_idx - max_context_turns)
                context_turn_ids = word_ids[context_start:turn_idx + 1]
                context_roles = roles[context_start:turn_idx + 1]

                text_parts = []
                for turn_id_seq, role in zip(context_turn_ids, context_roles):
                    role_prefix = "[Agent]" if role == 1 else "[Customer]"
                    turn_text = ' '.join([str(int(wid)) for wid in turn_id_seq if wid != 0])
                    text_parts.append(f"{role_prefix} {turn_text}")
                combined_text = " [SEP] ".join(text_parts)

                encoding = tokenizer(
                    combined_text, max_length=max_seq_len,
                    padding='max_length', truncation=True, return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                token_type_ids = encoding['token_type_ids'].to(device)

                kwargs = {}
                if trends is not None and trend_indices:
                    tv = np.array(trends[turn_idx], dtype=np.float32)[trend_indices]
                    kwargs['trend_features'] = torch.tensor(tv, dtype=torch.float32).unsqueeze(0).to(device)

                logits = model(input_ids, attention_mask, token_type_ids, **kwargs)
                pred = torch.argmax(logits, dim=-1).item()

                dia_labels.append(int(labels[turn_idx]))
                dia_preds.append(pred)

            labels_seq.append(dia_labels)
            preds_seq.append(dia_preds)

    return labels_seq, preds_seq


def save_metrics(summary, output_dir, split):
    json_path = os.path.join(output_dir, f'metrics_{split}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    txt_path = os.path.join(output_dir, f'metrics_{split}.txt')
    lines = ['Metric\tValue']
    for key, val in summary.items():
        lines.append(f"{key}\t{val*100:.2f}")
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return json_path, txt_path


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticklabels(CLASS_NAMES, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=16, fontweight='bold', color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(labels_flat, scores_flat, auc_score, output_path, model_label='BERT'):
    fpr_arr, tpr_arr, _ = roc_curve(labels_flat, scores_flat, pos_label=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_arr, tpr_arr, 'b-', linewidth=2.5,
            label=f'{model_label} (AUC = {auc_score*100:.2f}%)')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
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
    ax.set_title('GT Scores vs Lambda', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return [{'lambda': l, 'gt1': g1, 'gt2': g2, 'gt3': g3}
            for l, g1, g2, g3 in zip(lambdas, gt1_vals, gt2_vals, gt3_vals)]


def main():
    args = parse_args()
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    trend_features = args.trend_features if args.model_type == 'bert_flags' else 'baseline'
    model_label = 'BERT' if args.model_type == 'bert_only' else f'BERT+Flags({trend_features})'

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results',
        args.data_name, args.model_type, trend_features, args.split
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Model: {args.model_type}, Trend: {trend_features}")
    logger.info(f"Dataset: {args.data_name}, Split: {args.split}")
    logger.info(f"Output: {output_dir}")

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    model = load_model(args, device)

    logger.info("Collecting turn-level predictions...")
    dataloader = create_dataloader(
        data_name=args.data_name, split=args.split, tokenizer=tokenizer,
        batch_size=args.batch_size, max_seq_len=args.max_seq_len,
        max_context_turns=args.max_context_turns,
        trend_features=trend_features, shuffle=False, mode='turn',
    )
    labels_flat, preds_flat, scores_flat = collect_turn_predictions(model, dataloader, device)

    logger.info("Collecting sequence-level predictions for GT scores...")
    labels_seq, preds_seq = collect_sequence_predictions(
        model, args.data_name, args.split, tokenizer,
        args.max_seq_len, args.max_context_turns, trend_features, device
    )

    turn_metrics = compute_all_metrics(labels_flat, preds_flat, scores_flat)
    seq_metrics = compute_sequence_metrics(labels_seq, preds_seq)
    summary = {**turn_metrics, **seq_metrics}

    print(f"\n{'='*60}")
    print(f"  {args.split.upper()} Results — {model_label}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {summary['accuracy']*100:.2f}%")
    print(f"  Precision: {summary['precision']*100:.2f}%")
    print(f"  Recall:    {summary['recall']*100:.2f}%")
    print(f"  F1:        {summary['f1']*100:.2f}%")
    print(f"  Macro-F1:  {summary['macro_f1']*100:.2f}%")
    print(f"  AUC:       {summary['auc']*100:.2f}%")
    print(f"  GT-I:      {summary['gt1']*100:.2f}%")
    print(f"  GT-II:     {summary['gt2']*100:.2f}%")
    print(f"  GT-III:    {summary['gt3']*100:.2f}%")
    print(f"{'='*60}\n")

    json_path, txt_path = save_metrics(summary, output_dir, args.split)
    logger.info(f"Saved: {json_path}")
    logger.info(f"Saved: {txt_path}")

    report_path = os.path.join(output_dir, f'classification_report_{args.split}.txt')
    cm = confusion_matrix(labels_flat, preds_flat)
    report = classification_report(labels_flat, preds_flat, target_names=CLASS_NAMES, digits=4)
    with open(report_path, 'w') as f:
        f.write(report)
        f.write(f'\n\nConfusion matrix:\n{cm}\n')
    logger.info(f"Saved: {report_path}")

    cm_path = os.path.join(output_dir, f'confusion_matrix_{args.split}.png')
    plot_confusion_matrix(cm, cm_path)
    logger.info(f"Saved: {cm_path}")

    roc_path = os.path.join(output_dir, f'roc_curve_{args.split}.png')
    plot_roc_curve(labels_flat, scores_flat, summary['auc'], roc_path, model_label)
    logger.info(f"Saved: {roc_path}")

    lambda_path = os.path.join(output_dir, f'lambda_analysis_{args.split}.png')
    lambda_results = plot_lambda_analysis(labels_seq, preds_seq, lambda_path)
    logger.info(f"Saved: {lambda_path}")

    lambda_json = os.path.join(output_dir, 'lambda_analysis.json')
    with open(lambda_json, 'w') as f:
        json.dump(lambda_results, f, indent=2)

    logger.info(f"\nAll artefacts saved to: {output_dir}/")


if __name__ == '__main__':
    main()
