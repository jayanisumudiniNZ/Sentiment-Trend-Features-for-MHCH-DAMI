"""
Evaluation metrics for chat handover prediction.
Includes GT-I/II/III scores compatible with MHCH-DAMI evaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)


def get_gtt_score(labels_seq, preds_seq, lamb=0.5):
    """
    GT-I, GT-II, GT-III scoring for sequence-level evaluation.
    Compatible with MHCH-DAMI utility.get_gtt_score.
    """
    gt1_scores, gt2_scores, gt3_scores = [], [], []

    for labels, preds in zip(labels_seq, preds_seq):
        labels = np.array(labels)
        preds = np.array(preds)

        transfer_indices = np.where(labels == 1)[0]
        if len(transfer_indices) == 0:
            continue

        first_true = transfer_indices[0]
        pred_transfer_indices = np.where(preds == 1)[0]

        if len(pred_transfer_indices) == 0:
            gt1_scores.append(0.0)
            gt2_scores.append(0.0)
            gt3_scores.append(0.0)
            continue

        first_pred = pred_transfer_indices[0]
        n = len(labels)

        # GT-I: binary correctness at first handover point
        gt1 = 1.0 if preds[first_true] == 1 else 0.0
        gt1_scores.append(gt1)

        # GT-II: penalizes late prediction
        delta = first_pred - first_true
        if delta <= 0:
            gt2 = 1.0
        else:
            gt2 = max(0, 1.0 - delta / n)
        gt2_scores.append(gt2)

        # GT-III: rewards early correct with lambda weighting
        if first_pred <= first_true:
            bonus = lamb * (first_true - first_pred) / max(first_true, 1)
            gt3 = min(1.0, 1.0 + bonus)
        else:
            penalty = (1 - lamb) * (first_pred - first_true) / (n - first_true)
            gt3 = max(0, 1.0 - penalty)
        gt3_scores.append(gt3)

    gt1 = np.mean(gt1_scores) if gt1_scores else 0.0
    gt2 = np.mean(gt2_scores) if gt2_scores else 0.0
    gt3 = np.mean(gt3_scores) if gt3_scores else 0.0

    return gt1, gt2, gt3


def compute_all_metrics(labels_flat, preds_flat, scores_flat=None):
    """Compute comprehensive turn-level metrics."""
    acc = accuracy_score(labels_flat, preds_flat)
    precision = precision_score(labels_flat, preds_flat, pos_label=1, zero_division=0)
    recall = recall_score(labels_flat, preds_flat, pos_label=1, zero_division=0)
    f1 = f1_score(labels_flat, preds_flat, pos_label=1, zero_division=0)
    macro_f1 = f1_score(labels_flat, preds_flat, average='macro', zero_division=0)

    auc_score = 0.0
    if scores_flat is not None and len(np.unique(labels_flat)) > 1:
        fpr, tpr, _ = roc_curve(labels_flat, scores_flat, pos_label=1)
        auc_score = auc(fpr, tpr)

    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'macro_f1': float(macro_f1),
        'auc': float(auc_score),
    }


def compute_sequence_metrics(labels_seq, preds_seq):
    """Compute GT-I/II/III for dialogue-level sequence evaluation."""
    gt1, gt2, gt3 = get_gtt_score(labels_seq, preds_seq)
    return {'gt1': float(gt1), 'gt2': float(gt2), 'gt3': float(gt3)}
