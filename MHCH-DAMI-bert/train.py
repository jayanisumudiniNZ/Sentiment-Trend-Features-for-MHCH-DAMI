#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train BERT models for chat handover decision.

Supports two modes:
  --model_type bert_only    : Standalone BERT model
  --model_type bert_flags   : BERT + sentiment trend flag features

Examples:
  # BERT only - clothing dataset
  python train.py --data_name clothing --model_type bert_only --suffix .bert

  # BERT + flag features (full trend bundle)
  python train.py --data_name clothing --model_type bert_flags --trend_features full --suffix .bert.full

  # BERT + polarity flag only
  python train.py --data_name clothing --model_type bert_flags --trend_features pol_only --suffix .bert.pol
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.bert_handover import BertHandover
from models.bert_handover_with_flags import BertHandoverWithFlags
from utils.data_loader import create_dataloader, TREND_MODE_INDICES
from utils.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BERT-Train')


def parse_args():
    parser = argparse.ArgumentParser('MHCH-DAMI BERT — train')
    parser.add_argument('--data_name', default='clothing', choices=['clothing', 'makeup'])
    parser.add_argument('--model_type', default='bert_only', choices=['bert_only', 'bert_flags'])
    parser.add_argument('--trend_features', default='full',
                        help='baseline|full|full_slope3|pol_only|slope3_only|slope5_only|slope7_only|volatility5_only')
    parser.add_argument('--suffix', default='.bert', help='Checkpoint suffix')
    parser.add_argument('--config', default=None, help='Path to config JSON')
    parser.add_argument('--bert_model', default='bert-base-chinese')
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_context_turns', type=int, default=5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--device', default=None, help='cuda/mps/cpu (auto-detect if None)')
    return parser.parse_args()


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, accumulation_steps=1):
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        kwargs = {}
        if 'trend_features' in batch:
            kwargs['trend_features'] = batch['trend_features'].to(device)

        logits = model(input_ids, attention_mask, token_type_ids, **kwargs)
        loss = criterion(logits, labels)
        loss = loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs[:, 1].detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_all_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_scores)
    )
    metrics['loss'] = avg_loss
    return metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            kwargs = {}
            if 'trend_features' in batch:
                kwargs['trend_features'] = batch['trend_features'].to(device)

            logits = model(input_ids, attention_mask, token_type_ids, **kwargs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_all_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_scores)
    )
    metrics['loss'] = avg_loss
    return metrics


def main():
    args = parse_args()
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            config = json.load(f)

    bert_model_name = config.get('bert_model_name', args.bert_model)
    max_seq_len = config.get('max_seq_len', args.max_seq_len)
    batch_size = config.get('batch_size', args.batch_size)
    epochs = config.get('epochs', args.epochs)
    lr = config.get('learning_rate', args.lr)
    warmup_ratio = config.get('warmup_ratio', args.warmup_ratio)
    weight_decay = config.get('weight_decay', args.weight_decay)
    dropout = config.get('dropout', args.dropout)
    patience = config.get('early_stopping_patience', args.patience)
    accumulation_steps = config.get('gradient_accumulation_steps', args.gradient_accumulation_steps)

    trend_features = args.trend_features if args.model_type == 'bert_flags' else 'baseline'
    trend_dim = len(TREND_MODE_INDICES.get(trend_features, []))

    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Dataset: {args.data_name}")
    logger.info(f"Trend features: {trend_features} (dim={trend_dim})")
    logger.info(f"BERT model: {bert_model_name}")

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    logger.info("Loading training data...")
    train_loader = create_dataloader(
        data_name=args.data_name, split='train', tokenizer=tokenizer,
        batch_size=batch_size, max_seq_len=max_seq_len,
        max_context_turns=args.max_context_turns,
        trend_features=trend_features, shuffle=True, mode='turn',
    )

    logger.info("Loading validation data...")
    val_loader = create_dataloader(
        data_name=args.data_name, split='eval', tokenizer=tokenizer,
        batch_size=batch_size, max_seq_len=max_seq_len,
        max_context_turns=args.max_context_turns,
        trend_features=trend_features, shuffle=False, mode='turn',
    )

    if args.model_type == 'bert_only':
        model = BertHandover(
            bert_model_name=bert_model_name,
            nb_classes=2,
            dropout=dropout,
        )
    else:
        model = BertHandoverWithFlags(
            bert_model_name=bert_model_name,
            nb_classes=2,
            dropout=dropout,
            trend_feature_dim=trend_dim if trend_dim > 0 else 5,
            flag_hidden_dim=config.get('flag_hidden_dim', 32),
        )

    model = model.to(device)
    logger.info(f"Model parameters: {model.get_num_params():,}")

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_params = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_params, lr=lr)

    total_steps = len(train_loader) * epochs // accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss()

    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'weights',
        args.data_name, f'{args.model_type}{args.suffix}'
    )
    os.makedirs(save_dir, exist_ok=True)

    best_val_f1 = 0.0
    patience_counter = 0
    history = []

    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Save directory: {save_dir}")

    for epoch in range(epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*60}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, accumulation_steps
        )
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f} | "
            f"F1: {train_metrics['f1']*100:.2f} | "
            f"Macro-F1: {train_metrics['macro_f1']*100:.2f} | "
            f"AUC: {train_metrics['auc']*100:.2f}"
        )

        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(
            f"Val   - Loss: {val_metrics['loss']:.4f} | "
            f"F1: {val_metrics['f1']*100:.2f} | "
            f"Macro-F1: {val_metrics['macro_f1']*100:.2f} | "
            f"AUC: {val_metrics['auc']*100:.2f}"
        )

        history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
        })

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            checkpoint_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args),
            }, checkpoint_path)
            logger.info(f"  -> New best model saved (F1={best_val_f1*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1} (patience={patience})")
                break

    last_checkpoint_path = os.path.join(save_dir, 'last_model.pt')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'args': vars(args),
    }, last_checkpoint_path)

    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete! Best val F1: {best_val_f1*100:.2f}%")
    logger.info(f"Checkpoints saved to: {save_dir}")
    logger.info(f"Training history: {history_path}")


if __name__ == '__main__':
    main()
