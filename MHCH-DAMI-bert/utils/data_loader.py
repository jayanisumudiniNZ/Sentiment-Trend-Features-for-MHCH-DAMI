"""
Data loader for BERT chat handover model.
Reads existing MHCH-DAMI pickle files and converts dialogues to BERT-compatible format.
"""

import os
import pickle as pkl
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

MHCH_DAMI_ROOT = os.environ.get(
    'MHCH_DAMI_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'MHCH-DAMI'))
)

TREND_FEATURE_NAMES = ('pol_t', 'slope_3', 'slope_5', 'slope_7', 'volatility_5')

TREND_MODE_INDICES = {
    'baseline': [],
    'pol_only': [0],
    'slope3_only': [1],
    'slope5_only': [2],
    'slope7_only': [3],
    'volatility5_only': [4],
    'full': [0, 2, 4],  # pol_t + slope_5 + volatility_5
    'full_slope3': [0, 1, 4],  # pol_t + slope_3 + volatility_5
}


def get_data_path(data_name: str, split: str) -> str:
    split_file = {'train': 'train.pkl', 'eval': 'eval.pkl', 'test': 'test.pkl'}
    return os.path.join(MHCH_DAMI_ROOT, 'data', data_name, split_file[split])


def load_raw_dialogues(data_name: str, split: str):
    """Load raw dialogue data from MHCH-DAMI pickles."""
    path = get_data_path(data_name, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    with open(path, 'rb') as f:
        dialogues_ids_list = pkl.load(f)
        role_list = pkl.load(f)
        tf_list = pkl.load(f)
        pos_list = pkl.load(f)
        senti_list = pkl.load(f)
        dialogues_sent_len_list = pkl.load(f)
        dialogues_len_list = pkl.load(f)
        label_list = pkl.load(f)
        try:
            trend_list = pkl.load(f)
        except EOFError:
            trend_list = None

    logger.info(f"Loaded {len(dialogues_len_list)} dialogues from {path}")
    return {
        'dialogues_ids': dialogues_ids_list,
        'roles': role_list,
        'tf': tf_list,
        'pos': pos_list,
        'senti': senti_list,
        'sent_lens': dialogues_sent_len_list,
        'dia_lens': dialogues_len_list,
        'labels': label_list,
        'trends': trend_list,
    }


def load_vocab(data_name: str):
    """Load vocabulary from MHCH-DAMI data directory."""
    vocab_path = os.path.join(MHCH_DAMI_ROOT, 'data', data_name, 'vocab.pkl')
    with open(vocab_path, 'rb') as f:
        vocab = pkl.load(f)
    return vocab


class DialogueTurnDataset(Dataset):
    """
    Per-turn classification dataset for BERT.
    Each sample = one turn in a dialogue, with context from previous turns.
    Label = handover decision (0=Normal, 1=Transferable).
    """

    def __init__(
        self,
        data_name: str,
        split: str,
        tokenizer: BertTokenizer,
        max_seq_len: int = 128,
        max_context_turns: int = 5,
        trend_features: str = 'baseline',
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_context_turns = max_context_turns
        self.trend_features = trend_features
        self.trend_indices = TREND_MODE_INDICES.get(trend_features, [])

        raw = load_raw_dialogues(data_name, split)
        self.samples = self._build_samples(raw)
        logger.info(
            f"Built {len(self.samples)} turn samples from {len(raw['dia_lens'])} dialogues "
            f"(trend_features={trend_features})"
        )

    def _build_samples(self, raw: dict) -> List[dict]:
        """Flatten dialogues into per-turn samples with context."""
        samples = []
        vocab = None

        for dia_idx in range(len(raw['dia_lens'])):
            dia_len = raw['dia_lens'][dia_idx]
            labels = raw['labels'][dia_idx][:dia_len]
            roles = raw['roles'][dia_idx][:dia_len]
            senti = raw['senti'][dia_idx][:dia_len]
            trends = raw['trends'][dia_idx][:dia_len] if raw['trends'] is not None else None
            word_ids = raw['dialogues_ids'][dia_idx][:dia_len]

            for turn_idx in range(dia_len):
                label = int(labels[turn_idx])

                context_start = max(0, turn_idx - self.max_context_turns)
                context_turn_ids = word_ids[context_start:turn_idx + 1]
                context_roles = roles[context_start:turn_idx + 1]

                trend_vec = None
                if trends is not None and self.trend_indices:
                    turn_trend = np.array(trends[turn_idx], dtype=np.float32)
                    trend_vec = turn_trend[self.trend_indices]

                senti_val = float(senti[turn_idx]) if senti is not None else 0.0

                samples.append({
                    'context_ids': context_turn_ids,
                    'context_roles': context_roles,
                    'label': label,
                    'trend_vec': trend_vec,
                    'senti': senti_val,
                    'dia_idx': dia_idx,
                    'turn_idx': turn_idx,
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        text_parts = []
        for i, (turn_ids, role) in enumerate(zip(sample['context_ids'], sample['context_roles'])):
            role_prefix = "[Agent]" if role == 1 else "[Customer]"
            turn_text = ' '.join([str(int(wid)) for wid in turn_ids if wid != 0])
            text_parts.append(f"{role_prefix} {turn_text}")

        combined_text = " [SEP] ".join(text_parts)

        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'label': torch.tensor(sample['label'], dtype=torch.long),
        }

        if sample['trend_vec'] is not None:
            item['trend_features'] = torch.tensor(sample['trend_vec'], dtype=torch.float32)

        return item


class DialogueSequenceDataset(Dataset):
    """
    Sequence-level dataset: each sample = one full dialogue.
    Returns per-turn labels for sequence evaluation (GT-I/II/III metrics).
    """

    def __init__(
        self,
        data_name: str,
        split: str,
        tokenizer: BertTokenizer,
        max_seq_len: int = 128,
        max_dialogue_len: int = 30,
        max_context_turns: int = 5,
        trend_features: str = 'baseline',
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_dialogue_len = max_dialogue_len
        self.max_context_turns = max_context_turns
        self.trend_features = trend_features
        self.trend_indices = TREND_MODE_INDICES.get(trend_features, [])

        raw = load_raw_dialogues(data_name, split)
        self.dialogues = self._build_dialogues(raw)
        logger.info(f"Built {len(self.dialogues)} dialogue sequences")

    def _build_dialogues(self, raw: dict) -> List[dict]:
        dialogues = []

        for dia_idx in range(len(raw['dia_lens'])):
            dia_len = min(raw['dia_lens'][dia_idx], self.max_dialogue_len)
            labels = raw['labels'][dia_idx][:dia_len]
            roles = raw['roles'][dia_idx][:dia_len]
            senti = raw['senti'][dia_idx][:dia_len]
            trends = raw['trends'][dia_idx][:dia_len] if raw['trends'] is not None else None
            word_ids = raw['dialogues_ids'][dia_idx][:dia_len]

            turn_encodings = []
            turn_trends = []

            for turn_idx in range(dia_len):
                context_start = max(0, turn_idx - self.max_context_turns)
                context_turn_ids = word_ids[context_start:turn_idx + 1]
                context_roles = roles[context_start:turn_idx + 1]

                text_parts = []
                for turn_id_seq, role in zip(context_turn_ids, context_roles):
                    role_prefix = "[Agent]" if role == 1 else "[Customer]"
                    turn_text = ' '.join([str(int(wid)) for wid in turn_id_seq if wid != 0])
                    text_parts.append(f"{role_prefix} {turn_text}")

                combined_text = " [SEP] ".join(text_parts)
                encoding = self.tokenizer(
                    combined_text,
                    max_length=self.max_seq_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                turn_encodings.append(encoding)

                if trends is not None and self.trend_indices:
                    tv = np.array(trends[turn_idx], dtype=np.float32)[self.trend_indices]
                    turn_trends.append(tv)

            dialogues.append({
                'encodings': turn_encodings,
                'labels': labels[:dia_len],
                'dia_len': dia_len,
                'trends': turn_trends if turn_trends else None,
            })

        return dialogues

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        dia_len = dialogue['dia_len']

        input_ids = torch.zeros(self.max_dialogue_len, self.max_seq_len, dtype=torch.long)
        attention_mask = torch.zeros(self.max_dialogue_len, self.max_seq_len, dtype=torch.long)
        token_type_ids = torch.zeros(self.max_dialogue_len, self.max_seq_len, dtype=torch.long)
        labels = torch.zeros(self.max_dialogue_len, dtype=torch.long)

        for t in range(dia_len):
            enc = dialogue['encodings'][t]
            input_ids[t] = enc['input_ids'].squeeze(0)
            attention_mask[t] = enc['attention_mask'].squeeze(0)
            token_type_ids[t] = enc['token_type_ids'].squeeze(0)
            labels[t] = int(dialogue['labels'][t])

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels,
            'dia_len': torch.tensor(dia_len, dtype=torch.long),
        }

        if dialogue['trends'] is not None:
            trend_dim = len(self.trend_indices)
            trend_tensor = torch.zeros(self.max_dialogue_len, trend_dim, dtype=torch.float32)
            for t in range(dia_len):
                trend_tensor[t] = torch.tensor(dialogue['trends'][t], dtype=torch.float32)
            item['trend_features'] = trend_tensor

        return item


def create_dataloader(
    data_name: str,
    split: str,
    tokenizer: BertTokenizer,
    batch_size: int = 16,
    max_seq_len: int = 128,
    max_context_turns: int = 5,
    trend_features: str = 'baseline',
    shuffle: bool = True,
    mode: str = 'turn',
) -> DataLoader:
    """Create a DataLoader for either turn-level or sequence-level prediction."""
    if mode == 'turn':
        dataset = DialogueTurnDataset(
            data_name=data_name,
            split=split,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_context_turns=max_context_turns,
            trend_features=trend_features,
        )
    else:
        dataset = DialogueSequenceDataset(
            data_name=data_name,
            split=split,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_context_turns=max_context_turns,
            trend_features=trend_features,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )
