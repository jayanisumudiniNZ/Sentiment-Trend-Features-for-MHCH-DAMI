"""
Standalone BERT model for chat handover decision.
Per-turn binary classification: Normal (0) vs Transferable (1).
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertHandover(nn.Module):
    """
    BERT-based chat handover classifier (standalone, no flag features).
    Input: tokenized dialogue context for each turn.
    Output: binary handover probability per turn.
    """

    def __init__(self, bert_model_name: str = 'bert-base-chinese', nb_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.nb_classes = nb_classes

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, nb_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        """
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            token_type_ids: [B, seq_len]
        Returns:
            logits: [B, nb_classes]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output  # [B, hidden_size]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def predict_proba(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.softmax(logits, dim=-1)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
