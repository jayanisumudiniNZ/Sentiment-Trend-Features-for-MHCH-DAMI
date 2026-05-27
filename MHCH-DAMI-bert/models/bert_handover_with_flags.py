"""
BERT + Flag Features model for chat handover decision.
Combines BERT text representation with sentiment trend flag features
to make the handover classification decision.
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BertHandoverWithFlags(nn.Module):
    """
    BERT + sentiment trend flag features for chat handover.

    Architecture:
        1. BERT encodes dialogue context -> [CLS] pooled representation
        2. Flag features (polarity, slope, volatility) pass through a small MLP
        3. Both representations are concatenated and passed through a fusion classifier

    This tests whether adding trend features (flag features) to BERT's
    contextual understanding improves handover decision threshold accuracy.
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-chinese',
        nb_classes: int = 2,
        dropout: float = 0.1,
        trend_feature_dim: int = 5,
        flag_hidden_dim: int = 32,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.nb_classes = nb_classes
        self.trend_feature_dim = trend_feature_dim

        self.dropout = nn.Dropout(dropout)

        self.flag_encoder = nn.Sequential(
            nn.Linear(trend_feature_dim, flag_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(flag_hidden_dim, flag_hidden_dim),
            nn.ReLU(),
        )

        fusion_input_dim = self.hidden_size + flag_hidden_dim
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, nb_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, trend_features=None, **kwargs):
        """
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            token_type_ids: [B, seq_len]
            trend_features: [B, trend_feature_dim] - flag features (polarity, slope, volatility)
        Returns:
            logits: [B, nb_classes]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        bert_pooled = outputs.pooler_output  # [B, hidden_size]
        bert_pooled = self.dropout(bert_pooled)

        if trend_features is None:
            trend_features = torch.zeros(
                input_ids.size(0), self.trend_feature_dim,
                device=input_ids.device, dtype=torch.float32
            )

        flag_encoded = self.flag_encoder(trend_features)  # [B, flag_hidden_dim]

        fused = torch.cat([bert_pooled, flag_encoded], dim=-1)
        logits = self.fusion_classifier(fused)
        return logits

    def predict_proba(self, input_ids, attention_mask, token_type_ids=None, trend_features=None, **kwargs):
        logits = self.forward(input_ids, attention_mask, token_type_ids, trend_features)
        return torch.softmax(logits, dim=-1)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_bert_params(self):
        return sum(p.numel() for p in self.bert.parameters() if p.requires_grad)

    def get_flag_params(self):
        flag_params = sum(p.numel() for p in self.flag_encoder.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in self.fusion_classifier.parameters() if p.requires_grad)
        return flag_params + fusion_params
