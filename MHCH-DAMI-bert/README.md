# MHCH-DAMI BERT — Chat Handover with BERT

BERT-based models for Machine-Human Chat Handover decision, as a separate implementation alongside the existing DAMI baseline.

## Models

| Model | Description |
|-------|-------------|
| `bert_only` | Standalone BERT classifier — encodes dialogue context and predicts handover |
| `bert_flags` | BERT + sentiment trend flag features — fuses BERT with polarity/slope/volatility signals |

## Setup

```bash
cd MHCH-DAMI-bert
pip install -r requirements.txt
```

Ensure `MHCH_DAMI_ROOT` points to the upstream data (auto-detects sibling `MHCH-DAMI/` directory):

```bash
export MHCH_DAMI_ROOT=$PWD/../MHCH-DAMI
```

## Training

### BERT Only (standalone)

```bash
# Clothing dataset
python train.py --data_name clothing --model_type bert_only --suffix .bert

# Makeup dataset
python train.py --data_name makeup --model_type bert_only --suffix .bert
```

### BERT + Flag Features

```bash
# Full trend bundle (polarity + slope_5 + volatility_5)
python train.py --data_name clothing --model_type bert_flags --trend_features full --suffix .bert.full

# Full trend bundle (polarity + slope_3 + volatility_5)
python train.py --data_name clothing --model_type bert_flags --trend_features full_slope3 --suffix .bert.full.s3

# Polarity only
python train.py --data_name clothing --model_type bert_flags --trend_features pol_only --suffix .bert.pol

# Slope-3 only
python train.py --data_name clothing --model_type bert_flags --trend_features slope3_only --suffix .bert.slope3

# Slope-5 only
python train.py --data_name clothing --model_type bert_flags --trend_features slope5_only --suffix .bert.slope5

# Slope-7 only
python train.py --data_name clothing --model_type bert_flags --trend_features slope7_only --suffix .bert.slope7

# Volatility-5 only
python train.py --data_name clothing --model_type bert_flags --trend_features volatility5_only --suffix .bert.vol5
```

## Evaluation

```bash
# BERT only
python evaluate.py --data_name clothing --model_type bert_only --suffix .bert --split test

# BERT + flags
python evaluate.py --data_name clothing --model_type bert_flags --trend_features full --suffix .bert.full --split test
```

## Flag Features (Sentiment Trend)

These features are computed from customer utterance sentiment history:

| Flag | Description |
|------|-------------|
| `pol_t` | Current turn polarity (SnowNLP, mapped to [-1, 1]) |
| `slope_3` | Linear sentiment slope over last 3 user turns |
| `slope_5` | Linear sentiment slope over last 5 user turns |
| `slope_7` | Linear sentiment slope over last 7 user turns |
| `volatility_5` | Sentiment std-dev over last 5 user turns |

### Trend Feature Modes

| `--trend_features` | Features Used | Dimension |
|--------------------|---------------|-----------|
| `baseline` | None (BERT only) | 0 |
| `pol_only` | pol_t | 1 |
| `slope3_only` | slope_3 | 1 |
| `slope5_only` | slope_5 | 1 |
| `slope7_only` | slope_7 | 1 |
| `volatility5_only` | volatility_5 | 1 |
| `full` | pol_t + slope_5 + volatility_5 | 3 |
| `full_slope3` | pol_t + slope_3 + volatility_5 | 3 |

## Output Structure

```
results/
└── <dataset>/
    └── <model_type>/
        └── <trend_mode>/
            └── <split>/
                ├── metrics_test.json
                ├── metrics_test.txt
                ├── classification_report_test.txt
                ├── confusion_matrix_test.png
                ├── roc_curve_test.png
                └── lambda_analysis_test.png
```

## Architecture

### BERT Only
```
Input (tokenized dialogue context)
  → BERT encoder
  → [CLS] pooled output
  → Dropout
  → Linear classifier (2 classes)
```

### BERT + Flags
```
Input (tokenized dialogue context)     Flag Features (trend vector)
  → BERT encoder                         → Linear(dim, 32) + ReLU
  → [CLS] pooled output                  → Linear(32, 32) + ReLU
  → Dropout                              ↓
         ↓                               ↓
         └──── Concatenate ──────────────┘
                    ↓
              Linear(hidden+32, 256) + ReLU
                    ↓
              Linear(256, 2)
```

## Comparison with DAMI Baseline

This module is fully independent of the existing TF-based DAMI pipeline. Both systems:
- Read from the same pickle data files (`MHCH-DAMI/data/`)
- Use the same trend features (stored in 9th pickle field)
- Report the same metrics (F1, Macro-F1, AUC, GT-I/II/III)
- Generate comparable artefacts for direct comparison
