# MCIS Research — Machine-Human Chatting Handoff

This workspace contains the **MHCH (Machine-Human Chatting Handoff)** stack: the published DAMI baseline and a linked Path A extension for sentiment-trend features.

---

## Projects


| Directory                                                   | Branch / role              | Purpose                                                                                                |
| ----------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------ |
| **[MHCH-DAMI](MHCH-DAMI/)**                                 | `main`                     | Official TensorFlow DAMI implementation (AAAI 2021). **Keep minimal changes.**                         |
| **[MHCH-DAMI-sentiment-trend](MHCH-DAMI-sentiment-trend/)** | `main` or feature branches | Path A: polarity, slope, volatility features + ablations. **Links to MHCH-DAMI** via `MHCH_DAMI_ROOT`. |
| **[MHCH-DAMI-bert](MHCH-DAMI-bert/)**                       | standalone                 | BERT-based handover model — standalone & with flag features. Independent PyTorch implementation.       |


### Documentation


| File                                                                         | Content                               |
| ---------------------------------------------------------------------------- | ------------------------------------- |
| `[path-a-sentiment-trend-guide.md](path-a-sentiment-trend-guide.md)`         | Full Path A research guide (Days 7–9) |
| `[MHCH-DAMI/README.md](MHCH-DAMI/README.md)`                                 | Upstream DAMI usage                   |
| `[MHCH-DAMI-sentiment-trend/README.md](MHCH-DAMI-sentiment-trend/README.md)` | Extension setup, flags, train/eval    |
| `[MHCH-DAMI-sentiment-trend/LINK.md](MHCH-DAMI-sentiment-trend/LINK.md)`     | How the two repos connect             |


---

## Quick workflow

### 1. Baseline DAMI (upstream only)

```bash
cd MHCH-DAMI
conda activate mhch-dami
python main.py --phase train --model_name dami --data_name clothing \
  --memory 0 --suffix .128 --mode train --ways dami
```

### 2. Path A (extension)

```bash
export MHCH_DAMI_ROOT=$PWD/MHCH-DAMI
cd MHCH-DAMI-sentiment-trend
pip install -r requirements.txt

# Build trend features (one-time, writes 9th field to shared pickles)
python -m path_a.scripts.build_trend_features --data_name clothing
```

### 3. Ablation training & evaluation


| Variant           | Train command                                                                               | Eval command                                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Full trend bundle | `python train.py --data_name clothing --trend_features full --suffix .128.trend`            | `python evaluate.py --data_name clothing --trend_features full --suffix .128.trend --split test`            |
| **Polarity only** | `python train.py --data_name clothing --trend_features pol_only --suffix .128.pol`          | `python evaluate.py --data_name clothing --trend_features pol_only --suffix .128.pol --split test`          |
| Slope-3 only      | `python train.py --data_name clothing --trend_features slope3_only --suffix .128.slope3`    | `python evaluate.py --data_name clothing --trend_features slope3_only --suffix .128.slope3 --split test`    |
| Slope-5 only      | `python train.py --data_name clothing --trend_features slope5_only --suffix .128.slope5`    | `python evaluate.py --data_name clothing --trend_features slope5_only --suffix .128.slope5 --split test`    |
| Slope-7 only      | `python train.py --data_name clothing --trend_features slope7_only --suffix .128.slope7`    | `python evaluate.py --data_name clothing --trend_features slope7_only --suffix .128.slope7 --split test`    |
| Volatility-5 only | `python train.py --data_name clothing --trend_features volatility5_only --suffix .128.vol5` | `python evaluate.py --data_name clothing --trend_features volatility5_only --suffix .128.vol5 --split test` |


Batch ablation report:

```bash
python run_path_a_ablation.py --data_name clothing --split test
```

---

## Repository layout

```
Project/
├── README.md                          ← this file
├── path-a-sentiment-trend-guide.md
├── MHCH-DAMI/                         ← upstream baseline
│   ├── main.py
│   ├── data/
│   └── networks/DAMI.py
├── MHCH-DAMI-sentiment-trend/         ← Path A extension
│   ├── path_a/
│   ├── train.py
│   ├── evaluate.py
│   ├── weights/
│   └── results/
└── MHCH-DAMI-bert/                    ← BERT models (standalone)
    ├── models/
    │   ├── bert_handover.py           ← BERT-only classifier
    │   └── bert_handover_with_flags.py ← BERT + trend flag features
    ├── utils/
    ├── train.py
    ├── evaluate.py
    ├── run_ablation.py
    ├── weights/
    └── results/
```

**Data** live under `MHCH-DAMI/data/` and are shared. Path A adds an optional 9th pickle field `trend_list`; upstream loaders ignore extra fields if only 8 are read.

**Weights:** baseline checkpoints under `MHCH-DAMI/weights/`; Path A under `MHCH-DAMI-sentiment-trend/weights/`.

**Results:** evaluation outputs under `MHCH-DAMI-sentiment-trend/results/path_a/<dataset>/<variant>/<split>/`.

---

## Ablation progress


| Variant          | Dataset  | Status   | F1    | GT-I  | GT-II | GT-III |
| ---------------- | -------- | -------- | ----- | ----- | ----- | ------ |
| Baseline (DAMI)  | clothing | done     | 67.02 | 68.32 | 75.74 | 79.65  |
| pol_only         | clothing | **next** | —     | —     | —     | —      |
| full             | clothing | pending  | —     | —     | —     | —      |
| slope3_only      | clothing | pending  | —     | —     | —     | —      |
| slope5_only      | clothing | pending  | —     | —     | —     | —      |
| slope7_only      | clothing | pending  | —     | —     | —     | —      |
| volatility5_only | clothing | pending  | —     | —     | —     | —      |


---

## Git branches (MHCH-DAMI)

- `**main`** — Match [WeijiaLau/MHCH-DAMI](https://github.com/WeijiaLau/MHCH-DAMI); use for paper reproduction.
- `**Sentiment-Trend-Features**` — Historical branch; new work should use `**MHCH-DAMI-sentiment-trend**` instead of modifying upstream.

Initialize the extension as its own git repo if you publish it separately:

```bash
cd MHCH-DAMI-sentiment-trend
git init
git add .
git commit -m "Path A sentiment-trend extension linked to MHCH-DAMI"
```

