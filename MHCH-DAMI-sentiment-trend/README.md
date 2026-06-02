# MHCH-DAMI — Path A Sentiment-Trend Extension

Extension project for **Path A**: sentiment-trend features on **DAMI (Difficulty-Assisted Matching Inference)**. It links to the upstream **[MHCH-DAMI](../MHCH-DAMI/)** repo (keep `**main`** unchanged) and adds training, evaluation, and ablation tooling here.

Research plan: `[../path-a-sentiment-trend-guide.md](../path-a-sentiment-trend-guide.md)`

---

## How this links to MHCH-DAMI

```
Research/Project/
├── MHCH-DAMI/                    ← upstream (main = paper baseline)
│   ├── data/clothing|makeup/     ← shared pickles & vocab
│   ├── main.py                   ← original train entry
│   └── networks/DAMI.py
│
└── MHCH-DAMI-sentiment-trend/    ← this repo (Path A only)
    ├── path_a/                   ← trends, flags, DAMI integration
    ├── train.py / evaluate.py
    ├── weights/                  ← Path A checkpoints
    └── results/
```


| What                  | Where                                       |
| --------------------- | ------------------------------------------- |
| Upstream code         | `$MHCH_DAMI_ROOT` (default: `../MHCH-DAMI`) |
| Datasets              | `$MHCH_DAMI_ROOT/data/`                     |
| Path A code & configs | This repository                             |
| Checkpoints & logs    | `./weights/`, `./networks/logs/`            |


See `[LINK.md](LINK.md)` and `[.env.example](.env.example)`.

```bash
export MHCH_DAMI_ROOT=/path/to/MHCH-DAMI
```

---

## Overview

**DAMI** classifies each turn as **Normal** or **Transferable**. Path A adds user-only trend signals:


| Signal         | Meaning                                    |
| -------------- | ------------------------------------------ |
| `pol_t`        | Current user polarity in [-1, 1]           |
| `slope_k`      | Linear trend over last *k* user turns      |
| `volatility_5` | Std dev of polarity over last 5 user turns |


Pickles store five columns per turn (`pol_t`, `slope_3`, `slope_5`, `slope_7`, `volatility_5`). Training flags select which columns feed the model.

---

## Setup

```bash
conda create -n mhch-path-a python=3.10 -y
conda activate mhch-path-a

# Clone / place MHCH-DAMI beside this repo
# Research/Project/MHCH-DAMI
# Research/Project/MHCH-DAMI-sentiment-trend

cd MHCH-DAMI-sentiment-trend
pip install -r requirements.txt
export MHCH_DAMI_ROOT=../MHCH-DAMI
```

Upstream data prep (in MHCH-DAMI):

```bash
cd ../MHCH-DAMI
python data_prepare.py   # needs MHCH JSON under data/
```

Path A feature build (writes 9th pickle field `trend_list` in **shared** data):

```bash
cd ../MHCH-DAMI-sentiment-trend
python -m path_a.scripts.build_trend_features --data_name clothing
python -m path_a.scripts.build_trend_features --data_name makeup
python -m path_a.scripts.validate_trends --data_name makeup --split train
```

---

## Feature flags (`--trend_features`)

```bash
python run_path_a_ablation.py --list_modes
```


| Flag               | Model             | Features                             | `input_x3` dim | `--suffix`    |
| ------------------ | ----------------- | ------------------------------------ | -------------- | ------------- |
| `baseline`         | DAMI (paper)      | `senti_list`                         | 1              | `.128`        |
| `full`             | DAMI + **bundle** | `pol_t` + `slope_5` + `volatility_5` | 3              | `.128.trend`  |
| `full_slope3`      | DAMI + **bundle** | `pol_t` + `slope_3` + `volatility_5` | 3              | `.128.trend.s3` |
| `pol_only`         | DAMI + polarity   | `pol_t`                              | 1              | `.128.pol`    |
| `slope3_only`      | DAMI + slope      | `slope_3`                            | 1              | `.128.slope3` |
| `slope5_only`      | DAMI + slope      | `slope_5`                            | 1              | `.128.slope5` |
| `slope7_only`      | DAMI + slope      | `slope_7`                            | 1              | `.128.slope7` |
| `volatility5_only` | DAMI + volatility | `volatility_5`                       | 1              | `.128.vol5`   |


Aliases: `volatility_5_only`, `bundle`, `trend_full`, etc. (see `path_a/core/trend_features.py`).

---

## Training

```bash
# Full trend bundle (slope_5)
python train.py --data_name clothing --trend_features full --suffix .128.trend --memory 0

# Full trend bundle (slope_3) — alternate flag
python train.py --data_name clothing --trend_features full_slope3 --suffix .128.trend.s3 --memory 0

# Baseline (same weights naming as paper; uses upstream senti only)
python train.py --data_name clothing --trend_features baseline --suffix .128 --memory 0

# Ablations
python train.py --data_name makeup --trend_features pol_only --suffix .128.pol
python train.py --data_name makeup --trend_features slope5_only --suffix .128.slope5
python train.py --data_name makeup --trend_features volatility5_only --suffix .128.vol5
```

Checkpoints: `weights/<dataset>/dami<suffix>train/`

---

## Evaluation

```bash
python evaluate.py --data_name clothing --split test --trend_features full \
  --suffix .128.trend --checkpoint best

python run_path_a_ablation.py --data_name clothing --split test
```

Outputs: `results/path_a/<dataset>/<flag>/<split>/`

Report template: copy from `[results/03_sentiment_trend_report.md](results/03_sentiment_trend_report.md)` when filling Day 9 results.

---

## Project layout

```
path_a/
├── bootstrap.py              # MHCH_DAMI_ROOT, sys.path, tf_compat
├── core/
│   ├── trends.py             # SnowNLP + slope + volatility
│   └── trend_features.py     # Feature flags
├── integrations/
│   ├── dami.py               # DAMI with variable sentiment dim
│   └── data_loader.py        # Loads trend_list from shared pickles
└── scripts/
    ├── build_trend_features.py
    ├── validate_trends.py
    └── run_path_a_ablation.py
train.py
evaluate.py
config/model/config.dami.json
tf_compat.py                  # TF 2.x shim for upstream graph
```

---

## Metrics & baselines


| Metric                | Description                               |
| --------------------- | ----------------------------------------- |
| F1 / Macro-F1         | Utterance-level transferable detection    |
| GT-I / GT-II / GT-III | Golden Transfer within Tolerance (timing) |


Published **Clothing** DAMI: F1 **67.3**, GT-I **70.3**, GT-II **79.1**, GT-III **83.9**.

---

## Troubleshooting


| Issue                  | Fix                                                    |
| ---------------------- | ------------------------------------------------------ |
| `MHCH-DAMI not found`  | Set `MHCH_DAMI_ROOT` to upstream clone                 |
| `trend_list missing`   | Run `build_trend_features.py`                          |
| `senti_list all zeros` | Add JSON to upstream `data/` and run `data_prepare.py` |
| Shape mismatch         | Match `--suffix` and `--trend_features` with training  |


---

## Citation

Liu et al., *Time to Transfer: Predicting and Evaluating Machine-Human Chatting Handoff*, AAAI 2021.  
Upstream: [WeijiaLau/MHCH-DAMI](https://github.com/WeijiaLau/MHCH-DAMI).