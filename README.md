# Uncertainty-Aware Lithology Classification from Well Logs

An independent, reproducible benchmark for classifying lithofacies along a
well. It compares a transparent XGBoost baseline, a context-aware 1D U-Net,
and a conditional latent-variable **Probabilistic 1D U-Net**.

> Status: implementation and real-data harness audit complete. The XGBoost and
> neural benchmark table remains explicitly pending until those training runs
> finish. No private or confidential well data is used.

## Why this project exists

Lithology labels occur along depth, not as unrelated spreadsheet rows. Two
adjacent samples share geological context, bed boundaries matter, rare classes
matter, and missing tools are common. This repository treats the task as both:

1. a tabular classification problem, to establish a strong XGBoost baseline;
2. a one-dimensional semantic-segmentation problem, so the network can learn
   multi-scale patterns and continuous intervals.

The probabilistic model goes one step further. It samples a latent variable to
produce multiple coherent label sequences when the log response permits more
than one interpretation. It is not Donald Specht's Probabilistic Neural
Network, and an ordinary softmax output alone is not called a Probabilistic
U-Net here.

## What makes the benchmark rigorous

- **Whole-well holdout:** no row from an evaluation well appears in training.
- **Train-only preprocessing:** imputation and robust scaling are fitted only
  on training wells.
- **Missingness channels:** each physical log has a matching missing-value
  indicator rather than silently treating imputed values as observations.
- **Three controlled models:** XGBoost, deterministic 1D U-Net, and
  Probabilistic 1D U-Net share the same well split and input logs.
- **Imbalance-aware training:** inverse-square-root class weighting prevents
  common shale intervals from dominating every update.
- **Overlapping-window reconstruction:** windows never cross well boundaries;
  overlapping probabilities are averaged back onto the original depth grid.
- **Beyond accuracy:** macro F1, weighted F1, balanced accuracy, negative log
  likelihood, calibration error, per-well metrics, and boundary F1.
- **Reproducible artifacts:** frozen well IDs, preprocessing object, model
  weights, history, reports, confusion matrix, and a held-out well track.

## Dataset

This project uses the public FORCE 2020 dataset: 118 wells from the Norwegian
Sea with professionally interpreted lithofacies and partially cleaned well
logs.

- Dataset DOI: [10.5281/zenodo.4351156](https://doi.org/10.5281/zenodo.4351156)
- Official archive: [Zenodo record](https://zenodo.org/records/4351156)
- Competition archive: [FORCE 2020 repository](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition)
- Original well-log licence: Norwegian Licence for Open Government Data 2.0

Required attribution:

> Lithofacies data was provided by the FORCE Machine Learning competition with
> well logs and seismic 2020.

The repository never commits the downloaded archive or processed observations.
See [`data/README.md`](data/README.md).

## Inputs and targets

Default physical logs:

| Curve | Interpretation |
|---|---|
| `GR` | Natural gamma-ray response |
| `RHOB` | Bulk density |
| `NPHI` | Neutron porosity |
| `RDEP` | Deep resistivity |
| `DTC` | Compressional slowness |
| `PEF` | Photoelectric factor |

Every input is accompanied by a binary missingness channel. The target is
`FORCE_2020_LITHOFACIES_LITHOLOGY`, containing twelve competition classes:
sandstone, sandstone/shale, shale, marl, dolomite, limestone, chalk, halite,
anhydrite, tuff, coal, and basement.

## Architecture

```text
Multichannel log sequence [channels × depth]
                    │
         ┌──────────┴──────────┐
         │                     │
   1D U-Net features     Conditional prior
         │                p(z | logs)
         │                     │
         └──────────┬──────────┘
                    │ sample z
                    ▼
            segmentation decoder
                    │
     class probability at every depth
                    │
         multiple coherent samples
```

During training, a posterior network `q(z | logs, labels)` guides the latent
representation. The objective is:

```text
weighted cross-entropy + beta × KL(q(z|x,y) || p(z|x))
```

At inference, samples come only from `p(z | logs)`. Averaging samples provides
the reported predictive distribution; retaining individual samples shows
alternative complete interpretations.

### Scientific limitation

FORCE supplies one reference interpretation per sample, not multiple expert
annotations. Consequently, the probabilistic model's diversity must not be
described as a validated distribution of all geologically correct answers.
This repository measures accuracy, calibration, sample diversity and boundary
behaviour separately and reports negative results if the latent model offers
no benefit.

## Installation

```bash
git clone <your-repository-url>
cd lithology-probabilistic-unet
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .
```

## Reproduce the benchmark

### 1. Download and prepare FORCE 2020

Download the LAS archive from the Zenodo record and extract it into
`data/raw/force2020`, then run:

```bash
lithology-unet prepare-las data/raw/force2020 \
  --output data/processed/force2020.csv
```

### 2. Train the controlled baselines

```bash
lithology-unet train data/processed/force2020.csv \
  --model xgboost --output artifacts/xgboost

lithology-unet train data/processed/force2020.csv \
  --model unet --output artifacts/unet

lithology-unet train data/processed/force2020.csv \
  --model probabilistic_unet --output artifacts/probabilistic_unet
```

All three commands use the same deterministic well-level split when the same
configuration seed is used.

### 3. Run without downloading data

The synthetic generator verifies engineering behaviour only; it is explicitly
not presented as a geological simulator or scientific benchmark.

```bash
lithology-unet make-synthetic --output data/synthetic/force_like.csv
lithology-unet train data/synthetic/force_like.csv \
  --model probabilistic_unet --output artifacts/synthetic_smoke
```

## Evaluation contract

The primary model-selection metric is **macro F1 on held-out wells**. Accuracy
is secondary because a high score can conceal failure on rare lithologies.

| Metric | Question answered |
|---|---|
| Macro F1 | Does each class receive equal importance? |
| Balanced accuracy | Is recall consistent across classes? |
| Weighted F1 | What is performance at observed class prevalence? |
| NLL | Does the model assign probability to the correct class? |
| ECE | Do confidence values agree with observed correctness? |
| Boundary F1 | Are lithological transitions placed near the reference? |
| Per-well metrics | Does a small group of wells dominate the pooled result? |

No claim that the neural model is “better” is made until it beats the baseline
on the same untouched wells and the improvement is stable across wells.

## Verified three-class results

For direct context with a public three-class XGBoost project, the FORCE labels
were restricted to sandstone (`30000`), shale (`65000`) and limestone
(`70000`). All of our models used the same frozen 76/18/24
train/validation/test well split. Test size was 266,222 rows.

| Model | Accuracy | Macro F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|---:|
| XGBoost | 0.9369 | **0.8694** | 0.8422 | 0.1985 |
| 1D U-Net | 0.9076 | 0.8321 | 0.8529 | 0.2514 |
| Probabilistic 1D U-Net | 0.9237 | 0.8458 | **0.8700** | 0.2326 |
| Validation-selected ensemble | **0.9377** | 0.8679 | 0.8632 | **0.1883** |

The ensemble uses 40% XGBoost and 60% Probabilistic U-Net, selected by macro F1
on validation wells. It improves accuracy and probability quality but does not
beat pure XGBoost's test macro F1. This is reported as a trade-off, not hidden.

A separate public repository reports 0.93 accuracy and 0.86 macro F1 on a
different random well holdout. That result is author-reported and not a paired
comparison. Our result demonstrates comparable or slightly higher performance
under our documented split; it does not prove universal superiority.

## Outputs

Each neural run writes:

```text
artifacts/<run>/
├── best_model.pt
├── preprocessor.joblib
├── well_splits.json
├── history.json
├── pooled_metrics.json
├── metrics_by_well.csv
├── classification_report.json
├── confusion_matrix.png
└── example_well_track.png
```

## Repository layout

```text
├── configs/                 Experiment configuration
├── data/                    Instructions only; observations are ignored
├── reports/                 Verified aggregate results
├── src/lithology_unet/
│   ├── baseline.py          XGBoost control
│   ├── data.py              Schema, LAS conversion and grouped split
│   ├── preprocessing.py     Train-only robust transformation
│   ├── sequences.py         Non-leaking sequence windows
│   ├── models/              1D U-Net and Probabilistic 1D U-Net
│   ├── training.py          Weighted training and early stopping
│   ├── evaluation.py        Reconstruction, metrics and figures
│   └── pipeline.py          End-to-end experiment
└── tests/                   Leakage, shape and metric tests
```

## Responsible interpretation

This is a research and educational workflow. Predicted lithology should assist,
not replace, geological interpretation. Performance on FORCE 2020 does not
establish suitability for another basin, logging program, tool vintage or
operational decision. Domain shift, missing tools, class definitions and label
subjectivity must be evaluated before deployment.

## References

1. Bormann, P. et al. (2020). *FORCE 2020 Well Log and Lithofacies Dataset for
   Machine Learning Competition*. Zenodo. https://doi.org/10.5281/zenodo.4351156
2. Ronneberger, O., Fischer, P. & Brox, T. (2015). *U-Net: Convolutional
   Networks for Biomedical Image Segmentation*. https://arxiv.org/abs/1505.04597
3. Kohl, S. A. A. et al. (2018). *A Probabilistic U-Net for Segmentation of
   Ambiguous Images*. https://arxiv.org/abs/1806.05034

## Independence statement

This repository was implemented independently. It does not copy source code,
figures, model artifacts or prose from other lithology-classification
repositories. Public projects using XGBoost motivated the inclusion of a tree
baseline, but the sequence models, evaluation contract and documentation here
were designed for this benchmark.
