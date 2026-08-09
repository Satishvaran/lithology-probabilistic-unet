# Lithology classification with a probabilistic 1D U-Net

This is an active project I built to explore lithology classification from
well logs. The current model works with sandstone, shale and limestone and is
evaluated on complete wells that were not used for training.

I started with XGBoost as a practical baseline, then trained a 1D U-Net to use
the order and context of measurements along a well. I also tested a
probabilistic version of the U-Net so the model can express uncertainty when
the log response is ambiguous.

## Why treat a well as a sequence?

Well-log samples are not independent rows. Neighbouring measurements usually
belong to the same geological interval, and changes between intervals are often
as important as individual values. A 1D U-Net can learn short local changes and
longer patterns at the same time.

The probabilistic model adds a latent variable to the network. During
inference, it can produce more than one plausible sequence and average those
predictions into class probabilities. This is useful for measuring uncertainty,
but it does not replace geological interpretation.

## Data used by the model

The experiments use the FORCE 2020 well-log dataset. The main input curves are:

| Curve | Meaning |
|---|---|
| `GR` | Gamma ray |
| `RHOB` | Bulk density |
| `NPHI` | Neutron porosity |
| `RDEP` | Deep resistivity |
| `DTC` | Compressional slowness |
| `PEF` | Photoelectric factor |

Missing values are imputed using statistics learned from the training wells.
The model also receives a missing-value indicator for every curve, so an
imputed value is not mistaken for a real measurement.

Dataset: [FORCE 2020 on Zenodo](https://doi.org/10.5281/zenodo.4351156)

The downloaded observations and trained weights are deliberately excluded from
Git. Instructions are available in [`data/README.md`](data/README.md).

## Validation

The split is made by well, not by row. This matters because randomly separating
nearby rows would put almost identical measurements in both training and test
sets and make the score look better than it really is.

For the three-class experiment, the split contains:

- 76 training wells
- 18 validation wells
- 24 test wells
- 266,222 test rows

Preprocessing is fitted only on the training wells. The validation wells are
used for model selection, while the test wells are kept untouched until final
evaluation.

## Current results

| Model | Accuracy | Macro F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|---:|
| XGBoost | 0.9369 | **0.8694** | 0.8422 | 0.1985 |
| 1D U-Net | 0.9076 | 0.8321 | 0.8529 | 0.2514 |
| Probabilistic 1D U-Net | 0.9237 | 0.8458 | **0.8700** | 0.2326 |
| Ensemble | **0.9377** | 0.8679 | 0.8632 | **0.1883** |

The ensemble combines 40% XGBoost and 60% probabilistic U-Net probabilities.
That blend was selected using validation wells. It gives the best accuracy and
negative log-likelihood in this experiment, although XGBoost still has a
slightly better test macro F1.

The broad geological intervals are usually identified well. Thin beds and
rapid changes remain harder, so accuracy should not be read as evidence that
every small interval is detected correctly.

## Model outline

```text
well-log curves along depth
          |
   preprocessing and missing-value channels
          |
   1D U-Net encoder and decoder
          |
   class probabilities at each depth
          |
 sandstone / shale / limestone
```

For the probabilistic model, a conditional prior samples a latent variable
before the decoder. Training uses weighted cross-entropy together with a KL
divergence term:

```text
loss = weighted cross-entropy + beta * KL divergence
```

## Running the project

```bash
git clone https://github.com/Satishvaran/lithology-probabilistic-unet.git
cd lithology-probabilistic-unet
python -m venv .venv
```

Activate the environment and install the package:

```bash
# Windows
.venv\Scripts\activate

pip install -e .
```

After downloading and extracting the FORCE LAS files:

```bash
lithology-unet prepare-las data/raw/force2020 \
  --output data/processed/force2020.csv
```

Train a model:

```bash
lithology-unet train data/processed/force2020.csv \
  --model probabilistic_unet \
  --output artifacts/probabilistic_unet
```

The same command accepts `xgboost` or `unet` as the model name. With the same
configuration seed, all models use the same deterministic well split.

There is also a synthetic-data command for checking that the code runs. It is a
software test, not a geological benchmark:

```bash
lithology-unet make-synthetic --output data/synthetic/force_like.csv
```

## Outputs

A neural training run saves the best weights, preprocessing object, frozen well
split, training history, aggregate and per-well metrics, classification report,
confusion matrix and an example held-out-well plot under `artifacts/<run>/`.

These files are ignored by Git because they can be regenerated and may be
large. The verified aggregate results used in this README are kept under
[`reports/`](reports/).

## Project structure

```text
configs/                 experiment settings
data/                    download and preparation instructions
reports/                 verified aggregate results
src/lithology_unet/      data, models, training and evaluation code
tests/                   leakage, shape and metric checks
```

## Notes

This is a research and learning project, not an operational interpretation
tool. A result on these wells may not transfer directly to another basin,
logging programme or set of lithology definitions. The project is still active,
and I plan to continue testing ways to improve thin-bed and boundary detection.

## References

- Bormann, P. et al. (2020), *FORCE 2020 Well Log and Lithofacies Dataset for
  Machine Learning Competition*. [DOI: 10.5281/zenodo.4351156](https://doi.org/10.5281/zenodo.4351156)
- Ronneberger, O., Fischer, P. and Brox, T. (2015), *U-Net: Convolutional
  Networks for Biomedical Image Segmentation*.
- Kohl, S. A. A. et al. (2018), *A Probabilistic U-Net for Segmentation of
  Ambiguous Images*.
