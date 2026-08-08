# Experiment design

## Research questions

1. Does depth context improve lithofacies classification beyond a row-wise
   gradient-boosted tree on completely unseen wells?
2. Does a conditional latent-variable U-Net improve macro F1, calibration or
   boundary placement over a deterministic 1D U-Net?
3. On which wells and lithologies does each model fail?

## Frozen comparison

All models use identical physical curves, train/validation/test well IDs and
train-fitted preprocessing. The test wells remain unopened for model selection.

The XGBoost baseline receives the current depth sample plus missingness masks.
The neural models receive windows of the same channels. This comparison tests
the value of sequence context, but it does not attribute a gain solely to model
architecture. A later ablation may add rolling features to XGBoost.

## Primary endpoint

Macro F1 pooled across all samples in the held-out wells.

## Required secondary analyses

- balanced and weighted performance;
- per-class precision and recall;
- per-well metric distribution, not only its mean;
- calibration error and negative log likelihood;
- lithological transition placement;
- missing-curve sensitivity;
- probabilistic sample diversity.

## Leakage controls

- Split well names before fitting any preprocessing object.
- Do not construct a window across two wells.
- Do not tune using test wells.
- Preserve a machine-readable split manifest.
- Do not use depth-derived formation statistics calculated from validation or
  test labels.
- Do not report synthetic-data metrics as real scientific performance.

## Probabilistic-model acceptance criteria

The probabilistic model is retained only if it provides at least one measurable
benefit without an unacceptable regression elsewhere:

- improved held-out macro F1;
- improved NLL or calibration;
- improved boundary metric;
- meaningful, stable hypothesis diversity near ambiguous intervals.

Generating visually different samples is not by itself evidence of useful
uncertainty.
