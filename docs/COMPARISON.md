# Scope comparison

This project does not claim superiority from repository size or model
complexity. “Better” is defined as a more complete and auditable research
workflow.

| Capability | Typical XGBoost portfolio project | This repository |
|---|---|---|
| Public FORCE data | Yes | Yes, official DOI and licence citation |
| Row classifier | Yes | Controlled XGBoost baseline |
| Sequence context | Usually engineered manually | Learned by 1D U-Net |
| Alternative hypotheses | No | Conditional latent-variable U-Net |
| Whole-well holdout | Sometimes | Enforced and tested |
| Train-only preprocessing | Often implicit | Explicit serialized transformer |
| Missing-log indicators | Optional | Included for every curve |
| Accuracy and F1 | Usually | Macro/weighted F1 and balanced accuracy |
| Calibration | Rare | NLL and expected calibration error |
| Boundary evaluation | Rare | Boundary F1 |
| Per-well failures | Rare | CSV report for every held-out well |
| Reproducibility | Notebook-dependent | CLI, config, artifacts and CI tests |
| Limitations | Brief | Model card and acceptance criteria |

Actual benchmark performance must be filled in only after identical-split runs.

## Observed outcome

On our frozen three-class well holdout, XGBoost achieved 0.9369 accuracy and
0.8694 macro F1. A validation-selected XGBoost/Probabilistic-U-Net ensemble
achieved 0.9377 accuracy and 0.1883 NLL, while the probabilistic model alone had
the strongest balanced accuracy (0.8700). See `reports/README.md` for the full
table and comparison limitations.
