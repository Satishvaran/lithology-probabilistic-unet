# Model card

## Intended use

Research and education in supervised lithofacies classification from public
well-log sequences. The models predict one of the twelve FORCE 2020 classes at
each measured depth.

## Out-of-scope use

- autonomous drilling, reserves estimation or safety-critical decisions;
- inference in a new basin without external validation;
- treating predictive confidence as calibrated geological certainty;
- claiming multiple sampled outputs are verified expert interpretations.

## Evaluation

Complete wells are held out. Preprocessing parameters are fitted on training
wells only. Macro F1 is primary; calibration, boundary and per-well metrics are
reported alongside it.

## Known limitations

- severe class imbalance;
- missing and heterogeneous logging suites;
- one reference label sequence rather than multiple expert annotations;
- geographic and stratigraphic domain shift;
- overlapping sequence windows require probability aggregation.

## Data

FORCE 2020, DOI 10.5281/zenodo.4351156. Raw observations and trained weights
are excluded from source control by default.
