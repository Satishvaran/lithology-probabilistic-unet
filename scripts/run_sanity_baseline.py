"""Dependency-light Gaussian baseline used to validate the real-data harness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithology_unet.constants import CODE_TO_INDEX, LITHOLOGY_CODES
from lithology_unet.data import select_wells, split_wells
from lithology_unet.metrics import classification_metrics
from lithology_unet.preprocessing import RobustWellLogPreprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--output", default="reports/sanity_gaussian_metrics.json")
    args = parser.parse_args()
    features = ["GR", "RHOB", "NPHI", "RDEP", "DTC", "PEF"]
    target = "FORCE_2020_LITHOFACIES_LITHOLOGY"
    df = pd.read_csv(args.data, usecols=["WELL", "DEPTH_MD", target, *features])
    split = split_wells(df, target=target)
    train, test = select_wells(df, split.train), select_wells(df, split.test)
    prep = RobustWellLogPreprocessor(features).fit(train)
    x_train, x_test = prep.transform(train), prep.transform(test)
    y_train = train[target].map(CODE_TO_INDEX).astype(int).to_numpy()
    y_test = test[target].map(CODE_TO_INDEX).astype(int).to_numpy()
    means, variances, priors = [], [], []
    for cls in range(len(LITHOLOGY_CODES)):
        values = x_train[y_train == cls]
        means.append(values.mean(0)); variances.append(values.var(0) + 1e-3)
        priors.append(max(len(values) / len(x_train), 1e-12))
    means, variances = np.asarray(means), np.asarray(variances)
    logp = []
    for cls in range(len(LITHOLOGY_CODES)):
        score = -.5 * (np.log(2 * np.pi * variances[cls]) +
                       (x_test - means[cls]) ** 2 / variances[cls]).sum(1) + np.log(priors[cls])
        logp.append(score)
    logp = np.stack(logp, axis=1)
    logp -= logp.max(1, keepdims=True)
    probability = np.exp(logp); probability /= probability.sum(1, keepdims=True)
    result = classification_metrics(y_test, probability)
    result.update({"model": "diagonal_gaussian_sanity_baseline", "test_wells": len(split.test),
                   "test_rows": len(test), "warning": "Harness sanity check; not a competitive model."})
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
