from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from lithology_unet.constants import LITHOLOGY_CODE_TO_NAME
from lithology_unet.data import encode_force_labels, select_wells
from lithology_unet.feature_engineering import build_tabular_features
from lithology_unet.metrics import classification_metrics, confusion_and_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("artifact_dir")
    args = parser.parse_args()
    artifact = Path(args.artifact_dir)
    split = json.loads((artifact / "well_splits.json").read_text(encoding="utf-8"))
    codes = [30000, 65000, 70000]
    mapping = {code: i for i, code in enumerate(codes)}
    df = pd.read_csv(args.data)
    df = df[df.FORCE_2020_LITHOFACIES_LITHOLOGY.isin(codes)].copy()
    test = select_wells(df, split["test"])
    test = build_tabular_features(test)
    prep = joblib.load(artifact / "preprocessor.joblib")
    model = joblib.load(artifact / "xgboost.joblib")
    y = encode_force_labels(test.FORCE_2020_LITHOFACIES_LITHOLOGY, mapping)
    probability = model.predict_proba(prep.transform(test))
    metrics = classification_metrics(y, probability)
    metrics.update({"test_wells": len(split["test"]), "test_rows": len(test)})
    _, report = confusion_and_report(y, probability.argmax(1), len(codes))
    named = {LITHOLOGY_CODE_TO_NAME[code]: report[i] for i, code in enumerate(codes)}
    (artifact / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (artifact / "classification_report.json").write_text(json.dumps(named, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
