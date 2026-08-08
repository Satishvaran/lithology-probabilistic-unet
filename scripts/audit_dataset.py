from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lithology_unet.constants import LITHOLOGY_CODE_TO_NAME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--output", default="reports/dataset_audit.json")
    args = parser.parse_args()
    columns = ["WELL", "DEPTH_MD", "GR", "RHOB", "NPHI", "RDEP", "DTC", "PEF",
               "FORCE_2020_LITHOFACIES_LITHOLOGY", "FORCE_2020_LITHOFACIES_CONFIDENCE"]
    df = pd.read_csv(args.data, usecols=columns)
    counts = df["FORCE_2020_LITHOFACIES_LITHOLOGY"].value_counts().sort_index()
    audit = {
        "rows": int(len(df)),
        "wells": int(df.WELL.nunique()),
        "depth_min_m": float(df.DEPTH_MD.min()),
        "depth_max_m": float(df.DEPTH_MD.max()),
        "class_counts": {LITHOLOGY_CODE_TO_NAME[int(k)]: int(v) for k, v in counts.items()},
        "missing_fraction": {c: float(df[c].isna().mean()) for c in ["GR", "RHOB", "NPHI", "RDEP", "DTC", "PEF"]},
        "confidence_counts": {str(k): int(v) for k, v in df["FORCE_2020_LITHOFACIES_CONFIDENCE"].value_counts(dropna=False).items()},
    }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
