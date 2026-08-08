from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lithology_unet.data import split_wells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--output", default="reports/frozen_well_split.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target = "FORCE_2020_LITHOFACIES_LITHOLOGY"
    df = pd.read_csv(args.data, usecols=["WELL", target])
    split = split_wells(df, seed=args.seed, target=target)
    payload = {"seed": args.seed, "train": split.train,
               "validation": split.validation, "test": split.test}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"train={len(split.train)} validation={len(split.validation)} test={len(split.test)}")


if __name__ == "__main__":
    main()
