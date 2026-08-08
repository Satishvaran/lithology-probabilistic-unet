from __future__ import annotations

import argparse
from pathlib import Path

from .synthetic import make_synthetic_force_data
from .data import convert_force_las_directory


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe FORCE lithology benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    synth = sub.add_parser("make-synthetic")
    synth.add_argument("--output", default="data/synthetic/force_like.csv")
    synth.add_argument("--wells", type=int, default=12)
    prepare = sub.add_parser("prepare-las")
    prepare.add_argument("source")
    prepare.add_argument("--output", default="data/processed/force2020.csv")
    train = sub.add_parser("train")
    train.add_argument("data", nargs="+")
    train.add_argument("--config", default="configs/default.yaml")
    train.add_argument("--output", default="artifacts/run")
    train.add_argument("--model", choices=["xgboost", "unet", "probabilistic_unet"], default="unet")
    args = parser.parse_args()
    if args.command == "make-synthetic":
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_force_data(args.wells).to_csv(path, index=False)
        print(path.resolve())
    elif args.command == "prepare-las":
        print(convert_force_las_directory(args.source, args.output).resolve())
    else:
        from .pipeline import run
        print(run(args.data, args.config, args.output, args.model))


if __name__ == "__main__":
    main()
