from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lithology_unet.config import load_config
from lithology_unet.constants import LITHOLOGY_CODE_TO_NAME
from lithology_unet.data import encode_force_labels, select_wells
from lithology_unet.evaluation import collect_predictions
from lithology_unet.feature_engineering import build_tabular_features
from lithology_unet.metrics import classification_metrics, confusion_and_report
from lithology_unet.models import ProbabilisticUNet1D
from lithology_unet.sequences import WellSequenceDataset


def neural_predictions(frame, prep, model, cfg):
    mapping = {30000: 0, 65000: 1, 70000: 2}
    ds = WellSequenceDataset(frame, prep, cfg["target"], cfg["well_column"],
        cfg["depth_column"], cfg["sequence_length"], cfg["sequence_stride"], mapping)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    torch.manual_seed(cfg["seed"])
    return collect_predictions(model, loader, device=torch.device("cpu"), probabilistic_samples=12)


def flatten_aligned(frame, neural, xgb_prep, xgb_model):
    mapping = {30000: 0, 65000: 1, 70000: 2}
    tab = build_tabular_features(frame)
    tab["_ORDER"] = np.arange(len(tab))
    xgb_probability = xgb_model.predict_proba(xgb_prep.transform(tab))
    lookup = {(str(w), round(float(d), 3)): p for w, d, p in zip(tab.WELL, tab.DEPTH_MD, xgb_probability)}
    ys, pn, px = [], [], []
    for well in sorted(neural):
        depth, y, probability = neural[well]
        for d, label, npb in zip(depth, y, probability):
            key = (str(well), round(float(d), 3))
            if key in lookup:
                ys.append(label); pn.append(npb); px.append(lookup[key])
    return np.asarray(ys), np.asarray(px), np.asarray(pn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--config", default="configs/three_class.yaml")
    parser.add_argument("--xgb", default="artifacts/three_class_xgboost")
    parser.add_argument("--prob", default="artifacts/three_class_probabilistic_unet")
    parser.add_argument("--output", default="artifacts/three_class_ensemble")
    args = parser.parse_args()
    cfg = load_config(args.config); codes = [30000, 65000, 70000]
    df = pd.read_csv(args.data); df = df[df[cfg["target"]].isin(codes)].copy()
    split = json.loads((Path(args.xgb) / "well_splits.json").read_text(encoding="utf-8"))
    validation = select_wells(df, split["validation"]); test = select_wells(df, split["test"])
    xgb_prep = joblib.load(Path(args.xgb) / "preprocessor.joblib")
    xgb_model = joblib.load(Path(args.xgb) / "xgboost.joblib")
    neural_prep = joblib.load(Path(args.prob) / "preprocessor.joblib")
    model = ProbabilisticUNet1D(len(neural_prep.output_features), 3,
                               cfg["base_channels"], cfg["latent_dim"])
    model.load_state_dict(torch.load(Path(args.prob) / "best_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    val_neural = neural_predictions(validation, neural_prep, model, cfg)
    yv, xv, nv = flatten_aligned(validation, val_neural, xgb_prep, xgb_model)
    trials = []
    for alpha in np.linspace(0, 1, 21):
        metric = classification_metrics(yv, alpha * xv + (1 - alpha) * nv)
        trials.append({"xgboost_weight": float(alpha), **metric})
    best = max(trials, key=lambda row: (row["macro_f1"], row["accuracy"]))
    alpha = best["xgboost_weight"]
    test_neural = neural_predictions(test, neural_prep, model, cfg)
    yt, xt, nt = flatten_aligned(test, test_neural, xgb_prep, xgb_model)
    probability = alpha * xt + (1 - alpha) * nt
    metrics = classification_metrics(yt, probability)
    metrics.update({"xgboost_weight": alpha, "probabilistic_unet_weight": 1 - alpha,
                    "validation_macro_f1": best["macro_f1"], "test_rows": len(yt),
                    "test_wells": len(split["test"])})
    _, report = confusion_and_report(yt, probability.argmax(1), 3)
    named = {LITHOLOGY_CODE_TO_NAME[code]: report[i] for i, code in enumerate(codes)}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "blend_trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "classification_report.json").write_text(json.dumps(named, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
