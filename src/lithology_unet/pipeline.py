from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .baseline import train_xgboost
from .config import load_config
from .constants import LITHOLOGY_CODES, LITHOLOGY_CODE_TO_NAME
from .data import load_tables, select_wells, split_wells, validate_schema
from .evaluation import collect_predictions, evaluate_by_well, plot_well_track
from .feature_engineering import build_tabular_features
from .models import ProbabilisticUNet1D, UNet1D
from .preprocessing import RobustWellLogPreprocessor
from .sequences import WellSequenceDataset
from .training import seed_everything, train_model


def run(data_paths: list[str], config_path: str, output: str, model_kind: str = "unet"):
    cfg = load_config(config_path); seed_everything(cfg["seed"])
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    df = load_tables(data_paths)
    validate_schema(df, cfg["features"], cfg["target"], cfg["well_column"], cfg["depth_column"])
    codes = tuple(cfg.get("class_codes", LITHOLOGY_CODES))
    df = df[df[cfg["target"]].isin(codes)].copy()
    code_to_index = {int(code): i for i, code in enumerate(codes)}
    class_names = [LITHOLOGY_CODE_TO_NAME[int(code)] for code in codes]
    splits = split_wells(df, cfg["well_column"], cfg["validation_fraction"],
                         cfg["test_fraction"], cfg["seed"], cfg["target"])
    (out / "well_splits.json").write_text(json.dumps(splits.__dict__, indent=2), encoding="utf-8")
    train_df = select_wells(df, splits.train, cfg["well_column"])
    val_df = select_wells(df, splits.validation, cfg["well_column"])
    test_df = select_wells(df, splits.test, cfg["well_column"])
    if model_kind == "xgboost":
        train_tab = build_tabular_features(train_df, cfg["well_column"], cfg["depth_column"])
        test_tab = build_tabular_features(test_df, cfg["well_column"], cfg["depth_column"])
        excluded = {cfg["well_column"], cfg["target"], "FORCE_2020_LITHOFACIES_CONFIDENCE", "DEPT"}
        tabular_features = [c for c in train_tab.select_dtypes(include="number").columns if c not in excluded]
        prep = RobustWellLogPreprocessor(tabular_features).fit(train_tab)
        joblib.dump(prep, out / "preprocessor.joblib")
        (out / "feature_names.json").write_text(json.dumps(prep.output_features, indent=2), encoding="utf-8")
        return train_xgboost(train_tab, test_tab, prep, cfg["target"], out, cfg["seed"],
                             code_to_index=code_to_index, class_names=class_names)[1]
    prep = RobustWellLogPreprocessor(cfg["features"]).fit(train_df)
    joblib.dump(prep, out / "preprocessor.joblib")
    ds_args = dict(preprocessor=prep, target=cfg["target"], well_column=cfg["well_column"],
                   depth_column=cfg["depth_column"], length=cfg["sequence_length"],
                   stride=cfg["sequence_stride"], code_to_index=code_to_index)
    train_ds = WellSequenceDataset(train_df, **ds_args)
    val_ds = WellSequenceDataset(val_df, **ds_args)
    test_ds = WellSequenceDataset(test_df, **ds_args)
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                                             num_workers=cfg["num_workers"])
    in_channels = len(prep.output_features); n_classes = len(codes)
    probabilistic = model_kind == "probabilistic_unet"
    model = (ProbabilisticUNet1D(in_channels, n_classes, cfg["base_channels"], cfg["latent_dim"])
             if probabilistic else UNet1D(in_channels, n_classes, cfg["base_channels"]))
    labels = train_df[cfg["target"]].map(code_to_index).fillna(-100).to_numpy()
    train_model(model, loader(train_ds, True), loader(val_ds, False), labels, n_classes, out,
                cfg["epochs"], cfg["learning_rate"], cfg["weight_decay"], cfg["kl_weight"], probabilistic)
    predictions = collect_predictions(model, loader(test_ds, False),
                                      probabilistic_samples=12 if probabilistic else 0)
    metrics = evaluate_by_well(predictions, out, class_names)
    if predictions:
        well = sorted(predictions)[0]
        plot_well_track(well, predictions[well], out / "example_well_track.png")
    return metrics
