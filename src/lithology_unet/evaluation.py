from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from .constants import IGNORE_INDEX, INDEX_TO_NAME
from .metrics import boundary_f1, classification_metrics, confusion_and_report


@torch.no_grad()
def collect_predictions(model, loader, device=None, probabilistic_samples: int = 0):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    buckets = defaultdict(list)
    for x, y, depth, wells in loader:
        x = x.to(device)
        if probabilistic_samples:
            logits = model.sample(x, probabilistic_samples).softmax(2).mean(0)
        else:
            result = model(x)
            logits = (result[0] if isinstance(result, tuple) else result).softmax(1)
        probs = logits.cpu().numpy().transpose(0, 2, 1)
        for i, well in enumerate(wells):
            valid = y[i].numpy() != IGNORE_INDEX
            buckets[str(well)].append((depth[i].numpy()[valid], y[i].numpy()[valid], probs[i][valid]))
    # Overlapping windows are averaged at each well/depth.
    output = {}
    for well, chunks in buckets.items():
        frame = pd.concat([
            pd.DataFrame({"depth": d, "label": y, **{f"p_{j}": p[:, j] for j in range(p.shape[1])}})
            for d, y, p in chunks
        ])
        prob_cols = [c for c in frame if c.startswith("p_")]
        grouped = frame.groupby("depth", as_index=False).agg({"label": "first", **{c: "mean" for c in prob_cols}})
        output[well] = (grouped.depth.to_numpy(), grouped.label.to_numpy(), grouped[prob_cols].to_numpy())
    return output


def evaluate_by_well(predictions: dict, output_dir: str | Path,
                     class_names: list[str] | None = None) -> dict:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows, all_y, all_p = [], [], []
    for well, (depth, y, p) in predictions.items():
        metric = classification_metrics(y, p)
        metric["boundary_f1_tolerance_3_samples"] = boundary_f1(y, p.argmax(1), 3)
        metric["well"] = well; metric["samples"] = len(y)
        rows.append(metric); all_y.append(y); all_p.append(p)
    y, p = np.concatenate(all_y), np.concatenate(all_p)
    pooled = classification_metrics(y, p)
    pd.DataFrame(rows).to_csv(output_dir / "metrics_by_well.csv", index=False)
    cm, numeric_report = confusion_and_report(y, p.argmax(1), p.shape[1])
    class_names = class_names or [INDEX_TO_NAME[i] for i in range(p.shape[1])]
    report = {class_names[i]: numeric_report[i] for i in range(p.shape[1])}
    (output_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "pooled_metrics.json").write_text(json.dumps(pooled, indent=2), encoding="utf-8")
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, cmap="Blues", vmin=0, vmax=1, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set(xlabel="Predicted", ylabel="True", title="Well-held-out normalized confusion matrix")
    fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=180); plt.close(fig)
    return pooled


def plot_well_track(well: str, prediction: tuple, output: str | Path) -> None:
    depth, y, p = prediction; pred = p.argmax(1); confidence = p.max(1)
    fig, axes = plt.subplots(1, 3, figsize=(9, 10), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 2]})
    axes[0].imshow(y[:, None], aspect="auto", interpolation="nearest", origin="upper")
    axes[0].set_title("True")
    axes[1].imshow(pred[:, None], aspect="auto", interpolation="nearest", origin="upper")
    axes[1].set_title("Predicted")
    axes[2].plot(confidence, depth, color="#136f63")
    axes[2].fill_betweenx(depth, 0, confidence, color="#9fd8cb")
    axes[2].set(xlim=(0, 1), xlabel="Confidence", title="Model confidence")
    axes[0].set_ylabel("Sample index / depth order")
    fig.suptitle(f"Held-out well: {well}"); fig.tight_layout()
    fig.savefig(output, dpi=180); plt.close(fig)
