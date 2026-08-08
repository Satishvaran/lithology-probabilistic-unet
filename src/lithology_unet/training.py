from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .constants import IGNORE_INDEX
from .models.probabilistic_unet1d import kl_normal


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    valid = labels[labels >= 0]
    counts = np.bincount(valid, minlength=num_classes).astype(float)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def run_epoch(model, loader: DataLoader, optimizer, device: torch.device,
              weights: torch.Tensor, probabilistic: bool, kl_weight: float = 0.01):
    training = optimizer is not None
    model.train(training)
    ce = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=IGNORE_INDEX)
    total_loss = total_rows = correct = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if probabilistic:
                logits, prior, posterior = model(x, y)
                loss = ce(logits, y) + kl_weight * kl_normal(*posterior, *prior)
            else:
                logits = model(x)
                loss = ce(logits, y)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            valid = y != IGNORE_INDEX
            n = int(valid.sum())
            total_loss += float(loss.detach()) * n
            correct += int((logits.argmax(1)[valid] == y[valid]).sum())
            total_rows += n
    return {"loss": total_loss / max(total_rows, 1), "accuracy": correct / max(total_rows, 1)}


def train_model(model, train_loader, validation_loader, train_labels, num_classes: int,
                output: str | Path, epochs: int = 25, learning_rate: float = 1e-3,
                weight_decay: float = 1e-4, kl_weight: float = 0.01,
                probabilistic: bool = False, patience: int = 6) -> list[dict]:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    weights = class_weights(np.asarray(train_labels), num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2)
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, epochs + 1):
        train = run_epoch(model, train_loader, optimizer, device, weights, probabilistic, kl_weight)
        val = run_epoch(model, validation_loader, None, device, weights, probabilistic, kl_weight)
        scheduler.step(val["loss"])
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
               **{f"validation_{k}": v for k, v in val.items()},
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row); print(json.dumps(row))
        if val["loss"] < best - 1e-5:
            best, stale = val["loss"], 0
            torch.save(model.state_dict(), output / "best_model.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.load_state_dict(torch.load(output / "best_model.pt", map_location=device, weights_only=True))
    return history
