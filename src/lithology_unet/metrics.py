from __future__ import annotations

import numpy as np


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray,
                               bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs((prediction[mask] == labels[mask]).mean() - confidence[mask].mean())
    return float(ece)


def boundary_f1(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 3) -> float:
    true = np.flatnonzero(y_true[1:] != y_true[:-1]) + 1
    pred = np.flatnonzero(y_pred[1:] != y_pred[:-1]) + 1
    if len(true) == len(pred) == 0:
        return 1.0
    if len(true) == 0 or len(pred) == 0:
        return 0.0
    used = set()
    tp = 0
    for p in pred:
        candidates = [(abs(int(t) - int(p)), i) for i, t in enumerate(true) if i not in used]
        if candidates:
            distance, i = min(candidates)
            if distance <= tolerance:
                used.add(i); tp += 1
    precision = tp / len(pred)
    recall = tp / len(true)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def classification_metrics(y_true, probabilities) -> dict[str, float]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    pred = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    f1s, recalls, supports = [], [], []
    for cls in range(n_classes):
        tp = np.sum((pred == cls) & (y_true == cls))
        fp = np.sum((pred == cls) & (y_true != cls))
        fn = np.sum((pred != cls) & (y_true == cls))
        support = np.sum(y_true == cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            f1s.append(f1); recalls.append(recall); supports.append(support)
    chosen = np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-15, 1.0)
    return {
        "accuracy": float(np.mean(pred == y_true)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.average(f1s, weights=supports)),
        "balanced_accuracy": float(np.mean(recalls)),
        "negative_log_likelihood": float(-np.log(chosen).mean()),
        "expected_calibration_error": expected_calibration_error(probabilities, y_true),
    }


def confusion_and_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    report = {}
    for cls in range(num_classes):
        tp, support, predicted = cm[cls, cls], cm[cls].sum(), cm[:, cls].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[cls] = {"precision": float(precision), "recall": float(recall),
                       "f1-score": float(f1), "support": int(support)}
    return cm, report
