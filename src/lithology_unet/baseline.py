from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

from .constants import INDEX_TO_NAME
from .data import encode_force_labels
from .metrics import classification_metrics, confusion_and_report


def train_xgboost(train_df, test_df, preprocessor, target: str, output: str | Path,
                  seed: int = 42, max_rows: int = 500_000,
                  code_to_index: dict[int, int] | None = None,
                  class_names: list[str] | None = None):
    """Transparent row baseline; evaluation wells must be held out upstream."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_df))
    if len(idx) > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    x_train = preprocessor.transform(train_df.iloc[idx])
    y_train = encode_force_labels(train_df.iloc[idx][target], code_to_index)
    valid = y_train >= 0
    model = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=.06, subsample=.8,
        colsample_bytree=.8, objective="multi:softprob", eval_metric="mlogloss",
        tree_method="hist", n_jobs=-1, random_state=seed,
    )
    model.fit(x_train[valid], y_train[valid])
    x_test = preprocessor.transform(test_df)
    y_test = encode_force_labels(test_df[target], code_to_index); mask = y_test >= 0
    probabilities = model.predict_proba(x_test[mask])
    metrics = classification_metrics(y_test[mask], probabilities)
    _, numeric_report = confusion_and_report(y_test[mask], probabilities.argmax(1), probabilities.shape[1])
    class_names = class_names or [INDEX_TO_NAME[i] for i in range(probabilities.shape[1])]
    report = {class_names[i]: numeric_report[i] for i in range(probabilities.shape[1])}
    joblib.dump(model, output / "xgboost.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model, metrics
