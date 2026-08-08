from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RobustWellLogPreprocessor:
    features: list[str]
    medians: np.ndarray | None = None
    centers: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "RobustWellLogPreprocessor":
        x = df[self.features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        self.medians = np.nanmedian(x, axis=0)
        filled = np.where(np.isnan(x), self.medians, x)
        self.centers = np.nanmedian(filled, axis=0)
        q25, q75 = np.nanpercentile(filled, [25, 75], axis=0)
        self.scales = np.where((q75 - q25) > 1e-8, q75 - q25, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.centers is None or self.scales is None:
            raise RuntimeError("Preprocessor must be fit on training wells first")
        raw = df[self.features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        missing = np.isnan(raw).astype(np.float32)
        filled = np.where(np.isnan(raw), self.medians, raw)
        scaled = np.clip((filled - self.centers) / self.scales, -10.0, 10.0)
        return np.concatenate([scaled.astype(np.float32), missing], axis=1)

    @property
    def output_features(self) -> list[str]:
        return [*self.features, *[f"{f}_MISSING" for f in self.features]]
