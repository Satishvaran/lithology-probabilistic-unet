from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import IGNORE_INDEX
from .data import encode_force_labels
from .preprocessing import RobustWellLogPreprocessor


class WellSequenceDataset(Dataset):
    """Fixed-length windows that never cross well boundaries."""

    def __init__(self, df: pd.DataFrame, preprocessor: RobustWellLogPreprocessor,
                 target: str, well_column: str, depth_column: str,
                 length: int = 256, stride: int = 128,
                 code_to_index: dict[int, int] | None = None):
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
        for well, frame in df.groupby(well_column, sort=True):
            frame = frame.sort_values(depth_column).reset_index(drop=True)
            x = preprocessor.transform(frame)
            y = encode_force_labels(frame[target], code_to_index)
            depth = frame[depth_column].to_numpy(np.float32)
            starts = list(range(0, max(1, len(frame) - length + 1), stride))
            last = max(0, len(frame) - length)
            if not starts or starts[-1] != last:
                starts.append(last)
            for start in sorted(set(starts)):
                stop = min(start + length, len(frame))
                valid = stop - start
                px = np.zeros((length, x.shape[1]), np.float32)
                py = np.full(length, IGNORE_INDEX, np.int64)
                pdp = np.full(length, np.nan, np.float32)
                px[:valid], py[:valid], pdp[:valid] = x[start:stop], y[start:stop], depth[start:stop]
                self.samples.append((px, py, pdp, str(well)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        x, y, depth, well = self.samples[index]
        return torch.from_numpy(x.T), torch.from_numpy(y), torch.from_numpy(depth), well
