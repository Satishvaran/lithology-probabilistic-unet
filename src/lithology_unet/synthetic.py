from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import LITHOLOGY_CODES


def make_synthetic_force_data(n_wells: int = 12, rows_per_well: int = 512,
                              seed: int = 42) -> pd.DataFrame:
    """Non-confidential data for CI and demonstrations; not a geological simulator."""
    rng = np.random.default_rng(seed)
    rows = []
    means = np.array([
        [55, 2.30, .18, 25, 90, 2.0], [80, 2.42, .24, 8, 105, 2.5],
        [110, 2.48, .30, 4, 125, 3.0], [90, 2.55, .22, 7, 110, 3.5],
        [35, 2.78, .08, 120, 70, 4.0], [25, 2.70, .06, 200, 65, 5.0],
        [20, 2.62, .05, 180, 60, 4.5], [15, 2.10, .02, 300, 55, 1.5],
        [18, 2.95, .01, 250, 58, 2.0], [65, 2.50, .15, 15, 95, 2.5],
        [130, 1.60, .40, 80, 140, 1.0], [45, 2.75, .10, 50, 80, 3.0],
    ])
    for w in range(n_wells):
        depth = 1000 + np.arange(rows_per_well) * 0.1524
        labels = np.empty(rows_per_well, int)
        cursor = 0
        while cursor < rows_per_well:
            cls = int(rng.choice(len(LITHOLOGY_CODES), p=np.array([.22,.08,.34,.08,.03,.09,.03,.01,.01,.02,.06,.03])))
            run = int(rng.integers(20, 100))
            labels[cursor:cursor + run] = cls
            cursor += run
        signal = means[labels] + rng.normal(0, [8,.04,.025,8,5,.4], (rows_per_well, 6))
        missing = rng.random(signal.shape) < 0.04
        signal[missing] = np.nan
        frame = pd.DataFrame(signal, columns=["GR", "RHOB", "NPHI", "RDEP", "DTC", "PEF"])
        frame.insert(0, "DEPTH_MD", depth)
        frame.insert(0, "WELL", f"SYNTH_{w:03d}")
        frame["FORCE_2020_LITHOFACIES_LITHOLOGY"] = np.asarray(LITHOLOGY_CODES)[labels]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)
