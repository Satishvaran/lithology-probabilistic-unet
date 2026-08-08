from __future__ import annotations

import numpy as np
import pandas as pd


def build_tabular_features(df: pd.DataFrame, well_column: str = "WELL",
                           depth_column: str = "DEPTH_MD") -> pd.DataFrame:
    """Context features computed strictly within each well, never across wells."""
    out = df.sort_values([well_column, depth_column]).copy()
    eps = 1e-6
    if {"CALI", "BS"}.issubset(out):
        out["CALI_BS_DIFF"] = out.CALI - out.BS
        out["IS_WASHOUT"] = (out.CALI_BS_DIFF > 2).astype(float)
    if {"RHOB", "NPHI"}.issubset(out):
        out["RHOB_NPHI_DIFF"] = out.RHOB - out.NPHI
    if {"RDEP", "RMED"}.issubset(out):
        out["RES_RATIO"] = out.RDEP / (out.RMED + eps)
        out["LOG_RDEP"] = np.log1p(out.RDEP.clip(lower=0))
        out["LOG_RMED"] = np.log1p(out.RMED.clip(lower=0))
    if {"DTC", "RHOB"}.issubset(out):
        out["DTC_RHOB_PRODUCT"] = out.DTC * out.RHOB
    if {"PEF", "RHOB"}.issubset(out):
        out["PEF_RHOB_RATIO"] = out.PEF / (out.RHOB + eps)
    grouped = out.groupby(well_column, sort=False)
    well_min = grouped[depth_column].transform("min")
    span = grouped[depth_column].transform("max") - well_min
    out["DEPTH_REL"] = (out[depth_column] - well_min) / span.replace(0, np.nan)
    # Centred context is appropriate for post-drilling lithology classification.
    # It must not be described as causal real-time prediction.
    context_logs = [c for c in ["GR", "RHOB", "NPHI", "DTC", "PEF", "CALI", "RDEP"] if c in out]
    for col in context_logs:
        out[f"{col}_GRAD"] = grouped[col].diff()
        for window in (5, 15, 31):
            roll = grouped[col].rolling(window, center=True, min_periods=1)
            out[f"{col}_MEAN_{window}"] = roll.mean().reset_index(level=0, drop=True)
            out[f"{col}_STD_{window}"] = roll.std().reset_index(level=0, drop=True)
    return out
