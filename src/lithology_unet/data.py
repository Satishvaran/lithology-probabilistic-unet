from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import CODE_TO_INDEX


@dataclass(frozen=True)
class WellSplits:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def read_force_table(path: str | Path) -> pd.DataFrame:
    """Read a FORCE CSV/Parquet table without mutating label values."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def convert_force_las_directory(source: str | Path, output: str | Path) -> Path:
    """Convert public FORCE LAS files to one ML table; NULL values become NaN."""
    source, output = Path(source), Path(output)
    frames = []
    for path in sorted(source.rglob("*.las")):
        frame, well = read_las2(path)
        frame.insert(0, "WELL", well or path.stem)
        if "DEPTH_MD" not in frame and "DEPT" in frame:
            frame["DEPTH_MD"] = frame["DEPT"]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No LAS files beneath {source}")
    data = pd.concat(frames, ignore_index=True, sort=False)
    target = "FORCE_2020_LITHOFACIES_LITHOLOGY"
    if target in data:
        data = data[data[target].notna()].copy()
        data[target] = data[target].astype(int)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        data.to_parquet(output, index=False)
    else:
        data.to_csv(output, index=False)
    return output


def read_las2(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Minimal whitespace LAS 2.0 reader for the unwrapped public FORCE files."""
    path = Path(path)
    curves, rows, section, well = [], [], "", path.stem
    null_value = -999.25
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("~"):
                section = line[1:].split()[0].upper()
                continue
            if section.startswith("WELL"):
                mnemonic = line.split(".", 1)[0].strip().upper()
                before_description = line.split(":", 1)[0]
                value = before_description.split(None, 1)[1].strip() if len(before_description.split(None, 1)) > 1 else ""
                if mnemonic == "WELL" and value:
                    well = value
                elif mnemonic == "NULL" and value:
                    try: null_value = float(value)
                    except ValueError: pass
            elif section.startswith("CURVE"):
                curves.append(line.split(".", 1)[0].strip().upper())
            elif section.startswith("ASCII"):
                values = np.fromstring(line, sep=" ")
                if len(values) == len(curves):
                    rows.append(values)
    if not curves or not rows:
        raise ValueError(f"Could not parse LAS curves/data from {path}")
    data = np.vstack(rows)
    data[np.isclose(data, null_value)] = np.nan
    return pd.DataFrame(data, columns=curves), well


def discover_tables(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted([*directory.rglob("*.csv"), *directory.rglob("*.parquet")])


def load_tables(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [read_force_table(p) for p in paths]
    if not frames:
        raise FileNotFoundError("No CSV or Parquet tables were found")
    return pd.concat(frames, ignore_index=True, sort=False)


def validate_schema(df: pd.DataFrame, features: list[str], target: str,
                    well_column: str, depth_column: str) -> None:
    required = {well_column, depth_column, target, *features}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[well_column].isna().any():
        raise ValueError("Well identifiers contain missing values")


def encode_force_labels(series: pd.Series, code_to_index: dict[int, int] | None = None) -> np.ndarray:
    mapping = code_to_index or CODE_TO_INDEX
    unknown = sorted(set(series.dropna().astype(int)) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown FORCE lithology codes: {unknown}")
    return series.map(mapping).fillna(-100).astype(np.int64).to_numpy()


def split_wells(df: pd.DataFrame, well_column: str = "WELL",
                validation_fraction: float = 0.15, test_fraction: float = 0.15,
                seed: int = 42,
                target: str = "FORCE_2020_LITHOFACIES_LITHOLOGY") -> WellSplits:
    """Split whole wells. Rows from a well can never cross partitions."""
    wells = np.asarray(sorted(df[well_column].astype(str).unique()))
    if len(wells) < 5:
        raise ValueError("At least five wells are required for grouped evaluation")
    all_classes = set(df[target].dropna().astype(int).unique()) if target in df else set()
    # Retry deterministic grouped splits until every observed class remains in
    # training. This prevents a rare test-only class from breaking the model.
    for attempt in range(100):
        rng = np.random.default_rng(seed + attempt)
        shuffled = wells[rng.permutation(len(wells))]
        n_test = max(1, int(round(len(wells) * test_fraction)))
        n_val = max(1, int(round(len(wells) * validation_fraction)))
        test = shuffled[:n_test]
        validation = shuffled[n_test:n_test + n_val]
        train = shuffled[n_test + n_val:]
        candidate = WellSplits(tuple(train), tuple(validation), tuple(test))
        if not all_classes:
            return candidate
        train_classes = set(select_wells(df, candidate.train, well_column)[target].dropna().astype(int).unique())
        if train_classes == all_classes:
            return candidate
    raise RuntimeError("Could not create a grouped split containing every class in training")


def select_wells(df: pd.DataFrame, wells: Iterable[str], well_column: str = "WELL") -> pd.DataFrame:
    allowed = set(map(str, wells))
    return df[df[well_column].astype(str).isin(allowed)].copy()
