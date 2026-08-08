import numpy as np

from lithology_unet.data import split_wells
from lithology_unet.metrics import boundary_f1, classification_metrics
from lithology_unet.preprocessing import RobustWellLogPreprocessor
from lithology_unet.sequences import WellSequenceDataset
from lithology_unet.synthetic import make_synthetic_force_data


FEATURES = ["GR", "RHOB", "NPHI", "RDEP", "DTC", "PEF"]
TARGET = "FORCE_2020_LITHOFACIES_LITHOLOGY"


def test_grouped_split_has_no_overlap():
    df = make_synthetic_force_data(n_wells=24, rows_per_well=256)
    s = split_wells(df, seed=7)
    assert not (set(s.train) & set(s.validation))
    assert not (set(s.train) & set(s.test))
    assert not (set(s.validation) & set(s.test))


def test_sequence_shapes_and_missing_masks():
    df = make_synthetic_force_data(n_wells=6, rows_per_well=65)
    p = RobustWellLogPreprocessor(FEATURES).fit(df)
    ds = WellSequenceDataset(df, p, TARGET, "WELL", "DEPTH_MD", length=32, stride=16)
    x, y, depth, well = ds[0]
    assert x.shape == (12, 32)
    assert y.shape == depth.shape == (32,)
    assert np.isfinite(x.numpy()).all()
    assert isinstance(well, str)


def test_metrics_perfect_case():
    y = np.array([0, 0, 1, 1])
    p = np.eye(2)[y] * .98 + .01
    assert classification_metrics(y, p)["macro_f1"] == 1.0
    assert classification_metrics(y, p)["accuracy"] == 1.0
    assert boundary_f1(y, y) == 1.0
