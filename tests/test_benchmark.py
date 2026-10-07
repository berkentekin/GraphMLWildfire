import numpy as np
import torch
from sklearn.metrics import average_precision_score

from graphml_wildfire.benchmark import average_precision
from graphml_wildfire.data.patches import CENTRE_CROP, CHANNEL_INDEX, CHANNELS, benchmark_crop, burning_pixels, node_targets
from graphml_wildfire.data.splits import create_year_folds
from graphml_wildfire.models import ConvNet, NodeMLP, Persistence
from graphml_wildfire.train import transform


def test_folds_follow_the_official_order():
    folds = [(*fold.train_years, fold.validation_year, fold.test_year) for fold in create_year_folds()]
    assert folds[:3] == [(2018, 2019, 2020, 2021), (2018, 2019, 2021, 2020), (2018, 2020, 2019, 2021)]
    assert folds[-1] == (2020, 2021, 2019, 2018)
    assert len(set(folds)) == 12


def test_detections_before_one_am_do_not_burn():
    assert burning_pixels(np.array([np.nan, 0, 59, 100, 2359])).tolist() == [False, False, False, True, True]


def test_node_targets_count_every_pixel_of_the_benchmark_crop():
    rng = np.random.default_rng(0)
    data = np.full((3, 23, 300, 250), np.nan, dtype=np.float32)
    data[:, 22] = np.where(rng.random((3, 300, 250)) < 0.1, 1200, np.nan)
    rows, cols = benchmark_crop(300, 250)
    targets = node_targets(data, rows, cols, patch=4)
    assert (rows.stop - rows.start, cols.stop - cols.start) == (288, 224)
    assert targets["fire_pixels"].sum() == burning_pixels(data[:, 22, rows, cols]).sum()
    assert targets["centre_pixels"].sum() == CENTRE_CROP**2
    assert targets["centre_fire"].sum() <= targets["fire_pixels"].sum()


def test_average_precision_matches_sklearn_with_ties():
    torch.manual_seed(0)
    scores = torch.randint(0, 20, (5000,)).float()
    targets = (torch.rand(5000) < 0.1 + scores / 100).float()
    expected = average_precision_score(targets.numpy(), scores.numpy())
    assert abs(average_precision(scores, targets) - expected) < 1e-9


def test_count_weighted_average_precision_equals_pixel_level_average_precision():
    """A node scored once for all its pixels gives the AP of scoring every pixel separately."""

    torch.manual_seed(0)
    scores = torch.randn(400)
    totals = torch.randint(0, 17, (400,)).float()
    positives = torch.floor(totals * torch.rand(400) * (scores > 0))
    pixel_scores = torch.repeat_interleave(scores, totals.long())
    pixel_targets = torch.cat([
        torch.cat([torch.ones(int(fire)), torch.zeros(int(total - fire))]) for fire, total in zip(positives, totals)
    ])
    expected = average_precision_score(pixel_targets.numpy(), pixel_scores.numpy())
    assert abs(average_precision(scores, positives, totals) - expected) < 1e-9


def test_baselines_return_one_logit_per_node():
    features = torch.rand(2, 3, len(CHANNELS), 8, 12)
    for model in (ConvNet(len(CHANNELS), 3, hidden=16), NodeMLP(len(CHANNELS), 3, hidden=16),
                  NodeMLP(len(CHANNELS), 3, layers=0), Persistence()):
        assert model(features).shape == (2, 8, 12)


def test_augmentations_rotate_direction_vectors_with_the_grid():
    """Transposing the grid swaps the row and column components of every direction."""

    features = torch.zeros(1, 1, len(CHANNELS), 4, 4)
    for name in ("wind", "aspect", "fc_wind"):
        features[:, :, CHANNEL_INDEX[f"{name}_row"]] = 1.0
    valid = target = torch.ones(1, 4, 4)
    moved, _, _ = transform(features.clone(), valid, target, flip_rows=True, flip_cols=False, transpose=True)
    for name in ("wind", "aspect", "fc_wind"):
        assert torch.all(moved[:, :, CHANNEL_INDEX[f"{name}_row"]] == 0)
        assert torch.all(moved[:, :, CHANNEL_INDEX[f"{name}_col"]] == -1)
