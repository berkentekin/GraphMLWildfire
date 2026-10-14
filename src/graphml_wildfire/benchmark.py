"""The WildfireSpreadTS test protocols, their metrics and checks.

Both published protocols share the 12 year-wise folds, the test samples (target days 5..T-1 of
every test fire) and the metric: exact average precision pooled over every 375 m pixel of the test
year, one value per fold. They differ in which pixels are used:

* Gerard et al. (NeurIPS 2023): the whole image, centre-cropped to multiples of 32, is both the
  model input and the scored region. Reported as mean +- population std (ddof=0) over the folds.
* Lahrichi et al. (WACV 2026, the 0.478 state of the art): only the centre 128 x 128 pixels of the
  image are fed to the model and scored (torchvision `center_crop` rounding). Reported as mean
  +- sample std (ddof=1) over the folds; their per-fold values are in LAHRICHI_UTAE_RES18_VEG.

    uv run python -m graphml_wildfire.benchmark verify        # metric and window checks for reviewers
    uv run python -m graphml_wildfire.benchmark persistence   # reproduce Gerard et al., Table 2
    uv run python -m graphml_wildfire.benchmark summary       # mean +- std over folds per experiment
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from .data.channels import ACTIVE_FIRE_INDEX
from .data.convert_hdf5 import CENTRE_CROP, _centre_crop_start, _date_from_filename, benchmark_crop, burning_pixels
from .data.splits import create_wstsplus_folds, create_year_folds


YEARS = (2018, 2019, 2020, 2021)
# The 12 folds of WildfireSpreadTS, in the order of the official code.
FOLDS = create_year_folds(YEARS)
# WSTS+ (Lahrichi et al., 2026): the four extra years and the 4 folds of four training, two
# validation and two test years. Scored like WSTS: centre 128 x 128 window, AP pooled over the fold.
WSTSPLUS_YEARS = (2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023)
WSTSPLUS_FOLDS = create_wstsplus_folds()
PROTOCOLS = {"wsts": FOLDS, "wstsplus": WSTSPLUS_FOLDS}
# Every model is scored on the targets of the benchmark's 5-day models (target days 5..T-1),
# as with `n_leading_observations_test_adjustment: 5` in the official code.
FIRST_INPUT_DAY = 4

# Pixel-level persistence AP per test year, Gerard et al. (2023), Table 2.
PUBLISHED_PERSISTENCE = {2018: 0.187, 2019: 0.105, 2020: 0.194, 2021: 0.287}
# Per-fold centre-128 AP of the state of the art, UTAE(Res18) with Veg features, from the names of
# the released checkpoints (huggingface.co/saadlahrichi/WSTSPlus, Res18UTAE_T5/Veg/fold<k>_testAP<ap>.pth).
# Their published 0.478 +- 0.085 is the mean and sample std (ddof=1) of these values.
LAHRICHI_UTAE_RES18_VEG = [0.523, 0.443, 0.590, 0.345, 0.501, 0.348, 0.584, 0.507, 0.506, 0.512, 0.359, 0.516]
# Std convention of each leaderboard column, matching the paper the column is compared with.
STD_DDOF = {"test_ap": 0, "test_ap_centre128": 1, "test_f1": 1, "test_iou": 1}
# Lahrichi et al. (WACV 2026; arXiv 2502.12003v3), Table 2: 12-fold test AP (mean, sample std) on the
# centre 128 x 128 window, with the feature set of each entry. Veg = fire mask + VIIRS bands + NDVI/EVI2,
# Multi = Veg + land cover, topography, weather; All = all 23 channels.
PUBLISHED = [
    ("Res18-UNet", 1, "Veg", 0.455, 0.090, "14.3M"),
    ("Res18-UNet", 1, "Multi", 0.468, 0.087, "14.3M"),
    ("Res18-UNet", 1, "All", 0.460, 0.084, "14.3M"),
    ("Res50-UNet", 1, "Multi", 0.459, 0.090, "32.5M"),
    ("SwinUnet", 1, "Multi", 0.437, 0.082, "27.2M"),
    ("SegFormer-B2", 1, "Multi", 0.436, 0.083, "27.5M"),
    ("Res18-UNet, data-level fusion", 5, "Veg", 0.472, 0.083, "14.4M"),
    ("SwinUnet, data-level fusion", 5, "Multi", 0.453, 0.083, "27.3M"),
    ("SegFormer, data-level fusion", 5, "Multi", 0.439, 0.081, "27.7M"),
    ("UTAE", 5, "Multi", 0.459, 0.088, "1.1M"),
    ("UTAE(Res18)", 5, "Veg", 0.478, 0.085, "14.6M"),
    ("UTAE(Res18)", 5, "All", 0.475, 0.091, "14.6M"),
]
# Lahrichi et al., Table 5: 4-fold test AP (mean, std) on WSTS+, same feature sets.
PUBLISHED_WSTSPLUS = [
    ("Res18-UNet", 1, "Veg", 0.349, 0.109, "14.3M"),
    ("Res18-UNet", 1, "Multi", 0.351, 0.105, "14.3M"),
    ("Res18-UNet", 1, "All", 0.351, 0.122, "14.3M"),
    ("Res50-UNet", 1, "Veg", 0.345, 0.096, "32.5M"),
    ("Res50-UNet", 1, "Multi", 0.353, 0.122, "32.5M"),
    ("Res50-UNet", 1, "All", 0.351, 0.122, "32.5M"),
    ("UTAE(Res18)", 5, "Veg", 0.354, 0.113, "14.6M"),
    ("UTAE(Res18)", 5, "Multi", 0.363, 0.129, "14.6M"),
    ("UTAE(Res18)", 5, "All", 0.350, 0.117, "14.6M"),
]
LAHRICHI_URL = "https://arxiv.org/abs/2502.12003v3"


def average_precision(scores: torch.Tensor, positives: torch.Tensor, totals: torch.Tensor | None = None) -> float:
    """Exact area under the precision-recall curve, with tied scores sharing one threshold.

    Every score stands for `totals` pixels of which `positives` burn (one pixel each by default),
    which equals torchmetrics' BinaryAveragePrecision over the expanded pixels.
    """

    positives = positives.double()
    totals = torch.ones_like(positives) if totals is None else totals.double()
    keep = totals > 0
    scores, positives, totals = scores[keep].float(), positives[keep], totals[keep]
    order = torch.argsort(scores, descending=True)
    scores, positives, totals = scores[order], positives[order], totals[order]
    last_of_tie = torch.ones_like(scores, dtype=torch.bool)
    last_of_tie[:-1] = scores[1:] != scores[:-1]
    hits = torch.cumsum(positives, dim=0)[last_of_tie]
    predicted = torch.cumsum(totals, dim=0)[last_of_tie]
    recall = hits / hits[-1]
    precision = hits / predicted
    steps = recall - torch.cat([recall.new_zeros(1), recall[:-1]])
    return float((steps * precision).sum())


def threshold_metrics(scores: torch.Tensor, positives: torch.Tensor, totals: torch.Tensor, threshold: float) -> dict[str, float]:
    """Pixel-level F1 and IoU when every node scoring at least `threshold` is predicted burning."""

    predicted = scores >= threshold
    hit = positives[predicted].sum().item()
    false_alarm = totals[predicted].sum().item() - hit
    miss = positives.sum().item() - hit
    return {
        "f1": 2 * hit / max(2 * hit + false_alarm + miss, 1),
        "iou": hit / max(hit + false_alarm + miss, 1),
    }


def best_f1_threshold(scores: torch.Tensor, positives: torch.Tensor, totals: torch.Tensor) -> float:
    candidates = torch.quantile(scores[positives > 0].float(), torch.linspace(0.02, 0.98, 49, device=scores.device))
    f1 = [threshold_metrics(scores, positives, totals, float(value))["f1"] for value in candidates]
    return float(candidates[int(np.argmax(f1))])


def pixel_persistence_ap(raw_dir: str | Path, year: int) -> float:
    """AP of predicting tomorrow's fire mask with today's, on the benchmark's test pixels.

    Today's mask is binary, so every pixel falls into one of four (score, target) cells and the
    exact AP follows from their counts.
    """

    counts = torch.zeros(2, 2, dtype=torch.float64)  # [score, target]
    fire_dirs = sorted(path for path in Path(raw_dir, str(year)).iterdir() if path.is_dir())
    for fire_dir in tqdm(fire_dirs, desc=str(year), unit="fire"):
        days = []
        for path in sorted(fire_dir.glob("*.tif"), key=_date_from_filename):
            with rasterio.open(path) as source:
                days.append(source.read(ACTIVE_FIRE_INDEX + 1))
        active_fire = np.stack(days)
        rows, cols = benchmark_crop(*active_fire.shape[-2:])
        burning = burning_pixels(active_fire[:, rows, cols])
        today, tomorrow = burning[FIRST_INPUT_DAY:-1], burning[FIRST_INPUT_DAY + 1 :]
        for score in (0, 1):
            for target in (0, 1):
                counts[score, target] += float(((today == score) & (tomorrow == target)).sum())
    return average_precision(torch.tensor([0.0, 1.0]), counts[:, 1], counts.sum(dim=1))


def summary(results_file: str | Path) -> None:
    """Print the markdown leaderboards under the Lahrichi et al. protocol: their Table 2 (WSTS) and
    Table 5 (WSTS+), each followed by every experiment of that protocol in `results_file` as
    mean ± sample std over its folds."""

    runs = {protocol: defaultdict(dict) for protocol in PROTOCOLS}
    if Path(results_file).exists():
        for line in Path(results_file).read_text().splitlines():
            run = json.loads(line)
            runs[run.get("protocol", "wsts")][run["experiment_name"]][(run["fold"], run["seed"])] = run
    for protocol, published in (("wsts", PUBLISHED), ("wstsplus", PUBLISHED_WSTSPLUS)):
        if protocol != "wsts":
            print()
        _leaderboard(published, len(PROTOCOLS[protocol]), runs[protocol])


def _leaderboard(published: list, folds: int, runs: dict) -> None:
    print("| Model | Days | Features | Params | Folds | AP (centre 128) | F1 | IoU |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, days, features, mean, std, params in published:
        print(f"| [Lahrichi]({LAHRICHI_URL}): {name} | {days} | {features} | {params} | {folds} | {mean:.3f} ± {std:.3f} | | |")
    for name, by_fold in runs.items():
        first = next(iter(by_fold.values()))
        cells = []
        for metric in ("test_ap_centre128", "test_f1", "test_iou"):
            values = np.array([run[metric] for run in by_fold.values()])
            std = values.std(ddof=STD_DDOF[metric]) if len(values) > 1 else 0.0
            cells.append(f"{values.mean():.3f} ± {std:.3f}")
        print(f"| {name} | {first['window']} | all | {first['parameters']:,} | {len(by_fold)} | " + " | ".join(cells) + " |")


def verify() -> None:
    """Checks a reviewer can run in seconds: our AP against torchmetrics' (the metric of both papers),
    the centre window against torchvision's `center_crop`, and the published std conventions."""

    generator = torch.Generator().manual_seed(0)
    scores = torch.randint(0, 50, (200_000,), generator=generator).float() / 7 - 3
    targets = (torch.rand(200_000, generator=generator) < torch.sigmoid(scores - 2)).long()
    try:
        from torchmetrics.classification import BinaryAveragePrecision
        reference = float(BinaryAveragePrecision()(scores, targets))
        print(f"AP vs torchmetrics.BinaryAveragePrecision: ours {average_precision(scores, targets):.8f}, theirs {reference:.8f}")
    except ImportError:
        print("torchmetrics not installed: run with `uv run --with torchmetrics` to compare the AP implementations")
    try:
        import torchvision.transforms.functional as TF
        image = torch.arange(300 * 250, dtype=torch.float32).reshape(1, 300, 250)
        crop = TF.center_crop(image, [CENTRE_CROP, CENTRE_CROP])
        top, left = _centre_crop_start(300, CENTRE_CROP), _centre_crop_start(250, CENTRE_CROP)
        print("centre window vs torchvision center_crop (300 x 250 image):",
              "identical" if torch.equal(crop, image[:, top : top + CENTRE_CROP, left : left + CENTRE_CROP]) else "DIFFERENT")
    except ImportError:
        print("torchvision not installed: run with `uv run --with torchvision` to compare the centre window")
    folds = np.array(LAHRICHI_UTAE_RES18_VEG)
    print(f"Lahrichi UTAE(Res18) Veg per-fold values: mean {folds.mean():.3f}, std ddof=1 {folds.std(ddof=1):.3f}, "
          f"ddof=0 {folds.std():.3f} (published 0.478 ± 0.085)")
    persistence = np.array([PUBLISHED_PERSISTENCE[year] for fold in FOLDS for year in fold.test_years])
    print(f"Gerard persistence per-fold values: mean {persistence.mean():.3f}, std ddof=0 {persistence.std():.3f}, "
          f"ddof=1 {persistence.std(ddof=1):.3f} (published 0.193 ± 0.065)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["verify", "persistence", "summary"])
    parser.add_argument("--raw-dir", default="data/raw/wildfirespreadts")
    parser.add_argument("--results-file", default="results/benchmark_runs.jsonl")
    args = parser.parse_args()
    if args.command == "summary":
        summary(args.results_file)
        return
    if args.command == "verify":
        verify()
        return
    per_year = {year: pixel_persistence_ap(args.raw_dir, year) for year in YEARS}
    for year, value in per_year.items():
        print(f"{year}: AP {value:.3f} (published {PUBLISHED_PERSISTENCE[year]:.3f})")
    folds = [per_year[year] for fold in FOLDS for year in fold.test_years]
    print(f"12-fold mean: {np.mean(folds):.3f} ± {np.std(folds):.3f} (published 0.193 ± 0.065)")


if __name__ == "__main__":
    main()
