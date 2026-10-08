"""The WildfireSpreadTS benchmark protocol (Gerard et al., NeurIPS 2023), its metrics and checks.

Every 375 m pixel of the centre-cropped test images is scored, and AP is pooled over all pixels
of the test year: one AP per fold, reported as mean +- std over the 12 folds.

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
from .data.convert_hdf5 import _date_from_filename, benchmark_crop, burning_pixels
from .data.splits import create_year_folds


YEARS = (2018, 2019, 2020, 2021)
# The 12 folds of WildfireSpreadTS, in the order of the official code.
FOLDS = create_year_folds(YEARS)
# Every model is scored on the targets of the benchmark's 5-day models (target days 5..T-1),
# as with `n_leading_observations_test_adjustment: 5` in the official code.
FIRST_INPUT_DAY = 4

# Pixel-level persistence AP per test year, Gerard et al. (2023), Table 2.
PUBLISHED_PERSISTENCE = {2018: 0.187, 2019: 0.105, 2020: 0.194, 2021: 0.287}
# Published 12-fold test AP (mean, std) and parameter count, with all features unless marked as the best feature set.
# "full": Gerard et al. (2023), Table 5, full benchmark crop.
# "centre128": Lahrichi et al. (WACV 2026; arXiv 2502.12003v3), Table 2, centre 128 x 128 crop.
# Papers the published rows come from, linked in the leaderboard.
SOURCES = {
    "Gerard": "https://papers.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf",
    "Lahrichi": "https://arxiv.org/abs/2502.12003v3",
}
PUBLISHED = [
    ("full", "Gerard: persistence", 1, 0.193, 0.065, "0"),
    ("full", "Gerard: logistic regression (All)", 1, 0.286, 0.092, "361"),
    ("full", "Gerard: Res18 U-Net (All)", 1, 0.341, 0.086, "14.4M"),
    ("full", "Gerard: Res18 U-Net (All)", 5, 0.325, 0.108, "14.7M"),
    ("full", "Gerard: ConvLSTM (All)", 5, 0.292, 0.094, "240K"),
    ("full", "Gerard: UTAE (All)", 5, 0.321, 0.135, "1.1M"),
    ("full", "Gerard: UTAE (Veg, best)", 5, 0.372, 0.088, "1.1M"),
    ("centre128", "Lahrichi: Res18-UNet (All)", 1, 0.460, 0.084, "14.3M"),
    ("centre128", "Lahrichi: Res18-UNet (Multi, best T=1)", 1, 0.468, 0.087, "14.3M"),
    ("centre128", "Lahrichi: Res18-UNet, data fusion (Veg)", 5, 0.472, 0.083, "14.4M"),
    ("centre128", "Lahrichi: UTAE(Res18) (All)", 5, 0.475, 0.091, "14.6M"),
    ("centre128", "Lahrichi: UTAE(Res18) (Veg, best)", 5, 0.478, 0.085, "14.6M"),
]


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
    """Print a markdown leaderboard: the published results, then every experiment in `results_file`
    as mean ± std over its folds.

    The std is the population std (ddof=0), which is what the published tables use.
    """

    runs = defaultdict(dict)
    for line in Path(results_file).read_text().splitlines():
        run = json.loads(line)
        runs[run["experiment_name"]][(run["fold"], run["seed"])] = run
    print("| Model | Days | Params | Folds | AP (full) | AP (centre 128) | F1 | IoU |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for region, name, days, mean, std, params in PUBLISHED:
        cells = [f"{mean:.3f} ± {std:.3f}" if region == column else "–" for column in ("full", "centre128")]
        source, model = name.split(": ", 1)
        print(f"| [{source}]({SOURCES[source]}): {model} | {days} | {params} | 12 | {cells[0]} | {cells[1]} | | |")
    for name, by_fold in runs.items():
        first = next(iter(by_fold.values()))
        cells = []
        for metric in ("test_ap", "test_ap_centre128", "test_f1", "test_iou"):
            values = np.array([run[metric] for run in by_fold.values()])
            cells.append(f"{values.mean():.3f} ± {values.std():.3f}")
        print(f"| {name} | {first['window']} | {first['parameters']:,} | {len(by_fold)} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["persistence", "summary"])
    parser.add_argument("--raw-dir", default="data/raw/wildfirespreadts")
    parser.add_argument("--results-file", default="results/benchmark_runs.jsonl")
    args = parser.parse_args()
    if args.command == "summary":
        summary(args.results_file)
        return
    per_year = {year: pixel_persistence_ap(args.raw_dir, year) for year in YEARS}
    for year, value in per_year.items():
        print(f"{year}: AP {value:.3f} (published {PUBLISHED_PERSISTENCE[year]:.3f})")
    folds = [per_year[fold.test_year] for fold in FOLDS]
    print(f"12-fold mean: {np.mean(folds):.3f} ± {np.std(folds):.3f} (published 0.193 ± 0.065)")


if __name__ == "__main__":
    main()
