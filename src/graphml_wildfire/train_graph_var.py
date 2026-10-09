"""Fit and evaluate Graph VAR forecasting on the benchmark node grid.

The fit is closed-form and the fire signal needs no fold statistics, so one call builds the
graph filter outputs of every sample once and runs all folds. Evaluation and the result rows
follow `train.py`.

Example:
    uv run python -m graphml_wildfire.train_graph_var --shift grid --window 5 --hops 2
"""

import argparse
import json
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm

from .benchmark import FIRST_INPUT_DAY, FOLDS, YEARS, average_precision, best_f1_threshold, threshold_metrics
from .graph import build_grid_shift, normalize_shift, to_sparse
from .models import GraphVAR


@lru_cache(maxsize=None)
def _grid_shift(grid_shape: tuple[int, int]) -> sparse.csr_array:
    return to_sparse(normalize_shift(build_grid_shift(grid_shape)))


# Unit-norm shift operator S of one fire on one day, built from the fire's `_DATASETS` arrays.
# A graph that is rebuilt daily adds its drivers to `_DATASETS` and reads those of `day`.
SHIFTS = {
    "grid": lambda fire, day: _grid_shift(fire["fire_fraction"].shape[1:]),
}
_DATASETS = ("fire_fraction", "fire_pixels", "centre_fire", "centre_pixels")
_COUNTS = ("fire", "pixels", "centre_fire", "centre_pixels")


@dataclass
class GraphVARConfig:
    experiment_name: str = ""
    shift: str = "grid"
    window: int = 5
    hops: int = 2
    patch: int = 4
    cache_dir: str = ""
    results_file: str = "results/benchmark_runs.jsonl"

    def __post_init__(self) -> None:
        if self.shift not in SHIFTS:
            raise ValueError(f"Unknown shift: {self.shift}")
        if not 1 <= self.window <= FIRST_INPUT_DAY + 1:
            raise ValueError(f"window must be between 1 and {FIRST_INPUT_DAY + 1}")
        if not self.cache_dir:
            self.cache_dir = f"data/processed/wildfirespreadts_benchmark_b{self.patch}"
        if not self.experiment_name:
            self.experiment_name = f"gvar_{self.shift}_w{self.window}_k{self.hops}_p{self.patch}"


def year_rows(config: GraphVARConfig, year: int) -> dict[str, np.ndarray]:
    """Regressors [rows, parameters] and pixel counts [rows] of every node of every sample of a year.

    A row is a node on one day: `fire` of its `pixels` burn the next day, `centre_fire` of its
    `centre_pixels` inside the central 128 x 128 window. Nodes with all-zero regressors score 0
    whatever the coefficients, so those that do not burn the next day are pooled into the last
    row, which leaves every metric unchanged.
    """

    year_dir = Path(config.cache_dir) / str(year)
    if not year_dir.is_dir():
        raise FileNotFoundError(f"Converted fires do not exist: {year_dir}")
    model = GraphVAR(config.window, config.hops)
    regressors = []
    counts = {name: [] for name in _COUNTS}
    pooled = dict.fromkeys(_COUNTS, 0.0)
    for path in tqdm(sorted(year_dir.glob("*.hdf5")), desc=str(year), unit="fire"):
        with h5py.File(path, "r") as source:
            if not source.attrs.get("benchmark_crop", False) or source.attrs["block_size"] != config.patch:
                raise ValueError(f"{path} was not converted with --benchmark-crop --block-size {config.patch}")
            fire = {name: source[name][:] for name in _DATASETS}
        signal = fire["fire_fraction"].reshape(len(fire["fire_fraction"]), -1)
        for day in range(FIRST_INPUT_DAY, len(signal) - 1):
            sample = model.regressors(SHIFTS[config.shift](fire, day), signal[day - config.window + 1 : day + 1])
            nodes = {
                "fire": fire["fire_pixels"][day + 1].reshape(-1),
                "pixels": np.full(signal.shape[1], config.patch**2),
                "centre_fire": fire["centre_fire"][day + 1].reshape(-1),
                "centre_pixels": fire["centre_pixels"].reshape(-1),
            }
            active = sample.any(axis=1) | (nodes["fire"] > 0)
            regressors.append(sample[active].astype(np.float32))
            for name, values in nodes.items():
                counts[name].append(values[active])
                pooled[name] += float(values[~active].sum())
    rows = {name: np.concatenate([*values, [pooled[name]]]).astype(np.float64) for name, values in counts.items()}
    rows["regressors"] = np.concatenate([*regressors, np.zeros((1, model.parameters), dtype=np.float32)])
    return rows


def predict(model: GraphVAR, rows: dict[str, np.ndarray]):
    """Predicted burning fraction of every row and, for every evaluation region, its burning and total pixels."""

    scores = torch.from_numpy(model.predict(rows["regressors"]))
    regions = {"full": ("fire", "pixels"), "centre": ("centre_fire", "centre_pixels")}
    return scores, {region: tuple(torch.from_numpy(rows[name]) for name in names) for region, names in regions.items()}


def run(config: GraphVARConfig, folds: list[int]) -> list[dict]:
    rows = {year: year_rows(config, year) for year in YEARS}
    results = []
    for index in folds:
        fold = FOLDS[index]
        train = [rows[year] for year in fold.train_years]
        # Pixel-level least squares: the target is the fraction of a node's pixels that burn the next day.
        model = GraphVAR(config.window, config.hops).fit(
            np.concatenate([part["regressors"] for part in train]),
            np.concatenate([part["fire"] / np.maximum(part["pixels"], 1) for part in train]),
        )
        val_scores, val_regions = predict(model, rows[fold.validation_year])
        test_scores, test_regions = predict(model, rows[fold.test_year])
        threshold = best_f1_threshold(val_scores, *val_regions["full"])
        result = {
            "run_name": f"{config.experiment_name}_f{index}_s0",
            **asdict(config),
            "model": "gvar",
            "fold": index,
            "seed": 0,
            "parameters": model.parameters,
            "coefficients": model.coefficients.tolist(),
            "val_ap": average_precision(val_scores, *val_regions["full"]),
            "test_ap": average_precision(test_scores, *test_regions["full"]),
            "test_ap_centre128": average_precision(test_scores, *test_regions["centre"]),
            # Benchmark convention: predicted burning fraction >= 0.5.
            **{f"test_{name}": value for name, value in threshold_metrics(test_scores, *test_regions["full"], 0.5).items()},
            # Threshold that maximises F1 on the validation year.
            **{f"test_{name}_tuned": value for name, value in threshold_metrics(test_scores, *test_regions["full"], threshold).items()},
        }
        print(f"{result['run_name']}: val AP {result['val_ap']:.4f} test AP {result['test_ap']:.4f} "
              f"(centre128 {result['test_ap_centre128']:.4f}) F1 {result['test_f1']:.4f} IoU {result['test_iou']:.4f} "
              f"(tuned F1 {result['test_f1_tuned']:.4f} IoU {result['test_iou_tuned']:.4f})")
        results.append(result)

    results_file = Path(config.results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("a") as output:
        for result in results:
            output.write(json.dumps(result) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate Graph VAR forecasting")
    for field in fields(GraphVARConfig):
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=field.type, default=field.default)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(len(FOLDS))))
    args = vars(parser.parse_args())
    folds = args.pop("folds")
    run(GraphVARConfig(**args), folds)


if __name__ == "__main__":
    main()
