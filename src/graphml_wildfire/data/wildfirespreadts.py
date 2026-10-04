from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Iterable

import h5py
import numpy as np

from .splits import read_fire_manifest


@dataclass(frozen=True)
class FirePair:
    """One day of observations and the following day's fire target."""

    year: int
    fire_name: str
    file_path: Path
    input_date: date
    target_date: date
    input_index: int
    target_index: int

    @property
    def sample_id(self) -> str:
        return (
            f"{self.year}/{self.fire_name}/"
            f"{self.input_date.isoformat()}__{self.target_date.isoformat()}"
        )


def _read_hdf5_dates(values) -> list[date]:
    dates = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        dates.append(date.fromisoformat(str(value).replace(".tif", "")))
    return dates


def _read_strings(values) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def find_fire_pairs(
    data_dir: str | Path,
    years: Iterable[int] | None = None,
    fire_manifest: str | Path | None = None,
) -> list[FirePair]:
    """Find consecutive daily pairs in converted HDF5 fire files."""

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    if years is None:
        year_dirs = sorted(
            path for path in root.iterdir() if path.is_dir() and path.name.isdigit()
        )
    else:
        year_dirs = [root / str(year) for year in years]

    selected_fires = read_fire_manifest(fire_manifest) if fire_manifest else None
    pairs = []
    for year_dir in year_dirs:
        if not year_dir.is_dir():
            raise FileNotFoundError(f"Year directory does not exist: {year_dir}")
        files = sorted([*year_dir.glob("*.hdf5"), *year_dir.glob("*.h5")])
        for file_path in files:
            if selected_fires is not None and (
                int(year_dir.name), file_path.stem
            ) not in selected_fires:
                continue
            with h5py.File(file_path, "r") as source:
                dates = _read_hdf5_dates(source.attrs["img_dates"])

            for index, (input_date, target_date) in enumerate(zip(dates, dates[1:])):
                if (target_date - input_date).days != 1:
                    continue
                pairs.append(
                    FirePair(
                        year=int(year_dir.name),
                        fire_name=file_path.stem,
                        file_path=file_path,
                        input_date=input_date,
                        target_date=target_date,
                        input_index=index,
                        target_index=index + 1,
                    )
                )

    return pairs


class WildfireSpreadTSDataset:
    """Load unnormalized node grids for consecutive WildfireSpreadTS days."""

    def __init__(
        self,
        data_dir: str | Path,
        years: Iterable[int] | None = None,
        fire_manifest: str | Path | None = None,
        report_timing: bool = False,
        timing_interval: int = 100,
    ):
        self.pairs = find_fire_pairs(data_dir, years, fire_manifest)
        self.report_timing = report_timing
        self.timing_interval = timing_interval
        self._load_count = 0
        self._load_seconds = 0.0
        if timing_interval < 1:
            raise ValueError("timing_interval must be at least 1")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        started = perf_counter()
        pair = self.pairs[index]
        with h5py.File(pair.file_path, "r") as source:
            features = source["features"][pair.input_index].astype(np.float32)
            current_fire = source["fire_presence"][pair.input_index].astype(np.uint8)
            current_fire_fraction = source["fire_fraction"][pair.input_index].astype(
                np.float32
            )
            next_fire = source["fire_presence"][pair.target_index].astype(np.uint8)
            next_fire_fraction = source["fire_fraction"][pair.target_index].astype(
                np.float32
            )
            physics = source["physics"][pair.input_index].astype(np.float32)
            coverage = source["coverage"][:].astype(np.float32)
            positions = source["positions"][:]
            transform = tuple(source.attrs["node_transform"])
            crs = str(source.attrs["crs"])
            feature_names = _read_strings(source.attrs["feature_names"])
            continuous_feature_indices = tuple(
                int(value) for value in source.attrs["continuous_feature_indices"]
            )
            physics_names = _read_strings(source.attrs["physics_names"])

        if features.shape[1:] != next_fire.shape:
            raise ValueError(f"Image sizes do not match for sample {pair.sample_id}")

        elapsed = perf_counter() - started
        self._load_count += 1
        self._load_seconds += elapsed
        if self.report_timing and (
            self._load_count == 1 or self._load_count % self.timing_interval == 0
        ):
            mean_ms = self._load_seconds / self._load_count * 1000
            print(
                f"HDF5 loading: {self._load_count} samples, "
                f"mean={mean_ms:.2f} ms, latest={elapsed * 1000:.2f} ms"
            )

        return {
            "sample_id": pair.sample_id,
            "features": features,
            "feature_names": feature_names,
            "continuous_feature_indices": continuous_feature_indices,
            "current_fire": current_fire,
            "current_fire_fraction": current_fire_fraction,
            "next_fire": next_fire,
            "next_fire_fraction": next_fire_fraction,
            "physics": physics,
            "physics_names": physics_names,
            "coverage": coverage,
            "positions": positions,
            "grid_shape": next_fire.shape,
            "year": pair.year,
            "fire_name": pair.fire_name,
            "date": pair.input_date.isoformat(),
            "target_date": pair.target_date.isoformat(),
            "transform": transform,
            "crs": crs,
        }
