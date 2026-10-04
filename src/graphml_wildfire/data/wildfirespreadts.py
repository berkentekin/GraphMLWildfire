from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


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
        return f"{self.year}_{self.fire_name}_{self.input_date.isoformat()}"


def _read_hdf5_dates(values) -> list[date]:
    dates = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        dates.append(date.fromisoformat(str(value).replace(".tif", "")))
    return dates


def find_fire_pairs(data_dir: str | Path, years: Iterable[int] | None = None) -> list[FirePair]:
    """Find consecutive daily pairs in converted HDF5 fire files."""

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    if years is None:
        year_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    else:
        year_dirs = [root / str(year) for year in years]

    pairs = []
    for year_dir in year_dirs:
        if not year_dir.is_dir():
            raise FileNotFoundError(f"Year directory does not exist: {year_dir}")
        files = sorted([*year_dir.glob("*.hdf5"), *year_dir.glob("*.h5")])
        for file_path in files:
            with h5py.File(file_path, "r") as source:
                dates = _read_hdf5_dates(source["data"].attrs["img_dates"])

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
    """Dataset wrapper for one-day examples stored as HDF5 files."""

    def __init__(
        self,
        data_dir: str | Path,
        years: Iterable[int] | None = None,
    ):
        self.pairs = find_fire_pairs(data_dir, years)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        with h5py.File(pair.file_path, "r") as source:
            data = source["data"]
            features = data[pair.input_index].astype(np.float32)
            next_day = data[pair.target_index].astype(np.float32)
            transform = tuple(data.attrs["transform"])
            crs = str(data.attrs["crs"])

        if features.shape[1:] != next_day.shape[1:]:
            raise ValueError(f"Image sizes do not match for sample {pair.sample_id}")

        # The final WildfireSpreadTS channel is active-fire detection time.
        target = np.nan_to_num(next_day[-1], nan=0.0) > 0

        return {
            "sample_id": pair.sample_id,
            "features": features,
            "target": target.astype(np.int64),
            "year": pair.year,
            "fire_name": pair.fire_name,
            "date": pair.input_date.isoformat(),
            "transform": transform,
            "crs": crs,
        }
