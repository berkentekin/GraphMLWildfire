import csv
import hashlib
import random
from dataclasses import dataclass
from datetime import date
from itertools import combinations, permutations
from pathlib import Path
from typing import Iterable

import h5py


@dataclass(frozen=True)
class YearFold:
    train_years: tuple[int, int]
    validation_year: int
    test_year: int

    def __post_init__(self) -> None:
        years = (*self.train_years, self.validation_year, self.test_year)
        if len(set(years)) != 4:
            raise ValueError("Training, validation, and test years must be distinct")

    @property
    def name(self) -> str:
        train = "_".join(str(year) for year in self.train_years)
        return f"train_{train}_val_{self.validation_year}_test_{self.test_year}"


def create_year_folds(
    years: Iterable[int] = (2018, 2019, 2020, 2021),
) -> list[YearFold]:
    """Create the 12 ordered two-train, one-validation, one-test folds.

    The order matches the official WildfireSpreadTS code (FireSpreadDataModule.split_fires),
    so fold IDs refer to the same years as in the published papers.
    """

    years = tuple(sorted(years))
    if len(years) != 4 or len(set(years)) != 4:
        raise ValueError("Exactly four distinct years are required")

    folds = []
    for train_years in combinations(years, 2):
        held_out = [year for year in years if year not in train_years]
        for validation_year, test_year in permutations(held_out, 2):
            folds.append(YearFold(train_years, validation_year, test_year))  # type: ignore
    return folds


def discover_fires(data_dir: str | Path) -> dict[int, list[str]]:
    """List processed fire IDs independently for each available year."""

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    fires = {}
    for year_dir in sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.isdigit()
    ):
        files = [*year_dir.glob("*.hdf5"), *year_dir.glob("*.h5")]
        fires[int(year_dir.name)] = sorted({path.stem for path in files})
    return fires


def select_fires(
    fires_by_year: dict[int, list[str]],
    fraction: float = 1.0,
    seed: int = 42,
) -> dict[int, list[str]]:
    """Deterministically select a fraction of complete fires in every year."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the range (0, 1]")

    selected = {}
    for year, fire_ids in sorted(fires_by_year.items()):
        fire_ids = sorted(set(fire_ids))
        if not fire_ids:
            selected[year] = []
            continue
        count = max(1, round(fraction * len(fire_ids)))
        year_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{year}".encode("utf-8")).digest()[:8],
            "big",
        )
        selected[year] = sorted(random.Random(year_seed).sample(fire_ids, count))
    return selected


def write_fire_manifest(
    selected: dict[int, list[str]], output_path: str | Path
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("year", "fire_id"))
        for year, fire_ids in sorted(selected.items()):
            for fire_id in sorted(fire_ids):
                writer.writerow((year, fire_id))
    return output_path


def read_fire_manifest(manifest_path: str | Path) -> set[tuple[int, str]]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["year", "fire_id"]:
            raise ValueError("Fire manifest must contain year,fire_id columns")
        return {(int(row["year"]), row["fire_id"]) for row in reader}


def _decode_dates(values) -> list[date]:
    return [
        date.fromisoformat(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
        )
        for value in values
    ]


def summarize_selection(
    data_dir: str | Path,
    all_fires: dict[int, list[str]],
    selected: dict[int, list[str]],
    fraction: float,
) -> None:
    root = Path(data_dir)
    observations = 0
    windows = 0
    print("Selected fires:\n")
    for year in sorted(all_fires):
        print(f"{year}: {len(selected[year])} / {len(all_fires[year])}")
        for fire_id in selected[year]:
            with h5py.File(root / str(year) / f"{fire_id}.hdf5", "r") as source:
                dates = _decode_dates(source.attrs["img_dates"])
            observations += len(dates)
            windows += sum(
                (target - current).days == 1
                for current, target in zip(dates, dates[1:])
            )
    print(f"\nFraction: {fraction:.2f}")
    print(
        f"Total: {sum(map(len, selected.values()))} / "
        f"{sum(map(len, all_fires.values()))}"
    )
    print(f"Daily observations: {observations}")
    print(f"Valid temporal windows: {windows}")


def create_fire_manifest(
    data_dir: str | Path,
    output_path: str | Path,
    fraction: float = 1.0,
    seed: int = 42,
) -> Path:
    fires = discover_fires(data_dir)
    selected = select_fires(fires, fraction, seed)
    manifest_path = write_fire_manifest(selected, output_path)
    summarize_selection(data_dir, fires, selected, fraction)
    return manifest_path
