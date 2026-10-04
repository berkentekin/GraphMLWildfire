import argparse
from datetime import date
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import rasterio
from tqdm import tqdm


def _date_from_filename(path: Path) -> date:
    date_text = path.stem.split("_")[0]
    try:
        return date.fromisoformat(date_text)
    except ValueError as error:
        raise ValueError(f"Could not read a date from filename: {path.name}") from error


def convert_fire(fire_dir: Path, output_path: Path, overwrite: bool = False) -> bool:
    """Convert one directory of daily GeoTIFFs into one HDF5 file."""

    images = sorted(fire_dir.glob("*.tif"), key=_date_from_filename)
    if not images:
        return False
    if output_path.exists() and not overwrite:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(images[0]) as source:
        first_image = source.read().astype(np.float32)
        transform = tuple(source.transform)
        crs = str(source.crs)

    with h5py.File(output_path, "w") as output:
        chunks = (1, first_image.shape[0], min(128, first_image.shape[1]), min(128, first_image.shape[2]))
        data = output.create_dataset(
            "data",
            shape=(len(images), *first_image.shape),
            dtype=np.float32,
            chunks=chunks,
            compression="lzf",
        )
        data[0] = first_image

        for index, image_path in enumerate(images[1:], start=1):
            with rasterio.open(image_path) as source:
                image = source.read().astype(np.float32)
                if image.shape != first_image.shape:
                    raise ValueError(f"Image shape changed inside fire {fire_dir.name}")
                if tuple(source.transform) != transform or str(source.crs) != crs:
                    raise ValueError(f"Georeferencing changed inside fire {fire_dir.name}")
                data[index] = image

        dates = [str(_date_from_filename(path)) for path in images]
        data.attrs["year"] = int(fire_dir.parent.name)
        data.attrs["fire_name"] = fire_dir.name
        data.attrs["img_dates"] = np.asarray(dates, dtype=h5py.string_dtype("utf-8"))
        data.attrs["transform"] = transform
        data.attrs["crs"] = crs

    return True


def convert_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    years: Iterable[int] | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> list[Path]:
    """Convert all WildfireSpreadTS fire directories to HDF5."""

    root = Path(data_dir)
    target_root = Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    if years is None:
        year_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    else:
        year_dirs = [root / str(year) for year in years]

    fires = []
    for year_dir in year_dirs:
        if not year_dir.is_dir():
            raise FileNotFoundError(f"Year directory does not exist: {year_dir}")
        for fire_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
            fires.append((year_dir, fire_dir))

    converted = []
    progress = tqdm(fires, desc="Converting fires", unit="fire", disable=not show_progress)
    for year_dir, fire_dir in progress:
        progress.set_postfix_str(f"{year_dir.name}/{fire_dir.name}")
        output_path = target_root / year_dir.name / f"{fire_dir.name}.hdf5"
        if convert_fire(fire_dir, output_path, overwrite):
            converted.append(output_path)

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert WildfireSpreadTS GeoTIFFs to HDF5")
    parser.add_argument("data_dir", help="Directory containing the original year/fire/TIFF layout")
    parser.add_argument("output_dir", help="Directory where HDF5 files will be written")
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    converted = convert_dataset(args.data_dir, args.output_dir, args.years, args.overwrite)
    print(f"Created {len(converted)} HDF5 files")


if __name__ == "__main__":
    main()
