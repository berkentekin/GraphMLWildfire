import argparse
from datetime import date
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import rasterio
from affine import Affine
from tqdm import tqdm

from .channels import (
    ACTIVE_FIRE_INDEX,
    ANGLE_INDICES,
    BASE_FEATURE_NAMES,
    CONTINUOUS_FEATURE_INDICES,
    LAND_COVER_CLASSES,
    LAND_COVER_INDEX,
    PHYSICS_NAMES,
    PROCESSED_FEATURE_NAMES,
)


DEFAULT_BLOCK_SIZE = 4
# Side of the centre window on which Lahrichi et al. (WACV 2026) report test AP.
CENTRE_CROP = 128


def burning_pixels(active_fire: np.ndarray) -> np.ndarray:
    """Active-fire pixels as the WildfireSpreadTS benchmark sees them.

    The official HDF5 conversion floor-divides the detection time (hhmm) by 100 before
    testing > 0, so detections between 00:00 and 00:59 do not count.
    """

    return np.nan_to_num(active_fire) >= 100


def _centre_crop_start(size: int, crop: int) -> int:
    # Same rounding as torchvision.transforms.functional.center_crop.
    return int(round((size - crop) / 2.0))


def benchmark_crop(height: int, width: int) -> tuple[slice, slice]:
    """Rows and columns of a test image kept by the benchmark: a centre crop to multiples of 32."""

    crop_height, crop_width = height // 32 * 32, width // 32 * 32
    top, left = _centre_crop_start(height, crop_height), _centre_crop_start(width, crop_width)
    return slice(top, top + crop_height), slice(left, left + crop_width)


def _date_from_filename(path: Path) -> date:
    date_text = path.stem.split("_")[0]
    try:
        return date.fromisoformat(date_text)
    except ValueError as error:
        raise ValueError(f"Could not read a date from filename: {path.name}") from error


def _block_mean(values: np.ndarray, block_size: int) -> np.ndarray:
    """Average valid pixels in non-overlapping blocks."""

    if values.ndim == 2:
        values = values[None]
        squeeze = True
    else:
        squeeze = False

    channels, height, width = values.shape
    node_rows = (height + block_size - 1) // block_size
    node_columns = (width + block_size - 1) // block_size
    padded = np.full(
        (channels, node_rows * block_size, node_columns * block_size),
        np.nan,
        dtype=np.float64,
    )
    padded[:, :height, :width] = values
    blocks = padded.reshape(
        channels, node_rows, block_size, node_columns, block_size
    )
    valid = np.isfinite(blocks)
    counts = valid.sum(axis=(2, 4))
    sums = np.where(valid, blocks, 0.0).sum(axis=(2, 4))
    means = np.full(sums.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means[0] if squeeze else means


def _block_count(values: np.ndarray, block_size: int) -> np.ndarray:
    """Count true pixels of a boolean map [H, W] in non-overlapping blocks."""

    return np.rint(
        _block_mean(values.astype(np.float32), block_size)
        * _coverage(*values.shape, block_size)
        * block_size**2
    ).astype(np.uint16)


def _coverage(height: int, width: int, block_size: int) -> np.ndarray:
    return np.asarray(
        [
            [
                min(block_size, height - row * block_size)
                * min(block_size, width - column * block_size)
                / block_size**2
                for column in range((width + block_size - 1) // block_size)
            ]
            for row in range((height + block_size - 1) // block_size)
        ],
        dtype=np.float32,
    )


def _land_cover_proportions(values: np.ndarray, block_size: int) -> np.ndarray:
    proportions = []
    for class_id in range(1, LAND_COVER_CLASSES + 1):
        count = _block_mean((values == class_id).astype(np.float32), block_size)
        proportions.append(count)
    unknown = (
        ~np.isfinite(values)
        | (values < 1)
        | (values > LAND_COVER_CLASSES)
        | (values != np.floor(values))
    )
    proportions.append(_block_mean(unknown.astype(np.float32), block_size))
    return np.stack(proportions).astype(np.float32)


def _circular_components(values: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(values)
    return _block_mean(np.sin(radians), block_size), _block_mean(
        np.cos(radians), block_size
    )


def _circular_mean(values: np.ndarray, block_size: int) -> np.ndarray:
    sine, cosine = _circular_components(values, block_size)
    direction = np.mod(np.rad2deg(np.arctan2(sine, cosine)), 360.0)
    undefined = (
        ~np.isfinite(sine)
        | ~np.isfinite(cosine)
        | (np.hypot(sine, cosine) < 1e-8)
    )
    direction[undefined] = np.nan
    return direction.astype(np.float32)


def preprocess_image(
    image: np.ndarray,
    transform: Affine,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, np.ndarray]:
    """Aggregate one daily raster into graph-independent node grids."""

    if image.ndim != 3 or image.shape[0] != len(BASE_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(BASE_FEATURE_NAMES)} channels, got shape {image.shape}"
        )
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    features = []
    for index in range(len(BASE_FEATURE_NAMES)):
        values = image[index].astype(np.float64)
        if index in ANGLE_INDICES:
            features.extend(_circular_components(values, block_size))
        elif index == LAND_COVER_INDEX:
            features.extend(_land_cover_proportions(values, block_size))
        elif index != ACTIVE_FIRE_INDEX:
            features.append(_block_mean(values, block_size))

    fire_pixels = burning_pixels(image[ACTIVE_FIRE_INDEX])
    fire_fraction = _block_mean(fire_pixels.astype(np.float32), block_size)
    fire_presence = fire_fraction > 0
    # Mean detection time (hhmm) over the burning pixels of a node, NaN without fire.
    fire_time = _block_mean(
        np.where(fire_pixels, image[ACTIVE_FIRE_INDEX], np.nan), block_size
    )

    physics = np.stack(
        [
            _block_mean(image[6], block_size),
            _circular_mean(image[7], block_size),
            _block_mean(image[18], block_size),
            _circular_mean(image[19], block_size),
            _block_mean(image[14], block_size),
            _block_mean(image[12], block_size),
            _circular_mean(image[13], block_size),
        ]
    )

    height, width = image.shape[1:]
    rows, columns = np.meshgrid(
        np.arange(height, dtype=np.float64) + 0.5,
        np.arange(width, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    x = transform.a * columns + transform.b * rows + transform.c
    y = transform.d * columns + transform.e * rows + transform.f

    return {
        "features": np.stack(features).astype(np.float32),
        "fire_presence": fire_presence.astype(np.uint8),
        "fire_fraction": fire_fraction.astype(np.float32),
        "fire_pixels": _block_count(fire_pixels, block_size),
        "fire_time": fire_time.astype(np.float32),
        "viirs_missing": _block_mean(np.isnan(image[0]).astype(np.float32), block_size).astype(np.float32),
        "ndvi_missing": _block_mean(np.isnan(image[3]).astype(np.float32), block_size).astype(np.float32),
        "physics": physics.astype(np.float32),
        "coverage": _coverage(height, width, block_size),
        "positions": np.stack(
            (_block_mean(x, block_size), _block_mean(y, block_size))
        ),
    }


# Per-day node maps stored next to `fire_presence` and `fire_fraction`.
_DAILY_MAPS = ("fire_pixels", "fire_time", "viirs_missing", "ndvi_missing", "centre_fire")


def convert_fire(
    fire_dir: Path,
    output_path: Path,
    overwrite: bool = False,
    block_size: int = DEFAULT_BLOCK_SIZE,
    crop: bool = False,
) -> bool:
    """Preprocess one directory of daily GeoTIFFs into one HDF5 file.

    With `crop`, every image is first centre-cropped like a WildfireSpreadTS test image
    (`benchmark_crop`), so the nodes tile exactly the pixels the benchmark scores.
    """

    images = sorted(fire_dir.glob("*.tif"), key=_date_from_filename)
    if not images:
        return False
    if output_path.exists() and not overwrite:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(images[0]) as source:
        first_image = source.read().astype(np.float32)
        source_transform = source.transform
        crs = str(source.crs)
    height, width = first_image.shape[1:]
    rows, columns = benchmark_crop(height, width) if crop else (slice(0, height), slice(0, width))
    node_source_transform = source_transform * Affine.translation(columns.start, rows.start)
    # Pixels of the centre CENTRE_CROP x CENTRE_CROP window of the original image.
    top, left = _centre_crop_start(height, CENTRE_CROP), _centre_crop_start(width, CENTRE_CROP)
    in_centre = np.zeros((height, width), dtype=bool)
    in_centre[top : top + CENTRE_CROP, left : left + CENTRE_CROP] = True
    in_centre = in_centre[rows, columns]

    def process(image: np.ndarray) -> dict[str, np.ndarray]:
        processed = preprocess_image(image[:, rows, columns], node_source_transform, block_size)
        burning = burning_pixels(image[ACTIVE_FIRE_INDEX, rows, columns])
        processed["centre_fire"] = _block_count(burning & in_centre, block_size)
        return processed

    first = process(first_image)

    with h5py.File(output_path, "w") as output:
        node_rows, node_columns = first["fire_presence"].shape
        feature_chunks = (
            1,
            first["features"].shape[0],
            min(64, node_rows),
            min(64, node_columns),
        )
        features = output.create_dataset(
            "features",
            shape=(len(images), *first["features"].shape),
            dtype=np.float32,
            chunks=feature_chunks,
            compression="lzf",
        )
        physics = output.create_dataset(
            "physics",
            shape=(len(images), *first["physics"].shape),
            dtype=np.float32,
            chunks=(
                1,
                first["physics"].shape[0],
                min(64, node_rows),
                min(64, node_columns),
            ),
            compression="lzf",
        )
        fire_presence = output.create_dataset(
            "fire_presence",
            shape=(len(images), node_rows, node_columns),
            dtype=np.uint8,
            chunks=(1, min(128, node_rows), min(128, node_columns)),
            compression="lzf",
        )
        fire_fraction = output.create_dataset(
            "fire_fraction",
            shape=(len(images), node_rows, node_columns),
            dtype=np.float32,
            chunks=(1, min(128, node_rows), min(128, node_columns)),
            compression="lzf",
        )
        daily = {
            name: output.create_dataset(
                name,
                shape=(len(images), node_rows, node_columns),
                dtype=first[name].dtype,
                chunks=(1, min(128, node_rows), min(128, node_columns)),
                compression="lzf",
            )
            for name in _DAILY_MAPS
        }
        output.create_dataset("coverage", data=first["coverage"], compression="lzf")
        output.create_dataset(
            "centre_pixels", data=_block_count(in_centre, block_size), compression="lzf"
        )
        output.create_dataset("positions", data=first["positions"], compression="lzf")

        features[0] = first["features"]
        physics[0] = first["physics"]
        fire_presence[0] = first["fire_presence"]
        fire_fraction[0] = first["fire_fraction"]
        for name, dataset in daily.items():
            dataset[0] = first[name]

        for index, image_path in enumerate(images[1:], start=1):
            with rasterio.open(image_path) as source:
                image = source.read().astype(np.float32)
                if image.shape != first_image.shape:
                    raise ValueError(f"Image shape changed inside fire {fire_dir.name}")
                if source.transform != source_transform or str(source.crs) != crs:
                    raise ValueError(f"Georeferencing changed inside fire {fire_dir.name}")
                processed = process(image)
                features[index] = processed["features"]
                physics[index] = processed["physics"]
                fire_presence[index] = processed["fire_presence"]
                fire_fraction[index] = processed["fire_fraction"]
                for name, dataset in daily.items():
                    dataset[index] = processed[name]

        dates = [str(_date_from_filename(path)) for path in images]
        string_type = h5py.string_dtype("utf-8")
        output.attrs["schema_version"] = 1
        output.attrs["year"] = int(fire_dir.parent.name)
        output.attrs["fire_name"] = fire_dir.name
        output.attrs["img_dates"] = np.asarray(dates, dtype=string_type)
        output.attrs["source_transform"] = tuple(source_transform)
        output.attrs["node_transform"] = tuple(
            node_source_transform @ Affine.scale(block_size, block_size)
        )
        output.attrs["benchmark_crop"] = crop
        output.attrs["crop_offset"] = (rows.start, columns.start)
        output.attrs["crs"] = crs
        output.attrs["block_size"] = block_size
        output.attrs["source_shape"] = first_image.shape[1:]
        output.attrs["feature_names"] = np.asarray(
            PROCESSED_FEATURE_NAMES, dtype=string_type
        )
        output.attrs["continuous_feature_indices"] = CONTINUOUS_FEATURE_INDICES
        output.attrs["physics_names"] = np.asarray(PHYSICS_NAMES, dtype=string_type)

    return True


def convert_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    years: Iterable[int] | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
    block_size: int = DEFAULT_BLOCK_SIZE,
    crop: bool = False,
) -> list[Path]:
    """Preprocess all downloaded WildfireSpreadTS fires into HDF5."""

    root = Path(data_dir)
    target_root = Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    if years is None:
        year_dirs = sorted(
            path for path in root.iterdir() if path.is_dir() and path.name.isdigit()
        )
    else:
        year_dirs = [root / str(year) for year in years]

    fires = []
    for year_dir in year_dirs:
        if not year_dir.is_dir():
            raise FileNotFoundError(f"Year directory does not exist: {year_dir}")
        for fire_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
            fires.append((year_dir, fire_dir))

    converted = []
    progress = tqdm(
        fires, desc="Preprocessing fires", unit="fire", disable=not show_progress
    )
    for year_dir, fire_dir in progress:
        progress.set_postfix_str(f"{year_dir.name}/{fire_dir.name}")
        output_path = target_root / year_dir.name / f"{fire_dir.name}.hdf5"
        if convert_fire(fire_dir, output_path, overwrite, block_size, crop):
            converted.append(output_path)

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess WildfireSpreadTS GeoTIFFs to HDF5"
    )
    parser.add_argument(
        "data_dir", help="Directory containing the original year/fire/TIFF layout"
    )
    parser.add_argument("output_dir", help="Directory where HDF5 files will be written")
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--benchmark-crop",
        action="store_true",
        help="Centre-crop every image like a WildfireSpreadTS test image",
    )
    args = parser.parse_args()

    converted = convert_dataset(
        args.data_dir,
        args.output_dir,
        args.years,
        args.overwrite,
        block_size=args.block_size,
        crop=args.benchmark_crop,
    )
    print(f"Created {len(converted)} HDF5 files")


if __name__ == "__main__":
    main()
