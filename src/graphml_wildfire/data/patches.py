"""Benchmark-aligned node cache: per-node features and pixel-level targets for every fire.

Every daily GeoTIFF is centre-cropped exactly like a WildfireSpreadTS test image, aggregated
into patch nodes with `convert_hdf5.preprocess_image`, and stored together with per-node counts
of the burning pixels that the benchmark scores.
"""

import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from affine import Affine
from tqdm import tqdm

from .channels import ACTIVE_FIRE_INDEX, BASE_FEATURE_NAMES, PROCESSED_FEATURE_NAMES
from .convert_hdf5 import _block_mean, _date_from_filename, preprocess_image


_CONTINUOUS_SKIP = ("_sin", "_cos")
# Standardised per fold with training-year statistics.
CONTINUOUS = [
    name for name in PROCESSED_FEATURE_NAMES if not name.endswith(_CONTINUOUS_SKIP) and not name.startswith("land_cover_")
]
# Unit vectors in array coordinates (row grows southward, col grows eastward), from the circular
# block means of the direction channels. wind_* points downwind, aspect_* points downslope.
VECTOR = ["wind_row", "wind_col", "aspect_row", "aspect_col", "fc_wind_row", "fc_wind_col"]
LANDCOVER = [name for name in PROCESSED_FEATURE_NAMES if name.startswith("land_cover_")]
MASKS = ["viirs_missing", "ndvi_missing"]
# fire_frac/fire_time describe the current day; the rest summarise the HISTORY days up to it,
# so a node never sees further back than the 5-day models of the WildfireSpreadTS benchmark.
FIRE = ["fire_frac", "fire_time", "burned_ever", "recency_2d", "recency_7d"]
HISTORY = 5
CHANNELS = CONTINUOUS + VECTOR + LANDCOVER + MASKS + FIRE
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNELS)}

_LOG_SCALED = {"total_precipitation", "forecast_total_precipitation"}
_BAND = {name: index for index, name in enumerate(BASE_FEATURE_NAMES)}
# Side of the centre crop on which Lahrichi et al. (WACV 2026) report test AP.
CENTRE_CROP = 128


def burning_pixels(active_fire: np.ndarray) -> np.ndarray:
    """Active-fire pixels as the benchmark sees them.

    The WildfireSpreadTS HDF5 conversion floor-divides the detection time (hhmm) by 100
    before testing > 0, so detections between 00:00 and 00:59 do not count.
    """

    return np.nan_to_num(active_fire) >= 100


def _centre_crop_start(size: int, crop: int) -> int:
    # Same rounding as torchvision.transforms.functional.center_crop.
    return int(round((size - crop) / 2.0))


def benchmark_crop(height: int, width: int) -> tuple[slice, slice]:
    """Rows and columns of the test image kept by the benchmark: a centre crop to multiples of 32."""

    crop_height, crop_width = height // 32 * 32, width // 32 * 32
    top, left = _centre_crop_start(height, crop_height), _centre_crop_start(width, crop_width)
    return slice(top, top + crop_height), slice(left, left + crop_width)


def read_fire(fire_dir: str | Path) -> tuple[np.ndarray, Affine]:
    """All daily GeoTIFFs of one fire as [T, 23, H, W], in date order, and their transform."""

    images, transform = [], None
    for path in sorted(Path(fire_dir).glob("*.tif"), key=_date_from_filename):
        with rasterio.open(path) as source:
            images.append(source.read().astype(np.float32))
            if transform is None:
                transform = source.transform
            elif source.transform != transform:
                raise ValueError(f"Georeferencing changed inside fire {fire_dir}")
    if transform.e >= 0:
        raise ValueError(f"Expected a north-up raster: {fire_dir}")
    return np.stack(images), transform


def node_features(data: np.ndarray, transform: Affine, patch: int) -> np.ndarray:
    """Turn one cropped fire [T, 23, H, W] into node features [T, len(CHANNELS), H // patch, W // patch]."""

    if data.shape[-2] % patch or data.shape[-1] % patch:
        raise ValueError(f"Image {data.shape[-2:]} is not a multiple of the patch size {patch}")
    days = []
    for image in data:
        processed = preprocess_image(image, transform, patch)["features"]
        named = dict(zip(PROCESSED_FEATURE_NAMES, processed))
        out = {name: named[name] for name in CONTINUOUS + LANDCOVER}
        for name in _LOG_SCALED:
            out[name] = np.log1p(np.maximum(out[name], 0))
        # Directions are where the wind comes from and where the slope faces, clockwise from north.
        for prefix, name in (("", "wind_direction"), ("fc_", "forecast_wind_direction")):
            out[f"{prefix}wind_row"] = np.nan_to_num(named[f"{name}_cos"])
            out[f"{prefix}wind_col"] = np.nan_to_num(-named[f"{name}_sin"])
        out["aspect_row"] = np.nan_to_num(-named["aspect_cos"])
        out["aspect_col"] = np.nan_to_num(named["aspect_sin"])
        out["viirs_missing"] = _block_mean(np.isnan(image[_BAND["viirs_m11"]]).astype(np.float32), patch)
        out["ndvi_missing"] = _block_mean(np.isnan(image[_BAND["ndvi"]]).astype(np.float32), patch)
        days.append(out)

    detection_time = np.nan_to_num(data[:, ACTIVE_FIRE_INDEX])
    burning = burning_pixels(detection_time)
    fire_frac = _block_mean(burning.astype(np.float32), patch)
    # Mean detection time (hhmm / 2400) over the burning pixels of a node.
    fire_time = _block_mean(detection_time * burning, patch) / np.maximum(fire_frac, 1 / patch**2) / 2400.0

    burned_ever, recency = [], {2: [], 7: []}
    for day in range(len(fire_frac)):
        history = fire_frac[max(day - HISTORY + 1, 0) : day + 1][::-1]
        burned_ever.append(history.max(axis=0))
        # Days since the last fire within the history; never burning counts as infinitely long ago.
        days_since = np.where((history > 0).any(axis=0), (history > 0).argmax(axis=0), np.inf)
        for scale, values in recency.items():
            values.append(np.exp(-days_since / scale))
    for day, out in enumerate(days):
        out["fire_frac"] = fire_frac[day]
        out["fire_time"] = fire_time[day]
        out["burned_ever"] = burned_ever[day]
        out["recency_2d"] = recency[2][day]
        out["recency_7d"] = recency[7][day]

    return np.stack([np.stack([out[name] for name in CHANNELS]) for out in days]).astype(np.float32)


def _count_blocks(values: np.ndarray, patch: int) -> np.ndarray:
    """Sum boolean pixels [..., H, W] over non-overlapping patch x patch blocks."""

    lead, (height, width) = values.shape[:-2], values.shape[-2:]
    blocks = values.reshape(*lead, height // patch, patch, width // patch, patch)
    return blocks.sum(axis=(-3, -1)).astype(np.uint16)


def node_targets(data: np.ndarray, rows: slice, cols: slice, patch: int) -> dict[str, np.ndarray]:
    """Pixel-level fire targets of one fire, counted per node of the cropped image.

    `data` is the uncropped fire [T, 23, H, W] and (`rows`, `cols`) the benchmark crop.
    `fire_pixels[t]` counts the burning pixels of every node on day t. `centre_pixels` and
    `centre_fire` count the pixels, and burning pixels, of every node that lie inside the
    centre CENTRE_CROP x CENTRE_CROP window of the original image.
    """

    height, width = data.shape[-2:]
    burning = burning_pixels(data[:, ACTIVE_FIRE_INDEX])
    top, left = _centre_crop_start(height, CENTRE_CROP), _centre_crop_start(width, CENTRE_CROP)
    in_centre = np.zeros((height, width), dtype=bool)
    in_centre[top : top + CENTRE_CROP, left : left + CENTRE_CROP] = True
    burning, in_centre = burning[:, rows, cols], in_centre[rows, cols]
    if in_centre.sum() != CENTRE_CROP**2:
        raise ValueError(f"The centre {CENTRE_CROP} px window of a {height}x{width} image leaves the benchmark crop")
    return {
        "fire_pixels": _count_blocks(burning, patch),
        "centre_pixels": _count_blocks(in_centre, patch),
        "centre_fire": _count_blocks(burning & in_centre, patch),
    }


def _build_one(task: tuple[Path, Path, int]) -> None:
    fire_dir, output_path, patch = task
    data, transform = read_fire(fire_dir)
    rows, cols = benchmark_crop(*data.shape[-2:])
    crop_transform = transform * Affine.translation(cols.start, rows.start)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        features=node_features(data[:, :, rows, cols], crop_transform, patch),
        **node_targets(data, rows, cols, patch),
    )


def build_patch_cache(
    raw_dir: str | Path,
    output_dir: str | Path,
    patch: int = 4,
    workers: int = 8,
    overwrite: bool = False,
) -> int:
    """Convert every downloaded fire (year/fire/*.tif), centre-cropped like the benchmark test
    images, into node features and pixel-level targets."""

    tasks = []
    for fire_dir in sorted(path for path in Path(raw_dir).glob("*/*") if path.is_dir()):
        output_path = Path(output_dir) / fire_dir.parent.name / f"{fire_dir.name}.npz"
        if overwrite or not output_path.exists():
            tasks.append((fire_dir, output_path, patch))
    with Pool(workers) as pool:
        for _ in tqdm(pool.imap_unordered(_build_one, tasks), total=len(tasks), desc="Building patches", unit="fire"):
            pass
    return len(tasks)


def load_patch_cache(cache_dir: str | Path, years: Iterable[int]) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """Load cached fires as {(year, fire_name): arrays}; `features` is [T, C, H, W]."""

    fires = {}
    for year in years:
        year_dir = Path(cache_dir) / str(year)
        if not year_dir.is_dir():
            raise FileNotFoundError(f"Patch cache does not exist: {year_dir}")
        for path in sorted(year_dir.glob("*.npz")):
            with np.load(path) as arrays:
                fires[(int(year), path.stem)] = dict(arrays)
    return fires


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark-aligned node features and targets from the GeoTIFFs")
    parser.add_argument("--raw-dir", default="data/raw/wildfirespreadts")
    parser.add_argument("--output-dir", default=None, help="Defaults to data/cache/wsts_patch{N}")
    parser.add_argument("--patch", type=int, default=4, help="Patch side in 375 m pixels")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or f"data/cache/wsts_patch{args.patch}"
    built = build_patch_cache(args.raw_dir, output_dir, args.patch, args.workers, args.overwrite)
    print(f"Built {built} fires in {output_dir}")


if __name__ == "__main__":
    main()
