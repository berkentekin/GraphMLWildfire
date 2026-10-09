"""375 m pixel inputs and targets for the benchmark crop of every fire.

Node models cannot rank the pixels inside a node. This cache holds what a pixel-level
encoder/decoder needs on top of the node grids: the active-fire mask, the detection time
and NDVI of every pixel, and where the centre 128 x 128 window lies.

    uv run python -m graphml_wildfire.data.pixels --raw-dir data/raw/wildfirespreadts --output-dir data/cache/pixels
"""

import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from tqdm import tqdm

from .channels import ACTIVE_FIRE_INDEX, BASE_FEATURE_NAMES
from .convert_hdf5 import CENTRE_CROP, _centre_crop_start, _date_from_filename, benchmark_crop, burning_pixels

NDVI_INDEX = BASE_FEATURE_NAMES.index("ndvi")
# Pixel input channels, per day: burning (0/1), detection time (hhmm / 2400, 0 without fire), NDVI.
PIXEL_CHANNELS = ("mask", "time", "ndvi")


def build_one(task: tuple[Path, Path]) -> None:
    fire_dir, output_path = task
    masks, times, ndvis = [], [], []
    for path in sorted(fire_dir.glob("*.tif"), key=_date_from_filename):
        with rasterio.open(path) as source:
            ndvi, active_fire = source.read([NDVI_INDEX + 1, ACTIVE_FIRE_INDEX + 1]).astype(np.float32)
        height, width = active_fire.shape
        rows, cols = benchmark_crop(height, width)
        burning = burning_pixels(active_fire[rows, cols])
        masks.append(burning)
        times.append(np.where(burning, np.nan_to_num(active_fire[rows, cols]) / 2400.0, 0.0).astype(np.float16))
        ndvis.append(ndvi[rows, cols].astype(np.float16))
    # The centre window in cropped-pixel coordinates (it always lies inside the benchmark crop).
    centre = (_centre_crop_start(height, CENTRE_CROP) - rows.start, _centre_crop_start(width, CENTRE_CROP) - cols.start)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, mask=np.stack(masks), time=np.stack(times), ndvi=np.stack(ndvis), centre=np.array(centre))


def build_pixel_cache(raw_dir: str | Path, output_dir: str | Path, workers: int = 8, overwrite: bool = False) -> int:
    tasks = []
    for fire_dir in sorted(path for path in Path(raw_dir).glob("*/*") if path.is_dir()):
        output_path = Path(output_dir) / fire_dir.parent.name / f"{fire_dir.name}.npz"
        if overwrite or not output_path.exists():
            tasks.append((fire_dir, output_path))
    with Pool(workers) as pool:
        for _ in tqdm(pool.imap_unordered(build_one, tasks), total=len(tasks), desc="Building pixels", unit="fire"):
            pass
    return len(tasks)


def load_pixel_cache(cache_dir: str | Path, years: Iterable[int]) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """{(year, fire_name): mask [T, H, W] bool, time/ndvi [T, H, W] float16, centre [2]}."""

    fires = {}
    for year in years:
        for path in sorted((Path(cache_dir) / str(year)).glob("*.npz")):
            with np.load(path) as arrays:
                fires[(int(year), path.stem)] = dict(arrays)
    return fires


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 375 m pixel inputs and targets for the benchmark crop")
    parser.add_argument("--raw-dir", default="data/raw/wildfirespreadts")
    parser.add_argument("--output-dir", default="data/cache/pixels")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    built = build_pixel_cache(args.raw_dir, args.output_dir, args.workers, args.overwrite)
    print(f"Built {built} fires in {args.output_dir}")


if __name__ == "__main__":
    main()
