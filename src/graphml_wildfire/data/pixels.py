"""375 m pixel inputs and targets for the benchmark crop of every fire.

Node models cannot rank the pixels inside a node. This cache holds what the pixel encoder and
decoder need on top of the node grids: per day, the active-fire mask, the detection time and the
five native 375 m vegetation bands of every pixel, and where the centre 128 x 128 window lies.

    uv run python -m graphml_wildfire.data.pixels --raw-dir data/raw/wildfirespreadts --output-dir data/cache/pixels
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

from .channels import ACTIVE_FIRE_INDEX, BASE_FEATURE_NAMES
from .convert_hdf5 import CENTRE_CROP, _centre_crop_start, _date_from_filename, benchmark_crop, burning_pixels, detection_time_hhmm

# Native 375 m bands stored per pixel besides the fire mask.
BANDS = ("ndvi", "viirs_m11", "viirs_i2", "viirs_i1", "evi2")
# Pixel input channels of the model, per day: burning (0/1), detection time (hhmm / 2400, 0 without fire), the bands.
PIXEL_CHANNELS = ("mask", "time") + BANDS


def build_one(task: tuple[Path, Path]) -> None:
    fire_dir, output_path = task
    masks, times, values = [], [], {band: [] for band in BANDS}
    band_indices = [BASE_FEATURE_NAMES.index(band) + 1 for band in BANDS]
    for path in sorted(fire_dir.glob("*.tif"), key=_date_from_filename):
        with rasterio.open(path) as source:
            read = source.read(band_indices + [ACTIVE_FIRE_INDEX + 1]).astype(np.float32)
        active_fire = detection_time_hhmm(read[-1])
        height, width = active_fire.shape
        rows, cols = benchmark_crop(height, width)
        burning = burning_pixels(active_fire[rows, cols])
        masks.append(burning)
        times.append(np.where(burning, np.nan_to_num(active_fire[rows, cols]) / 2400.0, 0.0).astype(np.float16))
        for band, image in zip(BANDS, read[:-1]):
            values[band].append(image[rows, cols].astype(np.float16))
    if not masks:
        return  # a directory without rasters, converted to no node grid either
    # The centre window in cropped-pixel coordinates (it always lies inside the benchmark crop).
    centre = (_centre_crop_start(height, CENTRE_CROP) - rows.start, _centre_crop_start(width, CENTRE_CROP) - cols.start)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, mask=np.stack(masks), time=np.stack(times), centre=np.array(centre),
             **{band: np.stack(stack) for band, stack in values.items()})


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
