"""The training cache: uncompressed, memory-mappable copies of the node grids and the pixel cache.

The HDF5 node grids are lzf-compressed, so a training process would decompress 13 GB on start and
hold its own standardised copy in RAM. This cache stores the raw model inputs once as .npy files
(one directory per fire); training processes map them read-only, share the page cache, and
standardise per batch on the GPU.

    uv run python -m graphml_wildfire.data.mmap --cache-dir data/processed/wildfirespreadts_benchmark_b2 \
        --pixel-dir data/cache/pixels --output-dir data/cache/b2
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from .patches import node_features

TARGETS = ("fire_pixels", "centre_pixels", "centre_fire")


def _convert(task: tuple[Path, Path | None, Path]) -> None:
    hdf5, pixels, out = task
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5, "r") as source:
        if not source.attrs.get("benchmark_crop", False):
            raise ValueError(f"{hdf5} was not converted with --benchmark-crop")
        np.save(out / "features.npy", node_features(source).astype(np.float16))
        for name in TARGETS:
            np.save(out / f"{name}.npy", source[name][:].astype(np.int16))
    if pixels is not None:
        with np.load(pixels) as arrays:
            for name in arrays.files:
                np.save(out / f"pix_{name}.npy", arrays[name])


def build(cache_dir: Path, pixel_dir: Path | None, output_dir: Path, workers: int) -> int:
    tasks = []
    for hdf5 in sorted(cache_dir.glob("*/*.hdf5")):
        year, fire = hdf5.parent.name, hdf5.stem
        pixels = pixel_dir / year / f"{fire}.npz" if pixel_dir is not None else None
        out = output_dir / year / fire
        if (out / "features.npy").exists() and (pixels is None or (out / "pix_mask.npy").exists()):
            continue
        tasks.append((hdf5, pixels, out))
    with Pool(workers) as pool:
        for _ in tqdm(pool.imap_unordered(_convert, tasks), total=len(tasks), desc="Converting", unit="fire"):
            pass
    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True, help="node grids from convert_hdf5 --benchmark-crop")
    parser.add_argument("--pixel-dir", default=None, help="pixel cache from data.pixels (needed by the pixel model only)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    built = build(Path(args.cache_dir), Path(args.pixel_dir) if args.pixel_dir else None, Path(args.output_dir), args.workers)
    print(f"Converted {built} fires into {args.output_dir}")


if __name__ == "__main__":
    main()
