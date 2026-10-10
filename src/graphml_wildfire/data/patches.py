"""Model inputs and benchmark targets from the HDF5 node grids written by `convert_hdf5`.

Convert the fires with `--benchmark-crop`, so the nodes tile exactly the pixels the
WildfireSpreadTS benchmark scores, then load them with `load_patch_cache`.
"""

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from .channels import PROCESSED_FEATURE_NAMES


# Standardised per fold with training-year statistics.
CONTINUOUS = [
    name for name in PROCESSED_FEATURE_NAMES if not name.endswith(("_sin", "_cos")) and not name.startswith("land_cover_")
]
# Unit vectors in array coordinates (row grows southward, col grows eastward), from the circular
# block means of the direction channels. wind_* points downwind, aspect_* points downslope.
VECTOR = ["wind_row", "wind_col", "aspect_row", "aspect_col", "fc_wind_row", "fc_wind_col"]
# Orientation scalars: aspect and wind relative to north and to each other (south-facing slopes ignite
# measurably less often; these survive the flip/transpose augmentation, which the raw vectors do not).
ORIENTATION = ["aspect_north", "wind_north", "aspect_wind", "fc_wind_north"]
LANDCOVER = [name for name in PROCESSED_FEATURE_NAMES if name.startswith("land_cover_")]
MASKS = ["viirs_missing", "ndvi_missing"]
# fire_frac/fire_time describe the current day; the rest summarise the HISTORY days up to it,
# so a node never sees further back than the 5-day models of the WildfireSpreadTS benchmark.
FIRE = ["fire_frac", "fire_time", "burned_ever", "recency_2d", "recency_7d"]
HISTORY = 5
CHANNELS = CONTINUOUS + VECTOR + ORIENTATION + LANDCOVER + MASKS + FIRE
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNELS)}

_LOG_SCALED = {"total_precipitation", "forecast_total_precipitation"}
_TARGETS = ("fire_pixels", "centre_pixels", "centre_fire")


def node_features(source: h5py.File) -> np.ndarray:
    """Model inputs [T, len(CHANNELS), H, W] of one converted fire."""

    named = dict(zip(PROCESSED_FEATURE_NAMES, source["features"][:].transpose(1, 0, 2, 3)))
    out = {name: named[name] for name in CONTINUOUS + LANDCOVER}
    for name in _LOG_SCALED:
        out[name] = np.log1p(np.maximum(out[name], 0))
    # Directions are where the wind comes from and where the slope faces, clockwise from north.
    for prefix, name in (("", "wind_direction"), ("fc_", "forecast_wind_direction")):
        out[f"{prefix}wind_row"] = np.nan_to_num(named[f"{name}_cos"])
        out[f"{prefix}wind_col"] = np.nan_to_num(-named[f"{name}_sin"])
    out["aspect_row"] = np.nan_to_num(-named["aspect_cos"])
    out["aspect_col"] = np.nan_to_num(named["aspect_sin"])
    # north = (-1, 0) in (row, col) coordinates: the row index grows southward.
    out["aspect_north"] = -out["aspect_row"]
    out["wind_north"] = -out["wind_row"]
    out["fc_wind_north"] = -out["fc_wind_row"]
    out["aspect_wind"] = out["aspect_row"] * out["wind_row"] + out["aspect_col"] * out["wind_col"]
    for name in MASKS:
        out[name] = source[name][:]

    fire_frac = source["fire_fraction"][:]
    out["fire_frac"] = fire_frac
    out["fire_time"] = np.nan_to_num(source["fire_time"][:]) / 2400.0
    burned_ever, recency = [], {2: [], 7: []}
    for day in range(len(fire_frac)):
        history = fire_frac[max(day - HISTORY + 1, 0) : day + 1][::-1]
        burned_ever.append(history.max(axis=0))
        # Days since the last fire within the history; never burning counts as infinitely long ago.
        days_since = np.where((history > 0).any(axis=0), (history > 0).argmax(axis=0), np.inf)
        for scale, values in recency.items():
            values.append(np.exp(-days_since / scale))
    out["burned_ever"] = np.stack(burned_ever)
    out["recency_2d"] = np.stack(recency[2])
    out["recency_7d"] = np.stack(recency[7])

    return np.stack([out[name] for name in CHANNELS], axis=1).astype(np.float32)


def load_patch_cache(data_dir: str | Path, years: Iterable[int]) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """Load converted fires as {(year, fire_name): arrays}: `features` [T, C, H, W] and the
    per-node pixel counts `fire_pixels` [T, H, W], `centre_pixels` [H, W], `centre_fire` [T, H, W]."""

    fires = {}
    for year in years:
        year_dir = Path(data_dir) / str(year)
        if not year_dir.is_dir():
            raise FileNotFoundError(f"Converted fires do not exist: {year_dir}")
        for path in sorted(year_dir.glob("*.hdf5")):
            with h5py.File(path, "r") as source:
                if not source.attrs.get("benchmark_crop", False):
                    raise ValueError(f"{path} was not converted with --benchmark-crop")
                fires[(int(year), path.stem)] = {
                    "features": node_features(source),
                    **{name: source[name][:] for name in _TARGETS},
                }
    return fires
