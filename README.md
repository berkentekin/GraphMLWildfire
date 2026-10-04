# GraphMLWildfire

## Setup

```bash
uv sync
```

## Download

Downloading only extracts the original GeoTIFF files. Download one fire:

```bash
uv run python -m graphml_wildfire.data.download_sample \
  --year 2018 \
  --fire fire_22343689
```

Omit `--fire` to download a complete year, or use `--full-dataset` for the
approximately 50 GB archive.

## Preprocess

Preprocess every downloaded fire into node grids:

```bash
uv run python -m graphml_wildfire.data.convert_hdf5 \
  data/raw/wildfirespreadts \
  data/processed/wildfirespreadts
```

The default preprocessing block size is `4`, so each node aggregates a `4 x 4`
patch of approximately 375 m pixels and has a side length of 1.5 km. It can be changed,
for example to `2 x 2` patches, with:

```bash
uv run python -m graphml_wildfire.data.convert_hdf5 \
  data/raw/wildfirespreadts \
  data/processed/wildfirespreadts \
  --block-size 2
```

The node side length is `block size x 375 m`.

Each HDF5 file contains all days for one fire. It stores aggregated model
features, land-cover proportions, fire presence and fraction, raw physics
features, coverage, positions, dates, and geospatial metadata. Preprocessing
prints total and per-fire elapsed time.

Here, `D` is the number of days for the fire, while `R` and `C` are the node
grid dimensions.

| Dataset | Shape | Type | Contents |
| --- | --- | --- | --- |
| `features` | `[D, 42, R, C]` | `float32` | Unnormalized model features: aggregated continuous values, sine/cosine direction channels, and 17 land-cover proportions plus unknown. |
| `physics` | `[D, 7, R, C]` | `float32` | Wind speed and direction, forecast wind speed and direction, elevation, slope, and aspect in their original units. |
| `fire_presence` | `[D, R, C]` | `uint8` | Binary indicator that at least one pixel in the node contains an active-fire detection. |
| `fire_fraction` | `[D, R, C]` | `float32` | Proportion of the node's real pixels containing an active-fire detection. |
| `coverage` | `[R, C]` | `float32` | Fraction of a complete block covered by real raster pixels. Partial border nodes have values below one. |
| `positions` | `[2, R, C]` | `float64` | `x` and `y` coordinates of each node center. |

File attributes store the year, fire ID, dates, CRS, transforms, block size,
original raster shape, and channel names.

## Select Fires

Create one reproducible manifest that is reused across every year-wise fold:

```bash
uv run python -m graphml_wildfire.data.create_manifest \
  data/processed/wildfirespreadts \
  data/manifests/selected_fires.csv \
  --fraction 0.2 \
  --seed 42
```

This also allows us to reduce the size of the dataset by specifying the `fraction` argument.

## Normalize

For each cross-validation fold, compute statistics from only its two training
years once and keep them in memory. Reuse the same statistics object for the
training, validation, and test datasets. Statistics from validation or test
years must never be included.

```python
from graphml_wildfire.data import (
    NormalizedWildfireSpreadTSDataset,
    WildfireSpreadTSDataset,
    compute_statistics,
)

current_fold_training_years = [2018, 2019]

training_dataset = WildfireSpreadTSDataset(
    "data/processed/wildfirespreadts",
    years=current_fold_training_years,
    fire_manifest="data/manifests/selected_fires.csv",
    report_timing=True,
)
training_statistics = compute_statistics(raw_dataset)
normalized_training_dataset = NormalizedWildfireSpreadTSDataset(
    training_dataset,
    training_statistics,
    report_timing=True,
)

sample = normalized_training_dataset[0]
print(sample["features"].shape)
print(sample["target"].shape)
```

`create_year_folds()` returns the 12 two-train, one-validation, one-test folds
over 2018-2021. Graph edges and shift operators are intentionally constructed
after this data pipeline.
