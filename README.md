# GraphMLWildfire

Results are compared on the WildfireSpreadTS benchmark: see the [leaderboard](#benchmark).

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
features, coverage, positions, dates, and geospatial metadata.

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
)
training_statistics = compute_statistics(training_dataset)
normalized_training_dataset = NormalizedWildfireSpreadTSDataset(
    training_dataset,
    training_statistics,
)

sample = normalized_training_dataset[0]
print(sample["features"].shape)
print(sample["target"].shape)
```

`create_year_folds()` returns the 12 two-train, one-validation, one-test folds
over 2018-2021, in the order of the official WildfireSpreadTS code, so fold `i`
uses the same years as fold `i` in the published papers. Graph edges and shift operators are intentionally constructed
after this data pipeline.

## Build Graphs

Convert a normalized node grid into arrays, then build
the grid shift operator:

```python
from graphml_wildfire.graph import (
    build_grid_shift,
    create_graph_sample,
)

graph_sample = create_graph_sample(normalized_training_dataset[0])
grid_shift = build_grid_shift(graph_sample.grid_shape)
```

`GraphSample.x` has shape `[N, 45]`: the 42 preprocessed model features plus
current-fire presence, current-fire fraction, and coverage. Targets, positions,
raw physics values, and fire fractions remain available as separate arrays.

Each `ShiftOperator` stores the nonzero entries of `S` without constructing a
dense matrix:

```text
edge_index:  [2, E]
edge_weight: [E]
edge_attr:   [E, D] or None
```

For edge `i -> j`, the convention is `S[j, i] = edge_weight`. The grid operator
uses the directed eight-neighbor topology and gives every
edge raw weight `1`.

## Benchmark

We use the 12-fold protocol of the WildfireSpreadTS benchmark ([Gerard et al., 2023](https://papers.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf)): year-wise folds over 2018–2021, targets on days 5..T−1, and AP pooled over all 375 m pixels of each test year. Results are reported as mean ± std over the 12 folds.

**AP (full)** scores the whole test image (Gerard et al.). **AP (centre 128)** scores only each image's central 128 × 128 pixels, which is what [Lahrichi et al., 2026](https://arxiv.org/abs/2502.12003) evaluate. Compare only within a column. F1 and IoU are at probability 0.5.

| Model | Days | Params | Folds | AP (full) | AP (centre 128) | F1 | IoU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Published** | | | | | | | |
| Gerard: persistence | 1 | 0 | 12 | 0.193 ± 0.065 | – | | |
| Gerard: logistic regression | 1 | 361 | 12 | 0.286 ± 0.092 | – | | |
| Gerard: Res18 U-Net | 1 | 14.4M | 12 | 0.341 ± 0.086 | – | | |
| Gerard: Res18 U-Net | 5 | 14.7M | 12 | 0.325 ± 0.108 | – | | |
| Gerard: ConvLSTM | 5 | 240K | 12 | 0.292 ± 0.094 | – | | |
| Gerard: UTAE | 5 | 1.1M | 12 | 0.321 ± 0.135 | – | | |
| Gerard: UTAE (Veg, best) | 5 | 1.1M | 12 | 0.372 ± 0.088 | – | | |
| Lahrichi: Res18-UNet | 1 | 14.3M | 12 | – | 0.460 ± 0.084 | | |
| Lahrichi: Res18-UNet (Multi, best T=1) | 1 | 14.3M | 12 | – | 0.468 ± 0.087 | | |
| Lahrichi: UTAE(Res18) | 5 | 14.6M | 12 | – | 0.475 ± 0.091 | | |
| Lahrichi: UTAE(Res18) (Veg, best) | 5 | 14.6M | 12 | – | 0.478 ± 0.085 | | |
| **Ours: baselines (1.5 km nodes, this repository)** | | | | | | | |
| persistence_w5_p4 | 5 | 0 | 12 | 0.246 ± 0.077 | 0.246 ± 0.068 | 0.381 ± 0.068 | 0.237 ± 0.051 |
| logreg_w1_p4 | 1 | 50 | 12 | 0.301 ± 0.085 | 0.304 ± 0.076 | 0.187 ± 0.076 | 0.105 ± 0.046 |
| logreg_w5_p4 | 5 | 246 | 12 | 0.310 ± 0.080 | 0.315 ± 0.071 | 0.230 ± 0.069 | 0.132 ± 0.044 |
| mlp_w5_p4 | 5 | 28,289 | 12 | 0.351 ± 0.077 | 0.362 ± 0.067 | 0.310 ± 0.072 | 0.186 ± 0.051 |
| cnn_w5_p4 | 5 | 2,025,089 | 12 | 0.388 ± 0.072 | 0.397 ± 0.061 | 0.277 ± 0.055 | 0.162 ± 0.037 |
| **Ours: graph models** | | | | | | | |
| *to be added* | | | | | | | |

Our models predict on 4 × 4 pixel nodes (1.5 km). Each node's score is given to all its pixels, so the metric is exactly the pixel-level one. `benchmark persistence` reproduces Gerard et al.'s persistence AP exactly (0.193 ± 0.065).

```bash
uv run python -m graphml_wildfire.data.patches            # build the node cache from data/raw
scripts/run_baselines.sh                                  # baselines, 12 folds each (~1 h)
uv run python -m graphml_wildfire.benchmark summary       # leaderboard rows as markdown
```

To add a model, run it on all 12 folds and write its results to `results/benchmark_runs.jsonl`, then paste its row from `benchmark summary` into the table.
