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

All results use the 12-fold protocol of the **WildfireSpreadTS** benchmark ([Gerard et al., NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf)), so our rows are directly comparable to the published ones.

### Leaderboard

Test AP is the mean ± std over the 12 year-wise folds. **Compare only within a column.**

- **AP (full):** every 375 m pixel of the test images (Gerard et al.).
- **AP (centre 128):** only each test image's central 128 × 128 pixels. This is where [Lahrichi et al., WACV 2026](https://arxiv.org/abs/2502.12003) evaluate, per their [code](https://github.com/slahrichi/WildfireSpreadTS). That window contains more fire, so APs there are higher.
- **F1 and IoU:** at probability 0.5, the benchmark convention.
- **Published rows:** use all features, except those marked "best", which use the best feature set reported.
- **Ours:** use all features. `Days` is the number of input days.

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

**Rows:**
- *Gerard* and *Lahrichi* are the published results; see their papers for the training recipes.
- `persistence`, `logreg`, `mlp` and `cnn` are our fast baselines (`scripts/run_baselines.sh`):
  - `persistence`: tomorrow's burning fraction of a node is today's.
  - `logreg`: per-node logistic regression.
  - `mlp`: per-node MLP.
  - `cnn`: small U-Net over the node grid.
- Our graph models are added as their 12-fold runs finish.

**What the baselines show:**
- **Coarse 1.5 km nodes stay competitive.** The node U-Net (0.388) is above the best published full-image result (UTAE, 0.372), and our per-node logistic regression (0.301–0.310) is above the published pixel-level one (0.286). The training recipes differ too: we use pixel-level BCE and select models by validation AP.
- **The big gains in Lahrichi et al. likely come from their training recipe, not the smaller test window.** Their centre-128 numbers are ~0.1 higher than Gerard's full-image ones. For our models the centre window adds only ~0.01, which suggests most of their gain comes from focal loss, pretrained encoders and selecting models by validation AP.
- **F1 and IoU at 0.5 are low for our trained models.** Plain pixel-level BCE gives calibrated, low probabilities. `results/benchmark_runs.jsonl` also stores F1/IoU at the validation-tuned threshold (`test_f1_tuned`, `test_iou_tuned`): mean F1 0.41–0.45.

### Protocol

The protocol was read from the official code ([SebastianGer/WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS)) and is implemented in `benchmark.py` and `data/patches.py`:

- **Folds:** 12 year-wise folds over 2018–2021 (two training years, one validation year, one test year), in the official order.
- **Samples:** for every test fire with `T` days, the targets are days 5..T−1. That is 10,572 targets in total, the same for 1-day and 5-day models.
- **Inputs:** at most the 5 days before the target.
- **Target:** the next-day active-fire mask at 375 m. A pixel burns if its detection time (hhmm) floor-divided by 100 is > 0, as in the official HDF5 files.
- **Test pixels:** the whole image, centre-cropped to multiples of 32.
- **Metric:** exact AP (`torchmetrics.BinaryAveragePrecision`), pooled over all test pixels of the test year. This gives one AP per fold. The reported std is the population std (ddof=0).

**Check:** `uv run python -m graphml_wildfire.benchmark persistence` reproduces the published pixel-level persistence of Gerard et al. (Table 2) to the third decimal:

| Test year | 2018 | 2019 | 2020 | 2021 | 12 folds |
| --- | --- | --- | --- | --- | --- |
| This repository | 0.187 | 0.105 | 0.194 | 0.287 | 0.193 ± 0.065 |
| Gerard et al. | 0.187 | 0.105 | 0.194 | 0.287 | 0.193 ± 0.065 |

### How our models are scored

`data/patches.py` turns every fire into a grid of 4 × 4 pixel nodes (1.5 km):

- Each daily GeoTIFF is centre-cropped exactly like a benchmark test image.
- The crop is aggregated with `preprocess_image`.
- Fire-history features are added. They only look at the 5 input days.
- Per-node counts of the burning pixels the benchmark scores are stored alongside.

A node's score is given to all its pixels. AP computed from these counts equals pixel-level AP exactly (`tests/test_benchmark.py`).

Models are trained with pixel-level BCE, where the target is the fraction of a node's pixels that burn. A 1.5 km node cannot rank the pixels inside it, which costs some AP compared with 375 m models.

### Reproduce and add a model

```bash
uv run python -m graphml_wildfire.data.patches            # data/raw GeoTIFFs -> data/cache/wsts_patch4 (~2 min)
uv run python -m graphml_wildfire.benchmark persistence   # protocol check (optional)
scripts/run_baselines.sh                                  # the baselines, 12 folds each (~1 h on one GPU)
uv run python -m graphml_wildfire.benchmark summary       # prints the leaderboard as markdown
```

To add a model:
1. Run it on all 12 folds, appending one JSON line per fold to `results/benchmark_runs.jsonl`. `train.py` shows the required fields.
2. Paste its row from `benchmark summary` into the leaderboard above.

### Not comparable

These papers use WildfireSpreadTS with a different protocol, so their numbers do not belong in the leaderboard:

- **WSTS+** (Lahrichi et al., Table 5): 8 years, 4 folds.
- **Random event-level 4-fold CV** (Lahrichi et al., Table 8).
- **WildfireSpreadBench** ([arXiv 2609.22191](https://arxiv.org/abs/2609.22191)): fold 0 only.
- **Funk** ([arXiv 2605.03148](https://arxiv.org/abs/2605.03148)): fold ensembles, std over 4 years.
- **Yang et al.** ([arXiv 2603.09042](https://arxiv.org/abs/2603.09042)): leave-one-year-out, scenario subsets, 64 × 64 crops.
- **Luo et al.** ([arXiv 2602.11672](https://arxiv.org/abs/2602.11672)) and **Meco et al.** ([arXiv 2606.14071](https://arxiv.org/abs/2606.14071)): F1 only, splits not stated.
- **Chakravarty** ([arXiv 2510.09666](https://arxiv.org/abs/2510.09666)): claims the 12-fold protocol, but its persistence (0.191) does not match the benchmark's.
