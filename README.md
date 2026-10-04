# GraphMLWildfire

## Setup

```bash
uv sync
```

### Download the dataset

For development, download and convert one small fire:

```bash
uv run python -m graphml_wildfire.data.download_sample \
  --year 2018 \
  --fire fire_22343689
```

Omit the fire ID to download and convert every fire from one year:

```bash
uv run python -m graphml_wildfire.data.download_sample --year 2019
```

To download and convert the full dataset:

```bash
uv run python -m graphml_wildfire.data.download_sample --full-dataset
```

The full download is approximately 50 GB and may take several hours. All
download commands keep the raw TIFFs and create HDF5 files under
`data/processed/wildfirespreadts`.

### Load the data

```python
from graphml_wildfire.data import WildfireSpreadTSDataset

dataset = WildfireSpreadTSDataset(
    "data/processed/wildfirespreadts",
    years=[2018],
)
sample = dataset[0]

print(sample["features"].shape)
print(sample["target"].shape)
```
