import numpy as np

from .structures import GraphSample, ShiftOperator


NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _flatten_channels(values: np.ndarray) -> np.ndarray:
    return values.transpose(1, 2, 0).reshape(-1, values.shape[0])


def create_graph_sample(sample: dict) -> GraphSample:
    """Flatten one normalized HDF5 sample into row-major node arrays."""

    grid_shape = tuple(sample["grid_shape"])
    features = _flatten_channels(sample["features"]).astype(np.float32)
    current_fire = sample["current_fire"].reshape(-1).astype(np.float32)
    current_fire_fraction = (
        sample["current_fire_fraction"].reshape(-1).astype(np.float32)
    )
    coverage = sample["coverage"].reshape(-1).astype(np.float32)
    if np.any(~np.isfinite(coverage)) or np.any(coverage <= 0):
        raise ValueError("Every graph node must have positive finite coverage")

    x = np.column_stack(
        (features, current_fire, current_fire_fraction, coverage)
    ).astype(np.float32)
    return GraphSample(
        x=x,
        y=sample["target"].reshape(-1).astype(np.uint8),
        fire_signal=current_fire.astype(np.uint8),
        current_fire_fraction=current_fire_fraction,
        target_fire_fraction=sample["next_fire_fraction"]
        .reshape(-1)
        .astype(np.float32),
        positions=_flatten_channels(sample["positions"]),
        physics=_flatten_channels(sample["physics"]).astype(np.float32),
        coverage=coverage,
        feature_names=(
            *sample["feature_names"],
            "current_fire",
            "current_fire_fraction",
            "coverage",
        ),
        physics_names=sample["physics_names"],
        grid_shape=grid_shape,
        sample_id=sample["sample_id"],
    )


def build_eight_neighbor_edges(grid_shape: tuple[int, int]) -> np.ndarray:
    """Build directed eight-neighbor edges."""

    rows, columns = grid_shape
    if rows < 1 or columns < 1:
        raise ValueError("grid_shape dimensions must be positive")

    sources = []
    targets = []
    for row in range(rows):
        for column in range(columns):
            source = row * columns + column
            for row_offset, column_offset in NEIGHBOR_OFFSETS:
                target_row = row + row_offset
                target_column = column + column_offset
                if 0 <= target_row < rows and 0 <= target_column < columns:
                    sources.append(source)
                    targets.append(target_row * columns + target_column)
    return np.asarray((sources, targets), dtype=np.int64)


def build_grid_shift(grid_shape: tuple[int, int]) -> ShiftOperator:
    """Build an eight-neighbor shift with equal weights."""

    edge_index = build_eight_neighbor_edges(grid_shape)
    return ShiftOperator(
        edge_index=edge_index,
        edge_weight=np.ones(edge_index.shape[1], dtype=np.float32),
        number_of_nodes=grid_shape[0] * grid_shape[1],
    )
