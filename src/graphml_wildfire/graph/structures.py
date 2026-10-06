from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraphSample:
    """Model-independent node data for one next-day prediction sample."""

    x: np.ndarray  # node input features: [nodes, features]
    y: np.ndarray  # next day fire presence for nodes: [nodes]
    fire_signal: np.ndarray  # current day fire presence: [nodes]
    current_fire_fraction: (
        np.ndarray
    )  # fraction of node burning on current day: [nodes]
    target_fire_fraction: np.ndarray  # fraction of node burning on next day: [nodes]
    positions: np.ndarray  # spatial coordinates of each node [nodes, 2]
    physics: np.ndarray  # raw physics-related features [nodes, physics_features]
    coverage: np.ndarray  # # fraction of the cell that contains valid pixels (invalid pixels are the ones added as padding)
    feature_names: tuple[str, ...]
    physics_names: tuple[str, ...]
    grid_shape: tuple[int, int]
    sample_id: str

    def __post_init__(self) -> None:
        number_of_nodes = self.grid_shape[0] * self.grid_shape[1]
        if self.x.ndim != 2 or self.x.shape[0] != number_of_nodes:
            raise ValueError("x must have shape [number_of_nodes, number_of_features]")
        if self.x.shape[1] != len(self.feature_names):
            raise ValueError("feature_names must describe every column in x")
        if self.physics.shape != (number_of_nodes, len(self.physics_names)):
            raise ValueError(
                "physics must have shape [number_of_nodes, physics_features]"
            )
        if self.positions.shape != (number_of_nodes, 2):
            raise ValueError("positions must have shape [number_of_nodes, 2]")
        for name in (
            "y",
            "fire_signal",
            "current_fire_fraction",
            "target_fire_fraction",
            "coverage",
        ):
            if getattr(self, name).shape != (number_of_nodes,):
                raise ValueError(f"{name} must have shape [number_of_nodes]")

    @property
    def number_of_nodes(self) -> int:
        return self.x.shape[0]


@dataclass(frozen=True)
class ShiftOperator:
    """Sparse directed graph shift with S[target, source] = edge_weight."""

    edge_index: np.ndarray  # source and target node for each egde: [2, E]
    edge_weight: np.ndarray  # single weight for each edge: [E]
    number_of_nodes: int
    edge_attr: np.ndarray | None = (
        None  # optional additional properties for each edge: [E, D]
    )
    edge_attr_names: tuple[
        str, ...
    ] = ()  # names corresponding to the columns of `edge_attr`

    def __post_init__(self) -> None:
        if self.number_of_nodes < 1:
            raise ValueError("number_of_nodes must be positive")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, number_of_edges]")
        number_of_edges = self.edge_index.shape[1]
        if self.edge_weight.shape != (number_of_edges,):
            raise ValueError("edge_weight must have shape [number_of_edges]")
        if np.any(self.edge_index < 0) or np.any(
            self.edge_index >= self.number_of_nodes
        ):
            raise ValueError("edge_index contains an invalid node ID")
        if self.edge_attr is None:
            if self.edge_attr_names:
                raise ValueError("edge_attr_names require edge_attr")
        elif self.edge_attr.shape != (number_of_edges, len(self.edge_attr_names)):
            raise ValueError("edge_attr and edge_attr_names have incompatible shapes")
