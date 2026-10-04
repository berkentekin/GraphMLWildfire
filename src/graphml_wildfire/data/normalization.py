from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .wildfirespreadts import WildfireSpreadTSDataset


@dataclass(frozen=True)
class FeatureStatistics:
    feature_names: tuple[str, ...]
    continuous_feature_indices: tuple[int, ...]
    means: np.ndarray
    standard_deviations: np.ndarray


def compute_statistics(
    dataset: WildfireSpreadTSDataset,
    show_progress: bool = True,
) -> FeatureStatistics:
    """Compute node-feature statistics from exactly the dataset's input samples."""

    if not dataset.pairs:
        raise ValueError("Cannot compute statistics from an empty dataset")

    sums = None
    squared_sums = None
    counts = None
    feature_names = None
    continuous_indices = None
    progress = tqdm(
        range(len(dataset)),
        desc="Computing statistics",
        unit="sample",
        disable=not show_progress,
    )
    for index in progress:
        sample = dataset[index]
        if feature_names is None:
            feature_names = sample["feature_names"]
            continuous_indices = sample["continuous_feature_indices"]
            feature_count = len(feature_names)
            sums = np.zeros(feature_count, dtype=np.float64)
            squared_sums = np.zeros(feature_count, dtype=np.float64)
            counts = np.zeros(feature_count, dtype=np.int64)
        elif sample["feature_names"] != feature_names:
            raise ValueError("Feature names changed between HDF5 files")

        for feature_index in continuous_indices:
            values = sample["features"][feature_index].astype(np.float64)
            values = values[np.isfinite(values)]
            sums[feature_index] += values.sum()
            squared_sums[feature_index] += np.square(values).sum()
            counts[feature_index] += values.size

    means = np.zeros(len(feature_names), dtype=np.float64)
    standard_deviations = np.ones(len(feature_names), dtype=np.float64)
    for feature_index in continuous_indices:
        if counts[feature_index] == 0:
            raise ValueError(
                f"Feature {feature_names[feature_index]} has no valid training values"
            )
        means[feature_index] = sums[feature_index] / counts[feature_index]
        variance = (
            squared_sums[feature_index] / counts[feature_index]
            - means[feature_index] ** 2
        )
        standard_deviations[feature_index] = np.sqrt(max(variance, 0.0)) or 1.0

    return FeatureStatistics(
        feature_names,
        continuous_indices,
        means,
        standard_deviations,
    )


class NormalizedWildfireSpreadTSDataset:
    """Apply fold-specific statistics while retaining graph-independent fields."""

    def __init__(
        self,
        dataset: WildfireSpreadTSDataset,
        statistics: FeatureStatistics,
    ):
        self.dataset = dataset
        self.statistics = statistics
        if self.statistics.means.shape != self.statistics.standard_deviations.shape:
            raise ValueError("Statistics means and standard deviations must have equal shapes")
        if self.statistics.means.shape != (len(self.statistics.feature_names),):
            raise ValueError("Statistics must contain one value per feature")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        if sample["feature_names"] != self.statistics.feature_names:
            raise ValueError("Statistics do not match the dataset feature names")
        if (
            sample["continuous_feature_indices"]
            != self.statistics.continuous_feature_indices
        ):
            raise ValueError("Statistics do not match the continuous feature indices")

        features = sample["features"].copy()
        indices = np.asarray(self.statistics.continuous_feature_indices)
        features[indices] = (
            features[indices]
            - self.statistics.means[indices, None, None]
        ) / self.statistics.standard_deviations[indices, None, None]
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        result = dict(sample)
        result["features"] = features.astype(np.float32)
        result["target"] = sample["next_fire"].astype(np.int64)
        return result
