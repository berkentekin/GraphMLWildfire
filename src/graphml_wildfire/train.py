"""Train and evaluate next-day fire prediction on the benchmark node grid.

Evaluation follows the WildfireSpreadTS benchmark (see `benchmark.py`). A node's score is
shared by all pixels of its patch, so pixel-level AP is computed exactly from per-node counts
of burning and non-burning pixels.

Example:
    uv run python -m graphml_wildfire.train --model logreg --window 5 --fold 0
    uv run python -m graphml_wildfire.train --model pixel_gtcnn --graph attention --patch 2 --eval-batch-size 8 --fold 0
"""

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn

from .benchmark import FIRST_INPUT_DAY, FOLDS, YEARS, average_precision, best_f1_threshold, threshold_metrics
from .data.patches import CHANNEL_INDEX, CHANNELS, CONTINUOUS, load_patch_cache
from .data.pixels import PIXEL_CHANNELS, load_pixel_cache
from .models import ConvNet, GraphTimeCNN, NodeMLP, Persistence, PixelGraphTimeCNN


MODELS = ("gtcnn", "pixel_gtcnn", "cnn", "mlp", "logreg", "persistence")

_ROW_CHANNELS = [CHANNEL_INDEX[name] for name in ("wind_row", "aspect_row", "fc_wind_row")]
_COL_CHANNELS = [CHANNEL_INDEX[name] for name in ("wind_col", "aspect_col", "fc_wind_col")]


@dataclass
class TrainConfig:
    experiment_name: str = ""
    model: str = "logreg"
    graph: str = "grid"  # gtcnn only
    window: int = 5
    hops: int = 1  # gtcnn only
    hidden: int = 64
    layers: int = 4
    taps: int = 3  # gtcnn only
    heads: int = 4  # gtcnn only
    dropout: float = 0.1
    checkpoint_dir: str = "data/checkpoints"
    pixel_dir: str = "data/cache/pixels"  # pixel_gtcnn only
    pixel_hidden: int = 16  # pixel_gtcnn only
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 32
    eval_batch_size: int = 0  # 0: same as batch_size; full images need less at finer nodes
    crop: int = 32
    val_stride: int = 3
    pos_weight: float = 1.0
    fold: int = 0
    seed: int = 0
    patch: int = 4
    cache_dir: str = ""
    results_file: str = "results/benchmark_runs.jsonl"
    wandb_project: str = "graphml-wildfire"

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"Unknown model: {self.model}")
        if not 1 <= self.window <= FIRST_INPUT_DAY + 1:
            raise ValueError(f"window must be between 1 and {FIRST_INPUT_DAY + 1}")
        if not self.eval_batch_size:
            self.eval_batch_size = self.batch_size
        if not self.cache_dir:
            self.cache_dir = f"data/processed/wildfirespreadts_benchmark_b{self.patch}"
        if not self.experiment_name:
            graph = f"_{self.graph}" if self.model in ("gtcnn", "pixel_gtcnn") else ""
            self.experiment_name = f"{self.model}{graph}_w{self.window}_p{self.patch}"
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")

    @property
    def run_name(self) -> str:
        digest = hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.experiment_name}_f{self.fold}_s{self.seed}_{digest}"


class FireData:
    """All fires of the four years in memory, standardised with training-year statistics."""

    def __init__(self, config: TrainConfig):
        fold = FOLDS[config.fold]
        train_a, train_b = fold.train_years
        val_year, test_year = fold.validation_year, fold.test_year
        self.window = config.window
        self.node_pixels = config.patch**2
        continuous = [CHANNEL_INDEX[name] for name in CONTINUOUS]
        sample = np.concatenate(
            [fire["features"][::3, continuous].transpose(1, 0, 2, 3).reshape(len(continuous), -1)
             for fire in load_patch_cache(config.cache_dir, (train_a, train_b)).values()],
            axis=1,
        )
        self.mean = np.zeros(len(CHANNELS), dtype=np.float32)
        self.std = np.ones(len(CHANNELS), dtype=np.float32)
        self.mean[continuous] = np.nanmean(sample, axis=1)
        self.std[continuous] = np.nanstd(sample, axis=1) + 1e-6
        del sample

        self.fires, self.targets, self.samples = [], [], {"train": [], "val": [], "test": []}
        self.centre_windows = []  # node-aligned window around the centre 128 x 128 pixels of every fire
        # Pixel-level inputs and targets of every fire, only for the pixel model.
        self.patch = config.patch
        self.pixels = [] if config.model == "pixel_gtcnn" else None
        split_of = {train_a: "train", train_b: "train", val_year: "val", test_year: "test"}
        for year in YEARS:
            pixels = load_pixel_cache(config.pixel_dir, [year]) if self.pixels is not None else {}
            for key, arrays in load_patch_cache(config.cache_dir, [year]).items():
                fire = (arrays["features"] - self.mean[None, :, None, None]) / self.std[None, :, None, None]
                self.fires.append(torch.from_numpy(fire.astype(np.float16)))
                if self.pixels is not None:
                    self.pixels.append(self._pixel_arrays(pixels[key], fire.shape[2:]))
                self.targets.append({
                    name: torch.from_numpy(arrays[name].astype(np.int16))
                    for name in ("fire_pixels", "centre_pixels", "centre_fire")
                })
                index = len(self.fires) - 1
                self.samples[split_of[year]] += [(index, day) for day in range(FIRST_INPUT_DAY, len(fire) - 1)]
                rows, cols = (torch.nonzero(self.targets[index]["centre_pixels"].sum(dim=axis) > 0).flatten() for axis in (1, 0))
                self.centre_windows.append((int(rows[0]), int(cols[0]), int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1)))
        burning = sum(float(self.targets[index]["fire_pixels"][day + 1].sum()) for index, day in self.samples["train"])
        pixels = sum(self.fires[index][0, 0].numel() * self.node_pixels for index, _ in self.samples["train"])
        self.fire_rate = burning / pixels

    def _pixel_arrays(self, arrays: dict[str, np.ndarray], grid: tuple[int, int]) -> dict[str, torch.Tensor]:
        """Pixel inputs [T, len(PIXEL_CHANNELS), Hp, Wp] (NDVI standardised like the node feature),
        the burning mask [T, Hp, Wp] and the centre-128 window [Hp, Wp] of one fire."""

        mask = arrays["mask"]
        if mask.shape[1:] != (grid[0] * self.patch, grid[1] * self.patch):
            raise ValueError(f"Pixel grid {mask.shape[1:]} does not match the node grid {grid} x {self.patch}")
        ndvi = CHANNEL_INDEX["ndvi"]
        inputs = np.stack([
            mask.astype(np.float16),
            arrays["time"],
            np.nan_to_num((arrays["ndvi"].astype(np.float32) - self.mean[ndvi]) / self.std[ndvi]).astype(np.float16),
        ], axis=1)
        centre = torch.zeros(mask.shape[1:], dtype=torch.bool)
        top, left = (int(value) for value in arrays["centre"])
        centre[top : top + 128, left : left + 128] = True
        return {"inputs": torch.from_numpy(inputs), "mask": torch.from_numpy(mask), "centre": centre}

    def batch(
        self,
        samples: list[tuple[int, int]],
        device: torch.device,
        crop: int = 0,
        generator: torch.Generator | None = None,
        centre_input: bool = False,
    ):
        """Return features [B, T, C, H, W] (NaN where missing), valid [B, H, W], target [B, H, W],
        the pixel counts of every evaluation region and, for the pixel model, the pixel-level
        inputs, target, valid mask and regions (else None).

        The target is the fraction of a node's pixels that burn the next day. `regions` maps
        "full" (the benchmark crop) and "centre" (its central 128 x 128 pixels) to the number of
        burning pixels and of all pixels of every node in that region, both [B, H, W].
        With `crop`, every sample is a random crop x crop window placed around its fire. With
        `centre_input`, the model only sees the nodes covering the centre 128 x 128 pixels, as in
        the test protocol of Lahrichi et al. (2026).
        """

        fire_channel, burned_channel = CHANNEL_INDEX["fire_frac"], CHANNEL_INDEX["burned_ever"]
        windows = []
        for index, day in samples:
            height, width = self.fires[index].shape[2:]
            top = left = 0
            if centre_input:
                top, left, height, width = self.centre_windows[index]
            elif crop:
                fire = self.fires[index]
                burning = torch.nonzero((fire[day, burned_channel] > 0) | (fire[day + 1, fire_channel] > 0))
                jitter = torch.randint(-crop // 4, crop // 4 + 1, (2,), generator=generator)
                if len(burning):
                    centre = burning[torch.randint(len(burning), (1,), generator=generator).item()] + jitter
                else:
                    centre = (torch.rand(2, generator=generator) * torch.tensor([height, width])).long()
                top = int(centre[0].clamp(crop // 2, max(height - crop // 2, crop // 2))) - crop // 2
                left = int(centre[1].clamp(crop // 2, max(width - crop // 2, crop // 2))) - crop // 2
                height, width = min(crop, height), min(crop, width)
            windows.append((top, left, height, width))
        canvas = [(max(window[axis] for window in windows) + 3) // 4 * 4 for axis in (2, 3)]

        features = torch.zeros(len(samples), self.window, len(CHANNELS), *canvas, dtype=torch.float16)
        valid = torch.zeros(len(samples), *canvas)
        counts = {name: torch.zeros(len(samples), *canvas) for name in ("fire", "centre_fire", "centre_pixels")}
        for row, ((index, day), (top, left, height, width)) in enumerate(zip(samples, windows)):
            fire = self.fires[index][:, :, top : top + height, left : left + width]
            features[row, :, :, :height, :width] = fire[day - self.window + 1 : day + 1]
            valid[row, :height, :width] = 1
            targets = {name: values[..., top : top + height, left : left + width] for name, values in self.targets[index].items()}
            counts["fire"][row, :height, :width] = targets["fire_pixels"][day + 1]
            counts["centre_fire"][row, :height, :width] = targets["centre_fire"][day + 1]
            counts["centre_pixels"][row, :height, :width] = targets["centre_pixels"]
        counts = {name: values.to(device, non_blocking=True) for name, values in counts.items()}
        valid = valid.to(device, non_blocking=True)
        regions = {
            "full": (counts["fire"], valid * self.node_pixels),
            "centre": (counts["centre_fire"], counts["centre_pixels"]),
        }
        pixels = self._pixel_batch(samples, windows, canvas, device) if self.pixels is not None else None
        return features.to(device, non_blocking=True).float(), valid, counts["fire"] / self.node_pixels, regions, pixels

    def _pixel_batch(self, samples, windows, canvas, device) -> dict:
        p = self.patch
        shape = (len(samples), canvas[0] * p, canvas[1] * p)
        inputs = torch.zeros(len(samples), self.window, len(PIXEL_CHANNELS), *shape[1:], dtype=torch.float16)
        target, valid, centre = torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)
        for row, ((index, day), (top, left, height, width)) in enumerate(zip(samples, windows)):
            rows, cols = slice(top * p, (top + height) * p), slice(left * p, (left + width) * p)
            fire = self.pixels[index]
            inputs[row, :, :, : height * p, : width * p] = fire["inputs"][day - self.window + 1 : day + 1, :, rows, cols]
            target[row, : height * p, : width * p] = fire["mask"][day + 1, rows, cols]
            valid[row, : height * p, : width * p] = 1
            centre[row, : height * p, : width * p] = fire["centre"][rows, cols]
        target, valid, centre = (values.to(device, non_blocking=True) for values in (target, valid, centre))
        return {"inputs": inputs.to(device, non_blocking=True).float(), "target": target, "valid": valid, "centre": centre}


def transform(features, valid, target, flip_rows: bool, flip_cols: bool, transpose: bool):
    """Flip/transpose a batch, rotating its vector channels with it."""

    if flip_rows:
        features, valid, target = features.flip(-2), valid.flip(-2), target.flip(-2)
        features[:, :, _ROW_CHANNELS] = -features[:, :, _ROW_CHANNELS]
    if flip_cols:
        features, valid, target = features.flip(-1), valid.flip(-1), target.flip(-1)
        features[:, :, _COL_CHANNELS] = -features[:, :, _COL_CHANNELS]
    if transpose:
        features, valid, target = features.transpose(-1, -2), valid.transpose(-1, -2), target.transpose(-1, -2)
        features = features.clone()
        features[:, :, _ROW_CHANNELS + _COL_CHANNELS] = features[:, :, _COL_CHANNELS + _ROW_CHANNELS]
    return features, valid, target


def graph_fields(features: torch.Tensor, valid: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> dict[str, torch.Tensor]:
    """The physical fields the graph models build their shift operators from, each [B, T, H, W]."""

    def physical(name: str) -> torch.Tensor:
        index = CHANNEL_INDEX[name]
        return features[:, :, index] * std[index] + mean[index]

    return {
        "wind_row": features[:, :, CHANNEL_INDEX["wind_row"]],
        "wind_col": features[:, :, CHANNEL_INDEX["wind_col"]],
        "wind_speed": physical("wind_speed"),
        "elevation": physical("elevation"),
        "valid": valid[:, None].expand(-1, features.shape[1], -1, -1),
    }


def transform_pixels(pixels: dict | None, flip_rows: bool, flip_cols: bool, transpose: bool) -> dict | None:
    """Flip/transpose the pixel-level tensors of a batch like `transform` does the node grid."""

    if pixels is None:
        return None
    moved = {}
    for name, values in pixels.items():
        if flip_rows:
            values = values.flip(-2)
        if flip_cols:
            values = values.flip(-1)
        if transpose:
            values = values.transpose(-1, -2)
        moved[name] = values
    return moved


def pixel_regions(pixels: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Evaluation regions at pixel level: every pixel counts once, burning or not."""

    return {"full": (pixels["target"], pixels["valid"]), "centre": (pixels["target"] * pixels["centre"], pixels["centre"])}


def forward(model: nn.Module, features: torch.Tensor, valid: torch.Tensor, data: "FireData", pixels: dict | None = None) -> torch.Tensor:
    """Logits of a batch, [B, H, W] per node or per pixel; graph models also get the physical fields."""

    inputs = torch.nan_to_num(features)
    if isinstance(model, (GraphTimeCNN, PixelGraphTimeCNN)):
        mean, std = (torch.from_numpy(values).to(features.device) for values in (data.mean, data.std))
        fields = graph_fields(features, valid, mean, std)
        return model(inputs, fields, pixels["inputs"]) if isinstance(model, PixelGraphTimeCNN) else model(inputs, fields)
    return model(inputs)


def build_model(config: TrainConfig) -> nn.Module:
    channels = len(CHANNELS)
    gtcnn_kwargs = dict(
        graph=config.graph, hidden=config.hidden, layers=config.layers, taps=config.taps, heads=config.heads,
        hops=config.hops, dropout=config.dropout, patch_metres=375.0 * config.patch,
    )
    if config.model == "gtcnn":
        return GraphTimeCNN(channels, **gtcnn_kwargs)
    if config.model == "pixel_gtcnn":
        return PixelGraphTimeCNN(channels, len(PIXEL_CHANNELS), config.patch, config.pixel_hidden, **gtcnn_kwargs)
    if config.model == "cnn":
        return ConvNet(channels, config.window, config.hidden, config.dropout)
    if config.model == "mlp":
        return NodeMLP(channels, config.window, config.hidden, config.layers, config.dropout)
    if config.model == "logreg":
        return NodeMLP(channels, config.window, layers=0)
    return Persistence()


def init_output_bias(model: nn.Module, rate: float) -> None:
    """Start the final logit at the log-odds of the training fire rate instead of 0 (p = 0.5)."""

    heads = [module for module in model.modules() if isinstance(module, nn.Conv2d) and module.out_channels == 1]
    if heads and heads[-1].bias is not None:
        nn.init.constant_(heads[-1].bias, float(np.log(rate / (1 - rate))))


@torch.no_grad()
def predict(model: nn.Module, data: FireData, split: str, config: TrainConfig, device: torch.device, stride: int = 1, centre_input: bool = False):
    """Logits of every real node (or pixel) in a split, or every `stride`-th sample, and for every
    evaluation region the burning and total pixel counts of those nodes; all flat tensors."""

    model.eval()
    scores, counts = [], []
    samples = data.samples[split][::stride]
    for start in range(0, len(samples), config.eval_batch_size):
        features, valid, _, regions, pixels = data.batch(samples[start : start + config.eval_batch_size], device, centre_input=centre_input)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = forward(model, features, valid, data, pixels)
        if pixels is not None:
            # Pixel labels are 0/1: keep them as uint8 so a whole test year fits in memory.
            valid = pixels["valid"]
            regions = {name: (fire.to(torch.uint8), total.to(torch.uint8)) for name, (fire, total) in pixel_regions(pixels).items()}
        keep = valid > 0
        scores.append(logits.float()[keep].cpu())
        counts.append({name: (fire[keep].cpu(), total[keep].cpu()) for name, (fire, total) in regions.items()})
    regions = {
        name: (torch.cat([batch[name][0] for batch in counts]), torch.cat([batch[name][1] for batch in counts]))
        for name in counts[0]
    }
    return torch.cat(scores), regions


def run(config: TrainConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = FireData(config)
    model = build_model(config).to(device)
    init_output_bias(model, data.fire_rate)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    wandb.init(
        project=config.wandb_project,
        name=config.run_name,
        group=config.experiment_name,
        config={**asdict(config), "parameters": parameters, "fold_years": FOLDS[config.fold].name},
    )

    best = {"val_ap": -1.0, "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    started = time.time()
    if parameters:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        steps_per_epoch = len(data.samples["train"]) // config.batch_size
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr, total_steps=config.epochs * steps_per_epoch, pct_start=0.1
        )
        pos_weight = torch.tensor(config.pos_weight, device=device)
        generator = torch.Generator().manual_seed(config.seed)
        for epoch in range(1, config.epochs + 1):
            model.train()
            order = torch.randperm(len(data.samples["train"]), generator=generator).tolist()
            losses = []
            for step in range(steps_per_epoch):
                chosen = [data.samples["train"][index] for index in order[step * config.batch_size : (step + 1) * config.batch_size]]
                flags = (torch.rand(3, generator=generator) < 0.5).tolist()
                features, valid, target, _, pixels = data.batch(chosen, device, config.crop, generator)
                features, valid, target = transform(features, valid, target, *flags)
                pixels = transform_pixels(pixels, *flags)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = forward(model, features, valid, data, pixels)
                if pixels is not None:
                    target, valid = pixels["target"], pixels["valid"]
                # Pixel-level BCE: every node stands for its pixels, a fraction `target` of which burn.
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), target, weight=valid, pos_weight=pos_weight, reduction="none"
                )
                loss = loss.sum() / valid.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                losses.append(loss.item())
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            val_scores, val_regions = predict(model, data, "val", config, device, config.val_stride)
            val_ap = average_precision(val_scores, *val_regions["full"])
            if val_ap > best["val_ap"]:
                best = {"val_ap": val_ap, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            wandb.log({"epoch": epoch, "train/loss": float(np.mean(losses)), "val/ap": val_ap, "val/best_ap": best["val_ap"]})
            print(f"epoch {epoch:3d} loss {np.mean(losses):.5f} val AP {val_ap:.4f} (best {best['val_ap']:.4f} @ {best['epoch']}) "
                  f"{time.time() - started:.0f}s", flush=True)
        model.load_state_dict(best["state"])
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best["state"], checkpoint_dir / f"{config.run_name}.pt")

    val_scores, val_regions = predict(model, data, "val", config, device)
    threshold = best_f1_threshold(val_scores, *val_regions["full"])
    val_ap = average_precision(val_scores, *val_regions["full"])
    del val_scores, val_regions  # free the validation year before scoring the test year
    test_scores, test_regions = predict(model, data, "test", config, device)
    full_ap = average_precision(test_scores, *test_regions["full"])
    centre_from_full = average_precision(test_scores, *test_regions["centre"])
    del test_scores, test_regions
    centre_scores, centre_regions = predict(model, data, "test", config, device, centre_input=True)
    result = {
        "run_name": config.run_name,
        **asdict(config),
        "parameters": parameters,
        "best_epoch": best["epoch"],
        "val_ap": val_ap,
        # Gerard et al. (2023): whole image in, all pixels of the benchmark crop scored.
        "test_ap": full_ap,
        # Lahrichi et al. (2026): only the centre 128 x 128 window in, and scored.
        "test_ap_centre128": average_precision(centre_scores, *centre_regions["centre"]),
        # Diagnostic: the centre window scored from the whole-image pass (not a published protocol).
        "test_ap_centre128_full_input": centre_from_full,
        # F1 / IoU at probability 0.5 on the centre window, as their code logs them.
        **{f"test_{name}": value for name, value in threshold_metrics(centre_scores, *centre_regions["centre"], 0.0).items()},
        # Same, at the threshold that maximises F1 on the validation year (not a published number).
        **{f"test_{name}_tuned": value for name, value in threshold_metrics(centre_scores, *centre_regions["centre"], threshold).items()},
        "minutes": (time.time() - started) / 60,
    }
    if config.model in ("gtcnn", "pixel_gtcnn"):
        core = model.gtcnn if config.model == "pixel_gtcnn" else model
        result["wind_gain"], result["slope_gain"] = (float(value) for value in core.gains)
        result["product_mix"] = [torch.softmax(layer.mix, dim=0).tolist() for layer in core.layers]
    wandb.summary.update({key: value for key, value in result.items() if key.startswith(("val_", "test_", "best_"))})
    wandb.finish()

    results_file = Path(config.results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("a") as output:
        output.write(json.dumps(result) + "\n")
    print(f"{config.run_name}: val AP {result['val_ap']:.4f} test AP {result['test_ap']:.4f} "
          f"(centre128 {result['test_ap_centre128']:.4f}) F1 {result['test_f1']:.4f} IoU {result['test_iou']:.4f} "
          f"(tuned F1 {result['test_f1_tuned']:.4f} IoU {result['test_iou_tuned']:.4f})")
    result["state"] = best["state"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a next-day wildfire spread model")
    for field in fields(TrainConfig):
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=field.type, default=field.default)
    run(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
