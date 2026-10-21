"""Train and evaluate next-day fire prediction on the WildfireSpreadTS benchmark.

The default model is the pixel-graph-time CNN (`--model pixel_gtcnn`, 0.75 km nodes, 375 m
pixels); the node-level baselines of the leaderboard (`cnn`, `mlp`, `logreg`, `persistence`)
train with the same loop. Data comes from the training cache of `data/mmap.py`.

Evaluation follows the WildfireSpreadTS benchmark (see `benchmark.py`). A node model's score
is shared by all pixels of its patch, so pixel-level AP is computed exactly from per-node counts
of burning and non-burning pixels; the pixel model is scored per pixel directly.

Example:
    uv run python -m graphml_wildfire.train --fold 0
    uv run python -m graphml_wildfire.train --model mlp --patch 4 --fold 0
"""

import argparse
import copy
import hashlib
import json
import time
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn

from .benchmark import FIRST_INPUT_DAY, PROTOCOLS, average_precision, best_f1_threshold, threshold_metrics
from .data.mmap import TARGETS
from .data.patches import CHANNEL_INDEX, CHANNELS, CONTINUOUS, MASKS
from .data.pixels import PIXEL_CHANNELS
from .models import ConvNet, NodeMLP, Persistence, PixelGraphTimeCNN


MODELS = ("pixel_gtcnn", "pixel_gtcnn_uniform",  "cnn", "mlp", "logreg", "persistence")
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

_ROW_CHANNELS = [CHANNEL_INDEX[name] for name in ("wind_row", "aspect_row", "fc_wind_row")]
_COL_CHANNELS = [CHANNEL_INDEX[name] for name in ("wind_col", "aspect_col", "fc_wind_col")]
_CONTINUOUS = [CHANNEL_INDEX[name] for name in CONTINUOUS]
# The pixel model reads 45 of the 49 node channels: the missing-data masks and the aspect vector
# were dropped (measured neutral on the screening folds). The baselines read all 49.
_GRAPH_INPUTS = [index for index, name in enumerate(CHANNELS) if name not in MASKS + ["aspect_row", "aspect_col"]]


@dataclass
class TrainConfig:
    experiment_name: str = ""
    model: str = "pixel_gtcnn"
    window: int = 5  # input days
    hidden: int = 64  # node state width
    layers: int = 4  # graph-time layers (pixel_gtcnn) or MLP layers
    taps: int = 3  # graph filter order (pixel_gtcnn)
    heads: int = 4  # shift operators per layer (pixel_gtcnn)
    pixel_hidden: int = 16  # pixel encoder width (pixel_gtcnn)
    decoder_hidden: int = 32  # pixel decoder width (pixel_gtcnn)
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 0.0
    epochs: int = 30
    batch_size: int = 32
    eval_batch_size: int = 8  # whole fires: smaller batches than the training crops
    crop: int = 32  # training crop in nodes
    val_stride: int = 3  # validate on every third sample during training
    compile: int = 1  # torch.compile the model (static shapes for training, dynamic for evaluation)
    protocol: str = "wsts"  # "wsts": the 12 folds of 2018-2021; "wstsplus": the 4 folds of 2016-2023
    fold: int = 0
    seed: int = 0
    patch: int = 2  # pixels per node side: 2 (0.75 km nodes) for the pixel model, 4 (1.5 km) for the baselines
    data_dir: str = ""  # training cache from data/mmap.py; default data/cache/b{patch}
    checkpoint_dir: str = "data/checkpoints"
    results_file: str = "results/benchmark_runs.jsonl"
    wandb_project: str = "graphml-wildfire"

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"Unknown model: {self.model}")
        if not 1 <= self.window <= FIRST_INPUT_DAY + 1:
            raise ValueError(f"window must be between 1 and {FIRST_INPUT_DAY + 1}")
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"Unknown protocol: {self.protocol}")
        if not 0 <= self.fold < len(PROTOCOLS[self.protocol]):
            raise ValueError(f"fold must be between 0 and {len(PROTOCOLS[self.protocol]) - 1} for {self.protocol}")
        if not self.data_dir:
            self.data_dir = f"data/cache/b{self.patch}"
        if not self.experiment_name:
            suffix = "" if self.protocol == "wsts" else f"_{self.protocol}"
            self.experiment_name = f"{self.model}_w{self.window}_p{self.patch}{suffix}"
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")

    @property
    def run_name(self) -> str:
        digest = hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.experiment_name}_f{self.fold}_s{self.seed}_{digest}"


class FireData:
    """All fires of the years of a fold, memory-mapped from the training cache and standardised
    per batch with training-year statistics."""

    def __init__(self, config: TrainConfig):
        fold = PROTOCOLS[config.protocol][config.fold]
        train_years = set(fold.train_years)
        self.window, self.patch, self.node_pixels = config.window, config.patch, config.patch**2
        self.input_channels = _GRAPH_INPUTS if config.model in ["pixel_gtcnn", "pixel_gtcnn_uniform"] else list(range(len(CHANNELS)))
        self.pixels = [] if config.model in ["pixel_gtcnn", "pixel_gtcnn_uniform"] else None
        self.fires, self.targets, self.samples = [], [], {"train": [], "val": [], "test": []}
        self.centre_windows = []  # node-aligned window around the centre 128 x 128 pixels of every fire
        split_of = {**{year: "train" for year in train_years}, **{year: "val" for year in fold.validation_years}, **{year: "test" for year in fold.test_years}}
        total = torch.zeros(len(_CONTINUOUS), dtype=torch.float64)
        squares, count = torch.zeros_like(total), torch.zeros_like(total)
        for year in fold.years:
            for fire_dir in sorted(path for path in (Path(config.data_dir) / str(year)).iterdir() if path.is_dir()):
                fire = torch.from_numpy(np.load(fire_dir / "features.npy", mmap_mode="r"))
                if fire.shape[1] != len(CHANNELS):
                    raise ValueError(f"{fire_dir} has {fire.shape[1]} channels, expected {len(CHANNELS)}: rebuild the cache")
                self.fires.append(fire)
                self.targets.append({name: torch.from_numpy(np.load(fire_dir / f"{name}.npy", mmap_mode="r")) for name in TARGETS})
                if self.pixels is not None:
                    pixels = {name: torch.from_numpy(np.load(fire_dir / f"pix_{name}.npy", mmap_mode="r")) for name in PIXEL_CHANNELS}
                    pixels["centre"] = tuple(int(value) for value in np.load(fire_dir / "pix_centre.npy"))
                    self.pixels.append(pixels)
                if year in train_years:  # running moments of every third day, ignoring missing values
                    sample = fire[::3, _CONTINUOUS].double()
                    finite = sample.isfinite()
                    sample = sample.nan_to_num()
                    total += sample.sum(dim=(0, 2, 3))
                    squares += sample.pow(2).sum(dim=(0, 2, 3))
                    count += finite.sum(dim=(0, 2, 3))
                index = len(self.fires) - 1
                self.samples[split_of[year]] += [(index, day) for day in range(FIRST_INPUT_DAY, len(fire) - 1)]
                rows, cols = (torch.nonzero(self.targets[index]["centre_pixels"].sum(dim=axis) > 0).flatten() for axis in (1, 0))
                self.centre_windows.append((int(rows[0]), int(cols[0]), int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1)))
        self.mean = np.zeros(len(CHANNELS), dtype=np.float32)
        self.std = np.ones(len(CHANNELS), dtype=np.float32)
        self.mean[_CONTINUOUS] = (total / count).float().numpy()
        self.std[_CONTINUOUS] = (squares / count - (total / count) ** 2).clamp_min(0).sqrt().float().numpy() + 1e-6
        self.mean_t = torch.from_numpy(self.mean)[None, None, :, None, None]
        self.std_t = torch.from_numpy(self.std)[None, None, :, None, None]

    def batch(self, samples: list[tuple[int, int]], device: torch.device, crop: int = 0, generator: torch.Generator | None = None, centre_input: bool = False):
        """Return features [B, T, C, H, W] (NaN where missing), valid [B, H, W], target [B, H, W],
        the pixel counts of every evaluation region and, for the pixel model, the pixel-level
        inputs, target, valid mask and centre window (else None).

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
        # Training crops share one canvas (static shapes for torch.compile); evaluation pads to a multiple of 4.
        canvas = [crop, crop] if crop else [(max(window[axis] for window in windows) + 3) // 4 * 4 for axis in (2, 3)]

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
        features = (features.to(device, non_blocking=True).float() - self.mean_t.to(device)) / self.std_t.to(device)
        return features, valid, counts["fire"] / self.node_pixels, regions, pixels

    def _pixel_batch(self, samples, windows, canvas, device) -> dict[str, torch.Tensor]:
        p = self.patch
        shape = (len(samples), canvas[0] * p, canvas[1] * p)
        inputs = torch.zeros(len(samples), self.window, len(PIXEL_CHANNELS), *shape[1:], dtype=torch.float16)
        target, valid, centre = torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)
        for row, ((index, day), (top, left, height, width)) in enumerate(zip(samples, windows)):
            rows, cols = slice(top * p, (top + height) * p), slice(left * p, (left + width) * p)
            fire = self.pixels[index]
            days = slice(day - self.window + 1, day + 1)
            for channel, name in enumerate(PIXEL_CHANNELS):
                inputs[row, :, channel, : height * p, : width * p] = fire[name][days, rows, cols]
            target[row, : height * p, : width * p] = fire["mask"][day + 1, rows, cols]
            valid[row, : height * p, : width * p] = 1
            centre_top, centre_left = fire["centre"]
            r0, r1 = max(centre_top - rows.start, 0), min(centre_top + 128 - rows.start, height * p)
            c0, c1 = max(centre_left - cols.start, 0), min(centre_left + 128 - cols.start, width * p)
            if r1 > r0 and c1 > c0:
                centre[row, r0:r1, c0:c1] = 1
        target, valid, centre = (values.to(device, non_blocking=True) for values in (target, valid, centre))
        inputs = inputs.to(device, non_blocking=True).float()
        for channel, name in enumerate(PIXEL_CHANNELS):  # the bands are standardised like the node feature of the same name
            if name in CHANNEL_INDEX:
                index = CHANNEL_INDEX[name]
                inputs[:, :, channel] = torch.nan_to_num((inputs[:, :, channel] - float(self.mean[index])) / float(self.std[index]))
        return {"inputs": inputs, "target": target, "valid": valid, "centre": centre}


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


def forward(model: nn.Module, features: torch.Tensor, valid: torch.Tensor, data: FireData, pixels: dict | None = None) -> torch.Tensor:
    """Logits of a batch, [B, H, W] per node or per pixel."""

    inputs = torch.nan_to_num(features)[:, :, data.input_channels]
    call = model.compiled[model.training] if hasattr(model, "compiled") else model
    if isinstance(model, PixelGraphTimeCNN):
        return call(inputs, valid, pixels["inputs"])
    return call(inputs)


def build_model(config: TrainConfig, data: FireData) -> nn.Module:
    channels = len(data.input_channels)
    if config.model == "pixel_gtcnn":
        return PixelGraphTimeCNN(channels, len(PIXEL_CHANNELS), config.patch, config.pixel_hidden, config.decoder_hidden,
                                 hidden=config.hidden, layers=config.layers, taps=config.taps, heads=config.heads, dropout=config.dropout)
    if config.model == "pixel_gtcnn_uniform":
        return PixelGraphTimeCNN(channels, len(PIXEL_CHANNELS), config.patch, config.pixel_hidden, config.decoder_hidden,
                                 hidden=config.hidden, layers=config.layers, taps=config.taps, heads=config.heads, dropout=config.dropout, uniform=True)
    if config.model == "cnn":
        return ConvNet(channels, config.window, config.hidden, config.dropout)
    if config.model == "mlp":
        return NodeMLP(channels, config.window, config.hidden, config.layers, config.dropout)
    if config.model == "logreg":
        return NodeMLP(channels, config.window, layers=0)
    return Persistence()


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


def test_metrics(model: nn.Module, data: FireData, config: TrainConfig, device: torch.device) -> dict[str, float]:
    """The benchmark metrics of a trained model on the validation and test years of its fold."""

    val_scores, val_regions = predict(model, data, "val", config, device)
    threshold = best_f1_threshold(val_scores, *val_regions["full"])
    val_ap = average_precision(val_scores, *val_regions["full"])
    del val_scores, val_regions  # free the validation year before scoring the test year
    test_scores, test_regions = predict(model, data, "test", config, device)
    full_ap = average_precision(test_scores, *test_regions["full"])
    centre_from_full = average_precision(test_scores, *test_regions["centre"])
    del test_scores, test_regions
    centre_scores, centre_regions = predict(model, data, "test", config, device, centre_input=True)
    return {
        "val_ap": val_ap,
        # Gerard et al. (2023): whole image in, all pixels of the benchmark crop scored.
        "test_ap": full_ap,
        # Lahrichi et al. (2026): only the centre 128 x 128 window in, and scored. The leaderboard metric.
        "test_ap_centre128": average_precision(centre_scores, *centre_regions["centre"]),
        # Diagnostic: the centre window scored from the whole-image pass (not a published protocol).
        "test_ap_centre128_full_input": centre_from_full,
        # F1 / IoU at probability 0.5 on the centre window, as their code logs them.
        **{f"test_{name}": value for name, value in threshold_metrics(centre_scores, *centre_regions["centre"], 0.0).items()},
        # Same, at the threshold that maximises F1 on the validation year (not a published number).
        **{f"test_{name}_tuned": value for name, value in threshold_metrics(centre_scores, *centre_regions["centre"], threshold).items()},
    }


def run(config: TrainConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    data = FireData(config)
    model = build_model(config, data).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    wandb.init(
        project=config.wandb_project,
        name=config.run_name,
        group=config.experiment_name,
        config={**asdict(config), "parameters": parameters, "fold_years": PROTOCOLS[config.protocol][config.fold].name},
    )

    best = {"val_ap": -1.0, "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    if config.compile and device.type == "cuda" and parameters:
        # Training crops have one shape (static graph); evaluation sees every fire's own shape (dynamic).
        model.compiled = {True: torch.compile(model.forward), False: torch.compile(model.forward, dynamic=True)}
    started = time.time()
    if parameters:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, fused=device.type == "cuda")
        steps_per_epoch = len(data.samples["train"]) // config.batch_size
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr, total_steps=config.epochs * steps_per_epoch, pct_start=0.1
        )
        generator = torch.Generator().manual_seed(config.seed)
        for epoch in range(1, config.epochs + 1):
            model.train()
            order = torch.randperm(len(data.samples["train"]), generator=generator).tolist()
            losses = torch.zeros((), device=device)
            for step in range(steps_per_epoch):
                chosen = [data.samples["train"][index] for index in order[step * config.batch_size : (step + 1) * config.batch_size]]
                flags = (torch.rand(3, generator=generator) < 0.5).tolist()  # flip rows, flip columns, transpose
                features, valid, target, _, pixels = data.batch(chosen, device, config.crop, generator)
                features, valid, target = transform(features, valid, target, *flags)
                pixels = transform_pixels(pixels, *flags)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = forward(model, features, valid, data, pixels)
                if pixels is not None:
                    target, valid = pixels["target"], pixels["valid"]
                # Pixel-level BCE: a node stands for its pixels, a fraction `target` of which burn; the pixel model is scored per pixel.
                loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), target, weight=valid, reduction="sum") / valid.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                losses += loss.detach()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            val_scores, val_regions = predict(model, data, "val", config, device, config.val_stride)
            val_ap = average_precision(val_scores, *val_regions["full"])
            if val_ap > best["val_ap"]:
                best = {"val_ap": val_ap, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            wandb.log({"epoch": epoch, "train/loss": float((losses / steps_per_epoch).item()), "val/ap": val_ap, "val/best_ap": best["val_ap"]})
            print(f"epoch {epoch:3d} loss {(losses / steps_per_epoch).item():.5f} val AP {val_ap:.4f} (best {best['val_ap']:.4f} @ {best['epoch']}) "
                  f"{time.time() - started:.0f}s", flush=True)
        model.load_state_dict(best["state"])
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best["state"], checkpoint_dir / f"{config.run_name}.pt")

    result = {
        "run_name": config.run_name,
        **asdict(config),
        "parameters": parameters,
        "best_epoch": best["epoch"],
        **test_metrics(model, data, config, device),
        "minutes": (time.time() - started) / 60,
    }
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
        arg = f"--{field.name.replace('_', '-')}"

        if field.type is bool:
            parser.add_argument(
                arg,
                action=argparse.BooleanOptionalAction,
                default=field.default,
            )
        else:
            parser.add_argument(
                arg,
                type=field.type,
                default=field.default,
            ) 
    run(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
