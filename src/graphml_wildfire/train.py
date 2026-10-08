"""Train and evaluate next-day fire prediction on the benchmark node grid.

Evaluation follows the WildfireSpreadTS benchmark (see `benchmark.py`). A node's score is
shared by all pixels of its patch, so pixel-level AP is computed exactly from per-node counts
of burning and non-burning pixels.

Example:
    uv run python -m graphml_wildfire.train --model logreg --window 5 --fold 0
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
from .models import ConvNet, NodeMLP, Persistence


MODELS = ("cnn", "mlp", "logreg", "persistence")

_ROW_CHANNELS = [CHANNEL_INDEX[name] for name in ("wind_row", "aspect_row", "fc_wind_row")]
_COL_CHANNELS = [CHANNEL_INDEX[name] for name in ("wind_col", "aspect_col", "fc_wind_col")]


@dataclass
class TrainConfig:
    experiment_name: str = ""
    model: str = "logreg"
    window: int = 5
    hidden: int = 64
    layers: int = 4
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 32
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
        if not self.cache_dir:
            self.cache_dir = f"data/processed/wildfirespreadts_benchmark_b{self.patch}"
        if not self.experiment_name:
            self.experiment_name = f"{self.model}_w{self.window}_p{self.patch}"
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
        split_of = {train_a: "train", train_b: "train", val_year: "val", test_year: "test"}
        for year in YEARS:
            for arrays in load_patch_cache(config.cache_dir, [year]).values():
                fire = (arrays["features"] - self.mean[None, :, None, None]) / self.std[None, :, None, None]
                self.fires.append(torch.from_numpy(fire.astype(np.float16)))
                self.targets.append({
                    name: torch.from_numpy(arrays[name].astype(np.int16))
                    for name in ("fire_pixels", "centre_pixels", "centre_fire")
                })
                index = len(self.fires) - 1
                self.samples[split_of[year]] += [(index, day) for day in range(FIRST_INPUT_DAY, len(fire) - 1)]
        burning = sum(float(self.targets[index]["fire_pixels"][day + 1].sum()) for index, day in self.samples["train"])
        pixels = sum(self.fires[index][0, 0].numel() * self.node_pixels for index, _ in self.samples["train"])
        self.fire_rate = burning / pixels

    def batch(
        self,
        samples: list[tuple[int, int]],
        device: torch.device,
        crop: int = 0,
        generator: torch.Generator | None = None,
    ):
        """Return features [B, T, C, H, W] (NaN where missing), valid [B, H, W], target [B, H, W]
        and the pixel counts of every evaluation region.

        The target is the fraction of a node's pixels that burn the next day. `regions` maps
        "full" (the benchmark crop) and "centre" (its central 128 x 128 pixels) to the number of
        burning pixels and of all pixels of every node in that region, both [B, H, W].
        With `crop`, every sample is a random crop x crop window placed around its fire.
        """

        fire_channel, burned_channel = CHANNEL_INDEX["fire_frac"], CHANNEL_INDEX["burned_ever"]
        windows = []
        for index, day in samples:
            height, width = self.fires[index].shape[2:]
            top = left = 0
            if crop:
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
        return features.to(device, non_blocking=True).float(), valid, counts["fire"] / self.node_pixels, regions


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


def build_model(config: TrainConfig) -> nn.Module:
    if config.model == "cnn":
        return ConvNet(len(CHANNELS), config.window, config.hidden, config.dropout)
    if config.model == "mlp":
        return NodeMLP(len(CHANNELS), config.window, config.hidden, config.layers, config.dropout)
    if config.model == "logreg":
        return NodeMLP(len(CHANNELS), config.window, layers=0)
    return Persistence()


def init_output_bias(model: nn.Module, rate: float) -> None:
    """Start the final logit at the log-odds of the training fire rate instead of 0 (p = 0.5)."""

    heads = [module for module in model.modules() if isinstance(module, nn.Conv2d) and module.out_channels == 1]
    if heads and heads[-1].bias is not None:
        nn.init.constant_(heads[-1].bias, float(np.log(rate / (1 - rate))))


@torch.no_grad()
def predict(model: nn.Module, data: FireData, split: str, config: TrainConfig, device: torch.device, stride: int = 1):
    """Logits of every real node in a split (or every `stride`-th sample) and, for every
    evaluation region, the burning and total pixel counts of those nodes; all flat tensors."""

    model.eval()
    scores, counts = [], []
    samples = data.samples[split][::stride]
    for start in range(0, len(samples), config.batch_size):
        features, valid, _, regions = data.batch(samples[start : start + config.batch_size], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(torch.nan_to_num(features))
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
                features, valid, target = transform(*data.batch(chosen, device, config.crop, generator)[:3], *flags)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(torch.nan_to_num(features))
                # Pixel-level BCE: every node stands for its pixels, a fraction `target` of which burn.
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), target, weight=valid, pos_weight=pos_weight, reduction="sum"
                ) / valid.sum()
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

    val_scores, val_regions = predict(model, data, "val", config, device)
    test_scores, test_regions = predict(model, data, "test", config, device)
    threshold = best_f1_threshold(val_scores, *val_regions["full"])
    result = {
        "run_name": config.run_name,
        **asdict(config),
        "parameters": parameters,
        "best_epoch": best["epoch"],
        "val_ap": average_precision(val_scores, *val_regions["full"]),
        # Gerard et al. (2023): all pixels of the benchmark crop.
        "test_ap": average_precision(test_scores, *test_regions["full"]),
        # Lahrichi et al. (2026): the central 128 x 128 pixels only.
        "test_ap_centre128": average_precision(test_scores, *test_regions["centre"]),
        # Benchmark convention: probability >= 0.5.
        **{f"test_{name}": value for name, value in threshold_metrics(test_scores, *test_regions["full"], 0.0).items()},
        # Threshold that maximises F1 on the validation year.
        **{f"test_{name}_tuned": value for name, value in threshold_metrics(test_scores, *test_regions["full"], threshold).items()},
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
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=field.type, default=field.default)
    run(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
