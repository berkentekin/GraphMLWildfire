"""One real-cache training step and one validation forward pass per graph mode.

This is a wiring/memory check, not a benchmark or an accuracy comparison.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .models.gtcnn import GRAPH_MODES
from .train import FireData, TrainConfig, autocast_context, build_model, forward, predict, resolve_precision


def smoke(config: TrainConfig) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = resolve_precision(config.precision, device)
    data = FireData(config)
    if not data.samples["train"] or not data.samples["val"]:
        raise ValueError("Smoke test requires nonempty training and validation splits")
    data.samples["val"] = data.samples["val"][:1]
    results = []
    for mode in GRAPH_MODES:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        variant = replace(config, graph_mode=mode, compile=0, eval_batch_size=1)
        model = build_model(variant, data).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        chosen = data.samples["train"][: config.batch_size]
        features, valid, _, _, pixels = data.batch(
            chosen, device, config.crop, torch.Generator().manual_seed(config.seed)
        )
        with autocast_context(precision):
            logits = forward(model, features, valid, data, pixels)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.float(), pixels["target"], weight=pixels["valid"], reduction="sum"
        ) / pixels["valid"].sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{mode}: non-finite smoke loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError(f"{mode}: non-finite smoke gradients; try --precision fp32")
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scores, _ = predict(model, data, "val", variant, device)
        if not scores.numel() or not torch.isfinite(scores).all():
            raise FloatingPointError(f"{mode}: empty or non-finite validation predictions")
        result = {
            "graph_mode": mode,
            "device": str(device),
            "precision": precision,
            "loss": loss.item(),
            "logits_shape": list(logits.shape),
            "validation_pixels": scores.numel(),
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, optimizer, scaler, logits, loss, scores
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop", type=int, default=32)
    parser.add_argument("--output", default="results/smoke.json")
    args = parser.parse_args()
    results = smoke(TrainConfig(
        data_dir=args.data_dir, precision=args.precision, fold=args.fold,
        batch_size=args.batch_size, crop=args.crop, compile=0,
    ))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print("All three modes passed. Smoke losses are NOT benchmark results.")


if __name__ == "__main__":
    main()
