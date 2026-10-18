"""Re-score saved checkpoints under the benchmark protocol and append a run record per checkpoint.

    uv run python -m graphml_wildfire.evaluate data/checkpoints/pixel_gtcnn_w5_p2_f*.pt

Every checkpoint is matched to the run that produced it (by run name, in `--runs`) to recover
its configuration; the new record keeps that configuration and replaces the test metrics.
"""

import argparse
import json
from pathlib import Path

import torch

from .train import FireData, TrainConfig, build_model, resolve_precision, test_metrics


def evaluate(checkpoint: Path, run: dict) -> dict:
    config = TrainConfig(**{key: value for key, value in run.items() if key in TrainConfig.__dataclass_fields__})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = FireData(config)
    model = build_model(config, data).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return {**run, "resolved_precision": resolve_precision(config.precision, device),
            **test_metrics(model, data, config, device), "rescored_from_checkpoint": str(checkpoint)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--runs", default="results/benchmark_runs.jsonl", help="records of the runs the checkpoints came from")
    parser.add_argument("--output", default="results/benchmark_runs.jsonl")
    parser.add_argument("--eval-batch-size", type=int, default=0, help="override the run's evaluation batch size (smaller GPUs)")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), help="override precision for this device")
    args = parser.parse_args()
    runs = {run["run_name"]: run for run in map(json.loads, Path(args.runs).read_text().splitlines())}
    for checkpoint in args.checkpoints:
        run = dict(runs[checkpoint.stem], compile=0)
        if args.eval_batch_size:
            run["eval_batch_size"] = args.eval_batch_size
        if args.precision:
            run["precision"] = args.precision
        result = evaluate(checkpoint, run)
        with Path(args.output).open("a") as output:
            output.write(json.dumps(result) + "\n")
        print(f"{result['run_name']}: test AP {result['test_ap']:.4f} (centre128 {result['test_ap_centre128']:.4f}) "
              f"F1 {result['test_f1']:.4f} IoU {result['test_iou']:.4f}", flush=True)


if __name__ == "__main__":
    main()
