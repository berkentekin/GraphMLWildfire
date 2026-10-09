"""Re-score saved checkpoints under the benchmark protocol and append a run record per checkpoint.

    uv run python -m graphml_wildfire.evaluate data/checkpoints/pixel_gtcnn_attention_w5_p2_f*.pt

Every checkpoint is matched to the run that produced it (by run name, in `--runs`) to recover
its configuration; the new record keeps that configuration and replaces the test metrics.
"""

import argparse
import json
from pathlib import Path

import torch

from .benchmark import average_precision, best_f1_threshold, threshold_metrics
from .train import FireData, TrainConfig, build_model, predict


def evaluate(checkpoint: Path, run: dict) -> dict:
    config = TrainConfig(**{key: value for key, value in run.items() if key in TrainConfig.__dataclass_fields__})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = FireData(config)
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    val_scores, val_regions = predict(model, data, "val", config, device)
    threshold = best_f1_threshold(val_scores, *val_regions["full"])
    val_ap = average_precision(val_scores, *val_regions["full"])
    del val_scores, val_regions
    test_scores, test_regions = predict(model, data, "test", config, device)
    full_ap = average_precision(test_scores, *test_regions["full"])
    centre_from_full = average_precision(test_scores, *test_regions["centre"])
    del test_scores, test_regions
    centre_scores, centre_regions = predict(model, data, "test", config, device, centre_input=True)
    return {
        **run,
        "val_ap": val_ap,
        "test_ap": full_ap,
        "test_ap_centre128": average_precision(centre_scores, *centre_regions["centre"]),
        "test_ap_centre128_full_input": centre_from_full,
        **{f"test_{name}": value for name, value in threshold_metrics(centre_scores, *centre_regions["centre"], 0.0).items()},
        **{f"test_{name}_tuned": value for name, value in threshold_metrics(centre_scores, *centre_regions["centre"], threshold).items()},
        "rescored_from_checkpoint": str(checkpoint),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--runs", default="results/benchmark_runs.jsonl", help="records of the runs the checkpoints came from")
    parser.add_argument("--output", default="results/benchmark_runs.jsonl")
    args = parser.parse_args()
    runs = {run["run_name"]: run for run in map(json.loads, Path(args.runs).read_text().splitlines())}
    for checkpoint in args.checkpoints:
        result = evaluate(checkpoint, runs[checkpoint.stem])
        with Path(args.output).open("a") as output:
            output.write(json.dumps(result) + "\n")
        print(f"{result['run_name']}: test AP {result['test_ap']:.4f} (centre128 {result['test_ap_centre128']:.4f}) "
              f"F1 {result['test_f1']:.4f} IoU {result['test_iou']:.4f}", flush=True)


if __name__ == "__main__":
    main()
