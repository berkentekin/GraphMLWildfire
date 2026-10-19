import ast
import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from graphml_wildfire.benchmark import YEARS
from graphml_wildfire.data.patches import CHANNEL_INDEX, CHANNELS
from graphml_wildfire.data.pixels import PIXEL_CHANNELS
from graphml_wildfire.evaluate import evaluate
from graphml_wildfire.graphs import apply_shift, pair_validity, stencil_offsets
from graphml_wildfire.models.gtcnn import GRAPH_MODES, EdgeAttention, GraphTimeCNN, GraphTimeLayer
from graphml_wildfire.smoke import smoke
from graphml_wildfire.train import TrainConfig, resolve_precision, run


def make_cache(root: Path) -> None:
    rng = np.random.default_rng(12)
    days, height, width, p = 8, 4, 5, 2
    for year in YEARS:
        fire = root / str(year) / "synthetic_fire"
        fire.mkdir(parents=True)
        pixels = (rng.random((days, height * p, width * p)) > 0.7).astype(np.uint8)
        counts = pixels.reshape(days, height, p, width, p).sum(axis=(2, 4)).astype(np.int16)
        features = rng.normal(size=(days, len(CHANNELS), height, width)).astype(np.float16)
        features[:, CHANNEL_INDEX["fire_frac"]] = counts / p**2
        features[:, CHANNEL_INDEX["burned_ever"]] = np.maximum.accumulate(counts > 0, axis=0)
        np.save(fire / "features.npy", features)
        np.save(fire / "fire_pixels.npy", counts)
        np.save(fire / "centre_fire.npy", counts)
        np.save(fire / "centre_pixels.npy", np.full((height, width), p**2, dtype=np.int16))
        for name in PIXEL_CHANNELS:
            values = pixels if name == "mask" else rng.random(pixels.shape).astype(np.float16)
            np.save(fire / f"pix_{name}.npy", values)
        np.save(fire / "pix_centre.npy", np.array([0, 0]))


class GraphModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.offsets = stencil_offsets()

    def test_learned_default_matches_original_product_shift(self):
        layer = GraphTimeLayer(8, 2, 2, self.offsets, 0)
        state = torch.randn(6, 8, 3, 4)
        valid = pair_validity(torch.ones(6, 3, 4), self.offsets)
        weights = layer.attention(state, valid)
        mix = torch.softmax(layer.mix[None, :, None, None] + layer.gate(state).float(), dim=1)
        spatial = apply_shift(state, weights, self.offsets)

        def yesterday(values):
            values = values.unflatten(0, (2, 3))
            return torch.cat([torch.zeros_like(values[:, :1]), values[:, :-1]], dim=1).flatten(0, 1)

        expected = mix[:, :1] * state + mix[:, 1:2] * spatial
        expected = expected + mix[:, 2:3] * yesterday(state) + mix[:, 3:4] * yesterday(spatial)
        torch.testing.assert_close(layer.product_shift(state, weights, 3), expected, rtol=0, atol=0)
        self.assertIsInstance(layer.attention, EdgeAttention)

    def test_uniform_weights_and_neighbour_mean_at_boundaries_and_holes(self):
        layer = GraphTimeLayer(4, 1, 2, self.offsets, 0, "uniform")
        valid = torch.ones(1, 4, 4)
        valid[:, 1, 1] = 0
        pairs = pair_validity(valid, self.offsets)
        state = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4)
        with patch("graphml_wildfire.models.gtcnn.apply_shift", wraps=apply_shift) as shift:
            layer(state, pairs, 1)
        weights = shift.call_args.args[1]
        for r in range(4):
            for c in range(4):
                neighbours = [
                    (r + dr, c + dc) for dr, dc in self.offsets
                    if 0 <= r + dr < 4 and 0 <= c + dc < 4 and valid[0, r + dr, c + dc] and valid[0, r, c]
                ]
                for k, (dr, dc) in enumerate(self.offsets):
                    expected = 1 / len(neighbours) if (r + dr, c + dc) in neighbours else 0.0
                    torch.testing.assert_close(weights[0, :, k, r, c], torch.full((2,), expected))
        actual = apply_shift(state, weights, self.offsets)
        expected = torch.zeros_like(state)
        for r in range(4):
            for c in range(4):
                neighbours = [
                    state[0, :, r + dr, c + dc] for dr, dc in self.offsets
                    if 0 <= r + dr < 4 and 0 <= c + dc < 4 and pairs[0, self.offsets.index((dr, dc)), r, c]
                ]
                if neighbours:
                    expected[0, :, r, c] = torch.stack(neighbours).mean(0)
        torch.testing.assert_close(actual, expected)
        self.assertIsNone(layer.attention)
        self.assertFalse(any("attention" in name for name, _ in layer.named_parameters()))

    def test_uniform_isolated_node_is_zero_and_finite(self):
        layer = GraphTimeLayer(4, 1, 2, self.offsets, 0, "uniform")
        state = torch.randn(2, 4, 1, 1, requires_grad=True)
        pairs = pair_validity(torch.ones(2, 1, 1), self.offsets)
        out = layer(state, pairs, 2)
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_no_graph_renormalizes_self_and_temporal_mix(self):
        layer = GraphTimeLayer(4, 1, 2, self.offsets, 0, "none")
        state = torch.randn(3, 4, 2, 2)
        mix = torch.softmax(layer.mix[[0, 2]], dim=0)
        previous = torch.cat([torch.zeros_like(state[:1]), state[:-1]])
        expected = mix[0] * state + mix[1] * previous
        with patch("graphml_wildfire.models.gtcnn.apply_shift", side_effect=AssertionError("Spatial shift called")):
            torch.testing.assert_close(layer.product_shift(state, None, 3), expected)
            torch.testing.assert_close(layer.product_shift(state, None, 1), state)

    def test_no_graph_has_no_neighbour_dependence_but_keeps_temporal_path(self):
        model = GraphTimeCNN(3, hidden=8, layers=2, taps=2, heads=2, dropout=0, graph_mode="none").eval()
        inputs = torch.randn(1, 3, 3, 3, 3, requires_grad=True)
        out = model(inputs, torch.ones(1, 3, 3))
        gradient = torch.autograd.grad(out[0, 0, 1, 1], inputs)[0]
        self.assertGreater(gradient[0, 0, :, 1, 1].abs().sum().item(), 0)
        gradient[:, :, :, 1, 1] = 0
        self.assertEqual(gradient.abs().sum().item(), 0)

    def test_all_modes_have_finite_forward_and_backward(self):
        for mode in GRAPH_MODES:
            for days in (1, 3):
                with self.subTest(mode=mode, days=days):
                    model = GraphTimeCNN(3, hidden=8, layers=2, taps=2, heads=2, dropout=0, graph_mode=mode)
                    features = torch.randn(2, days, 3, 3, 4, requires_grad=True)
                    valid = torch.ones(2, 3, 4)
                    valid[1, -1] = 0
                    out = model(features, valid)
                    self.assertEqual(out.shape, (2, 8, 3, 4))
                    out.square().mean().backward()
                    self.assertTrue(torch.isfinite(out).all())
                    self.assertTrue(torch.isfinite(features.grad).all())

    def test_modes_reject_invalid_configuration(self):
        with self.assertRaises(ValueError):
            TrainConfig(graph_mode="invalid")
        with self.assertRaises(ValueError):
            TrainConfig(model="cnn", graph_mode="none")
        with self.assertRaises(ValueError):
            GraphTimeLayer(7, 1, 2, self.offsets, 0)

    def test_precision_selection(self):
        self.assertEqual(resolve_precision("auto", torch.device("cpu")), "fp32")
        with self.assertRaises(ValueError):
            resolve_precision("fp16", torch.device("cpu"))
        for supported, expected in ((True, "bf16"), (False, "fp16")):
            with patch("torch.cuda.is_bf16_supported", return_value=supported) as native_support:
                self.assertEqual(resolve_precision("auto", torch.device("cuda")), expected)
                native_support.assert_called_once_with(including_emulation=False)
                self.assertEqual(resolve_precision("fp32", torch.device("cuda")), "fp32")
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            with self.assertRaises(ValueError):
                resolve_precision("bf16", torch.device("cuda"))


class CacheIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_three_modes_train_evaluate_save_and_reload(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"WANDB_MODE": "disabled"}), \
                patch("torch.cuda.is_available", return_value=False):
            root = Path(temp)
            make_cache(root / "cache")
            for mode in GRAPH_MODES:
                with self.subTest(mode=mode):
                    config = TrainConfig(
                        graph_mode=mode, data_dir=str(root / "cache"), precision="fp32",
                        hidden=8, layers=1, taps=1, heads=2, pixel_hidden=4, decoder_hidden=4,
                        crop=4, batch_size=2, eval_batch_size=1, epochs=1, compile=0,
                        checkpoint_dir=str(root / "checkpoints"), results_file=str(root / "runs.jsonl"),
                    )
                    result = run(config)
                    for name in ("val_ap", "test_ap", "test_ap_centre128", "test_f1", "test_iou"):
                        self.assertTrue(np.isfinite(result[name]), name)
                        self.assertGreaterEqual(result[name], 0)
                        self.assertLessEqual(result[name], 1)
                    saved = torch.load(root / "checkpoints" / f"{config.run_name}.pt", weights_only=True)
                    for name, values in result["state"].items():
                        torch.testing.assert_close(saved[name], values)
                    metadata = json.loads((root / "checkpoints" / f"{config.run_name}.json").read_text())
                    self.assertEqual(metadata["graph_mode"], mode)
                    rescored = evaluate(root / "checkpoints" / f"{config.run_name}.pt", result)
                    self.assertEqual(rescored["test_ap_centre128"], result["test_ap_centre128"])
                    self.assertEqual(rescored["resolved_precision"], "fp32")
            records = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]
            self.assertEqual([row["graph_mode"] for row in records], list(GRAPH_MODES))
            self.assertEqual(len({row["run_name"] for row in records}), 3)
            results = smoke(TrainConfig(
                data_dir=str(root / "cache"), precision="fp32", batch_size=2, crop=4, compile=0,
            ))
            self.assertEqual([row["graph_mode"] for row in results], list(GRAPH_MODES))
            self.assertTrue(all(row["logits_shape"] == [2, 8, 8] for row in results))
            selected = smoke(TrainConfig(
                data_dir=str(root / "cache"), precision="fp32", batch_size=2, crop=4, compile=0,
            ), modes=("none", "uniform"))
            self.assertEqual([row["graph_mode"] for row in selected], ["none", "uniform"])
            gc.collect()


class NotebookTests(unittest.TestCase):
    def test_notebook_plans_twelve_no_graph_runs_at_batch_32(self):
        repo = Path(__file__).resolve().parents[1]
        notebook = json.loads((repo / "notebooks" / "kaggle_pixel_time_ablations.ipynb").read_text())
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        with tempfile.TemporaryDirectory() as temp:
            command = Mock()
            snapshot = Mock()
            root = Path(temp)
            namespace = {
                "OUTPUT_DIR": root, "REPO_DIR": repo, "DATA_DIR": root / "cache",
                "json": json, "FOLDS_TO_RUN": list(range(12)), "SEEDS": [0], "MODES": ["none"],
                "EPOCHS": 30, "BATCH_SIZE": 32, "EVAL_BATCH_SIZE": 1, "PRECISION": "auto",
                "RUN_TRAINING": True, "SMOKE_PASSED": True,
                "run_logged": command, "snapshot_results": snapshot,
                "subprocess": __import__("subprocess"), "sys": __import__("sys"),
                "SMOKE_SETTINGS": (str(root / "cache"), "auto", 0, 32, ("none",)),
            }
            exec(compile(cells["reference-check"], "<reference cell>", "exec"), namespace)
            self.assertEqual(len(namespace["REFERENCE_RUNS"]), 12)
            self.assertTrue((root / "learned_reference_runs.json").is_file())
            exec(compile(cells["training"], "<training cell>", "exec"), namespace)
            self.assertEqual(command.call_count, 12)
            self.assertEqual(snapshot.call_count, 12)
            for fold, call in enumerate(command.call_args_list):
                args = call.args[0]
                self.assertEqual(args[args.index("--graph-mode") + 1], "none")
                self.assertEqual(args[args.index("--fold") + 1], str(fold))
                self.assertEqual(args[args.index("--batch-size") + 1], "32")
                self.assertEqual(args[args.index("--epochs") + 1], "30")
                self.assertEqual(args[args.index("--seed") + 1], "0")
            namespace["BATCH_SIZE"] = 4
            with self.assertRaisesRegex(ValueError, "batch_size"):
                exec(compile(cells["reference-check"], "<reference cell>", "exec"), namespace)
            namespace["BATCH_SIZE"] = 32
            namespace["run_logged"] = Mock(side_effect=RuntimeError("training failed"))
            snapshot.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                exec(compile(cells["training"], "<training cell>", "exec"), namespace)
            snapshot.assert_called_once()

    def test_notebook_smoke_runs_only_selected_modes(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_pixel_time_ablations.ipynb"
        cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads(path.read_text())["cells"]}
        settings = ast.parse(cells["settings"])
        modes = next(ast.literal_eval(node.value) for node in settings.body
                     if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODES" for t in node.targets))
        self.assertEqual(modes, ["none"])
        self.assertIn("FOLDS_TO_RUN = list(range(12))", cells["settings"])
        self.assertIn("BATCH_SIZE = 32", cells["settings"])
        with tempfile.TemporaryDirectory() as temp:
            command = Mock()
            namespace = {
                "OUTPUT_DIR": Path(temp), "subprocess": __import__("subprocess"), "sys": __import__("sys"),
                "DATA_DIR": Path(temp), "REPO_DIR": Path(temp), "PRECISION": "fp32",
                "FOLDS_TO_RUN": list(range(12)), "BATCH_SIZE": 32, "MODES": modes,
                "run_logged": command,
            }
            exec(compile(cells["smoke"], "<smoke cell>", "exec"), namespace)
            args = command.call_args.args[0]
            self.assertEqual(args[args.index("--graph-modes") + 1:], modes)
            self.assertNotIn("learned", args)
            self.assertNotIn("uniform", args)
            self.assertEqual(args[args.index("--batch-size") + 1], "32")
            self.assertEqual(namespace["SMOKE_SETTINGS"][-1], tuple(modes))

    def test_notebook_streams_logs_before_process_exit_and_surfaces_errors(self):
        import subprocess
        import sys
        import os

        path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_pixel_time_ablations.ipynb"
        cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads(path.read_text())["cells"]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            namespace = {"Path": Path, "subprocess": subprocess, "os": os, "REPO_DIR": root}
            exec(compile(cells["live-logging"], "<logging cell>", "exec"), namespace)
            acknowledgement = root / "ack"
            messages = []

            def receive(message, **kwargs):
                messages.append(str(message))
                if "first message" in str(message):
                    acknowledgement.touch()

            namespace["print"] = receive
            script = (
                "import sys,time; from pathlib import Path; "
                "print('first message', flush=True); "
                "deadline=time.monotonic()+5\n"
                "while not Path('ack').exists() and time.monotonic()<deadline: time.sleep(.01)\n"
                "assert Path('ack').exists(), 'Output was not streamed live'\n"
                "print('stderr message', file=sys.stderr, flush=True)\n"
            )
            log = root / "logs" / "test.log"
            namespace["run_logged"]([sys.executable, "-u", "-c", script], log)
            self.assertTrue(acknowledgement.is_file())
            self.assertIn("first message", log.read_text())
            self.assertIn("stderr message", log.read_text())
            self.assertTrue(any("stderr message" in message for message in messages))
            with self.assertRaises(subprocess.CalledProcessError):
                namespace["run_logged"](
                    [sys.executable, "-u", "-c", "print('failure detail'); raise SystemExit(7)"], log,
                )
            self.assertIn("failure detail", log.read_text())

    def test_notebook_structure_and_python_syntax(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_pixel_time_ablations.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")


if __name__ == "__main__":
    unittest.main()
