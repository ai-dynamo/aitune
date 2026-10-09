# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for actionable GPU visibility and distributed worker failures."""

from datetime import timedelta
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from flux_schnell_recipe import distributed, distributed_worker, multi_gpu


class DistributedErrorTests(unittest.TestCase):
    def test_collective_timeout_is_short_only_when_requested(self):
        torch = ModuleType("torch")
        torch.device = lambda kind, rank: (kind, rank)
        torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
            set_device=Mock(), is_bf16_supported=lambda: True, get_device_name=lambda: "test GPU",
            get_device_capability=lambda: (9, 0),
            get_device_properties=lambda rank: SimpleNamespace(total_memory=80))
        torch.distributed = ModuleType("torch.distributed")
        torch.distributed.init_process_group = Mock()
        env = {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}
        with patch.dict("sys.modules", {"torch": torch, "torch.distributed": torch.distributed}), \
                patch.dict(os.environ, env), patch.object(distributed, "collect", return_value=[{}, {}]):
            for kwargs, seconds in (({}, 1800), ({"timeout_s": 120}, 120)):
                distributed.initialize({"execution": {"gpu_count": 2}}, **kwargs)
                self.assertEqual(torch.distributed.init_process_group.call_args.kwargs["timeout"],
                                 timedelta(seconds=seconds))

    def test_launcher_streams_output_before_workers_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "workers"
            acknowledged = directory / "acknowledged"
            script = "\n".join([
                "import json, sys, time",
                "from pathlib import Path",
                "directory = Path(sys.argv[1])",
                "print('initializing NCCL', flush=True)",
                "deadline = time.monotonic() + 10",
                "while not (directory / 'acknowledged').exists():",
                "    if time.monotonic() > deadline: sys.exit(8)",
                "    time.sleep(0.01)",
                "for rank in range(2):",
                "    path = directory / f'rank-{rank}'",
                "    path.mkdir()",
                "    (path / 'report.json').write_text(json.dumps({'rank': rank}))",
                "print('workers finished', file=sys.stderr)",
            ])
            popen = subprocess.Popen

            def child(command, **kwargs):
                return popen([sys.executable, "-u", "-c", script, str(directory)], **kwargs)

            class Terminal(io.StringIO):
                def write(self, text):
                    if "initializing NCCL" in text:
                        acknowledged.touch()
                    return super().write(text)

            terminal = Terminal()
            args = SimpleNamespace(model_id="test/model", checkpoint=None)
            cfg = SimpleNamespace(path=Path(temporary) / "recipe.yaml")
            with patch.object(multi_gpu.subprocess, "Popen", side_effect=child), patch("sys.stderr", terminal):
                ranks = multi_gpu.launch(args, cfg, "tune", "original", directory)
            self.assertEqual(ranks, [{"rank": 0}, {"rank": 1}])
            self.assertEqual(terminal.getvalue(), (directory / "torchrun.log").read_text())
            self.assertIn("workers finished", terminal.getvalue())

    def test_visibility_error_reports_actual_count_and_container_fix(self):
        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        torch.distributed = ModuleType("torch.distributed")
        env = {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0",
               "CUDA_VISIBLE_DEVICES": "0"}
        with patch.dict("sys.modules", {"torch": torch, "torch.distributed": torch.distributed}), \
                patch.dict(os.environ, env):
            with self.assertRaises(ValueError) as caught:
                distributed.initialize({"execution": {"gpu_count": 2}})
        message = str(caught.exception)
        self.assertIn("visible GPUs=1", message)
        self.assertIn('--gpus \'"device=0,1"\'', message)
        self.assertIn("current value: 0", message)

    def test_initialization_failure_is_saved_before_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ["worker", "--config", "recipe.yaml", "--model-id", "test/model", "--action", "tune",
                    "--variant", "original", "--run-dir", directory]
            with patch("sys.argv", argv), patch.dict(os.environ, {"RANK": "0"}), \
                    patch.object(distributed_worker, "load_config", return_value={}), \
                    patch.object(distributed_worker, "configure_cache"), \
                    patch.object(distributed_worker, "initialize", side_effect=ValueError("visible GPUs=1")), \
                    patch.object(distributed_worker, "shutdown") as cleanup:
                with self.assertRaisesRegex(ValueError, "visible GPUs=1"):
                    distributed_worker.main()
            cleanup.assert_called_once()
            error = json.loads((Path(directory) / "rank-0/error.json").read_text())
            self.assertEqual(error["rank"], 0)
            self.assertIn("ValueError: visible GPUs=1", error["traceback"])

    def test_launcher_includes_worker_error_even_when_summary_is_long(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "workers"

            def failed_process(command, **kwargs):
                rank_dir = directory / "rank-0"
                rank_dir.mkdir()
                (rank_dir / "error.json").write_text(json.dumps({
                    "rank": 0, "traceback": "Traceback:\nValueError: visible GPUs=1\n"}))
                kwargs["stdout"].write("torchrun summary\n" * 60)
                return Mock(wait=Mock(return_value=1))

            args = SimpleNamespace(model_id="test/model", checkpoint=None)
            cfg = SimpleNamespace(path=Path(temporary) / "recipe.yaml")
            with patch.object(multi_gpu.subprocess, "Popen", side_effect=failed_process):
                with self.assertRaises(RuntimeError) as caught:
                    multi_gpu.launch(args, cfg, "tune", "original", directory)
            message = str(caught.exception)
            self.assertIn("Rank 0 worker traceback", message)
            self.assertIn("ValueError: visible GPUs=1", message)
            self.assertIn(str(directory / "torchrun.log"), message)


if __name__ == "__main__":
    unittest.main()
