# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for checkpoint placement and worker diagnostics."""

from pathlib import Path
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

from flux_schnell_recipe.correctness import run_worker


class WorkerFailureTests(unittest.TestCase):
    def test_worker_output_is_streamed_before_exit_and_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            acknowledged = run_dir / "acknowledged"
            # The child can only succeed if its output reaches the terminal while it is alive.
            command = [sys.executable, "-u", "-c", "\n".join([
                "from pathlib import Path",
                "import sys, time",
                "print('worker is loading', flush=True)",
                "deadline = time.monotonic() + 10",
                "while not Path(sys.argv[1]).exists():",
                "    if time.monotonic() > deadline: sys.exit(8)",
                "    time.sleep(0.01)",
                "print('worker finished', file=sys.stderr)",
            ]), str(acknowledged)]

            class Terminal(io.StringIO):
                def write(self, text):
                    if "worker is loading" in text:
                        acknowledged.touch()
                    return super().write(text)

            terminal = Terminal()
            with patch("flux_schnell_recipe.correctness.worker_command", return_value=command), \
                    patch("sys.stderr", terminal):
                run_worker(None, None, "correctness", "original", run_dir)
            self.assertEqual(terminal.getvalue(), (run_dir / "original.log").read_text())
            self.assertIn("worker finished", terminal.getvalue())

    def test_child_error_and_log_path_are_reported(self):
        command = [sys.executable, "-c", "import sys; print('underlying error', file=sys.stderr); sys.exit(7)"]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with patch("flux_schnell_recipe.correctness.worker_command", return_value=command):
                with self.assertRaises(RuntimeError) as caught:
                    run_worker(None, None, "correctness", "aitune", run_dir)
            message = str(caught.exception)
            self.assertIn("status 7", message)
            self.assertIn(str(run_dir / "aitune.log"), message)
            self.assertIn("underlying error", message)
            self.assertIn("underlying error", (run_dir / "aitune.log").read_text())


class CheckpointPlacementTests(unittest.TestCase):
    def test_cuda_checkpoint_is_staged_on_cpu(self):
        import torch
        from aitune.torch.checkpoint.local_torch_storage import LocalTorchStorage
        from flux_schnell_recipe.checkpoint import cpu_staging_storage

        if not torch.cuda.is_available():
            self.skipTest("CUDA required to exercise a checkpoint saved on GPU")
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "roundtrip.ait"
            tensor = torch.arange(8, device="cuda")
            LocalTorchStorage().save(artifact, {"weights": tensor, "device": torch.device("cuda:0")})
            restored = cpu_staging_storage().load(artifact)
            self.assertEqual(restored["weights"].device.type, "cpu")
            torch.testing.assert_close(restored["weights"], tensor.cpu())
            # Backend deployment still targets its saved GPU.
            self.assertEqual(restored["device"], torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
