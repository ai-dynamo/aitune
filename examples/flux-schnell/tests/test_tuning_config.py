# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate distributed strategy identity with the installed AITune backend, without GPUs."""

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flux_schnell_recipe import tune
from flux_schnell_recipe.config import load_config
from flux_schnell_recipe.distributed import rank_config


class TuningConfigTests(unittest.TestCase):
    def test_same_strategy_keeps_runtime_timing_caches_separate(self):
        import aitune.torch as ait
        import aitune.torch.backend as backend_api
        from aitune.torch.distributed import DistributedContext

        # Torch-TensorRT cannot import its full settings without driver access.
        # Substitute only its option container; exercise AITune's real serializer
        # and runtime cache-path rewriting below.
        @dataclass
        class CompileOptions:
            min_block_size: int = 5
            truncate_double: bool = False
            timing_cache_path: str = ""
            use_distributed_mode_trace: bool = False

        @dataclass
        class CpuJitConfig(backend_api.TorchTensorRTJitBackendConfig):
            compile_config: CompileOptions = field(default_factory=CompileOptions)

        class StrategyCaptured(Exception):
            pass

        strategy_type = ait.MaxThroughputStrategy
        strategies, trt_backends = [], []

        def capture(*, backends):
            strategies.append(strategy_type(backends=backends).enable_find_max_batch_size(False))
            trt_backends.append(backends[0])
            raise StrategyCaptured

        cfg = load_config(Path(__file__).resolve().parents[1] / "recipes/multi-gpu.yaml")
        args = SimpleNamespace(model_id=cfg["model"]["base_model"], checkpoint=None)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workers"
            for rank in range(2):
                # Exercise the recipe through strategy construction. Model execution is mocked;
                # serialization and cache resolution are real AITune.
                with patch.object(tune, "configure_cache"), patch.object(tune, "load_model"), \
                        patch.object(tune, "synchronize"), patch.object(ait, "inspect"), \
                        patch("aitune.torch.backend.torch_tensorrt_jit_backend.assert_cuda_is_available"), \
                        patch("aitune.torch.backend.torch_tensorrt_jit_backend.assert_torch_tensorrt"), \
                        patch.object(backend_api, "TorchTensorRTConfig", CompileOptions), \
                        patch.object(backend_api, "TorchTensorRTJitBackendConfig", CpuJitConfig), \
                        patch.object(ait, "MaxThroughputStrategy", side_effect=capture):
                    with self.assertRaises(StrategyCaptured):
                        tune.run(args, rank_config(cfg, rank), run_dir=root / f"rank-{rank}")

            self.assertEqual(strategies[0].to_json_dict(), strategies[1].to_json_dict())
            cache_paths = []
            for rank, backend in enumerate(trt_backends):
                context = DistributedContext(rank=rank, local_rank=rank, world_size=2)
                with patch("aitune.torch.distributed.distributed_context", return_value=context):
                    settings = backend._compile_settings()
                self.assertTrue(settings["use_distributed_mode_trace"])
                path = Path(settings["timing_cache_path"])
                self.assertEqual(path.parent, root)
                self.assertIn(f"rank-{rank}-of-2", path.name)
                cache_paths.append(path)
            self.assertNotEqual(*cache_paths)
            self.assertEqual(strategies[0].to_json_dict(), strategies[1].to_json_dict())


if __name__ == "__main__":
    unittest.main()
