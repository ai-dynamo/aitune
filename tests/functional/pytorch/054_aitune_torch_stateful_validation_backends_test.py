# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# docker_image = "nvcr.io/nvidia/pytorch:26.06-py3"
# scope = "always"
# allow_failure = false
# ///

"""Exercise mutable-input acceptance and checkpoint round trips on real GPU backends."""

import logging
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aitune.torch.backend import (
    TorchInductorAotBackend,
    TorchInductorJitBackend,
    TorchTensorRTAotBackend,
    TorchTensorRTJitBackend,
)
from aitune.torch.task.stateful_validation import StatefulValidation
from aitune.torch.tune_strategy import OneBackendStrategy
from tests.utilities.helpers import make_recorded_graph_spec, make_sample_store


class StatefulAccumulator(torch.nn.Module):
    def forward(self, value, cache):
        cache["value"].add_(value.sin())
        cache["counter"].add_(1)
        return {"value": cache["value"], "cache": cache}


def sample(scale: float):
    return (
        (
            torch.full((8,), scale, device="cuda"),
            {"value": torch.zeros(8, device="cuda"), "counter": torch.tensor(0, device="cuda")},
        ),
        {},
    )


def run_backend(backend_class):
    device = torch.device("cuda:0")
    model = StatefulAccumulator().eval().to(device)
    recorded = sample(1.0)
    spec = make_recorded_graph_spec(model, recorded, strict=True)
    strategy = OneBackendStrategy(backend_class())
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(False)
    backend = restored = None
    with tempfile.TemporaryDirectory(prefix="aitune-stateful-functional-") as directory:
        root = Path(directory)
        try:
            backend = strategy.tune(
                model, "stateful", spec, make_sample_store([recorded], root), device, root / "build"
            )
            # Held-out values differ from recording/compilation; each check retains cache state for three calls.
            stateful_validation = StatefulValidation(root / "held-out", spec)
            stateful_validation.record(StatefulAccumulator(), [sample(2.75)])
            stateful_validation.check(backend.infer, device, name=f"{backend_class.__name__} held-out")
            state = backend.to_dict()
            backend.deactivate()
            restored = backend_class.from_dict(StatefulAccumulator().to(device), state)
            restored.activate()
            stateful_validation.check(restored.infer, device, name=f"{backend_class.__name__} checkpoint reload")
        finally:
            for candidate in (backend, restored):
                if candidate is not None and candidate.is_active:
                    candidate.deactivate()


def test_stateful_inductor_aot():
    run_backend(TorchInductorAotBackend)


def test_stateful_inductor_jit():
    run_backend(TorchInductorJitBackend)


def test_stateful_torch_tensorrt_aot():
    run_backend(TorchTensorRTAotBackend)


def test_stateful_torch_tensorrt_jit():
    run_backend(TorchTensorRTJitBackend)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    for backend_type in (
        TorchInductorAotBackend,
        TorchInductorJitBackend,
        TorchTensorRTAotBackend,
        TorchTensorRTJitBackend,
    ):
        run_backend(backend_type)
