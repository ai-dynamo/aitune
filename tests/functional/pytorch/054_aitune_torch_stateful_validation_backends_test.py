# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# docker_image = "nvcr.io/nvidia/pytorch:26.06-py3"
# scope = "always"
# allow_failure = false
# ///

"""Exercise mutable-input structural acceptance and checkpoint round trips on real GPU backends."""

import logging
import sys
import tempfile
from pathlib import Path

import pytest
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
from tests.toy_models.stateful import MUTATION_OPERATIONS, update_tensor
from tests.utilities.helpers import make_recorded_graph_spec, make_sample_store


class StatefulAccumulator(torch.nn.Module):
    def __init__(self, operation="add_"):
        super().__init__()
        self.operation = operation

    def forward(self, value, cache, index):
        update_tensor(cache["value"], value.sin(), index, self.operation)
        cache["counter"].add_(1)
        return {"value": cache["value"], "cache": cache}


def sample(scale: float):
    return (
        (
            torch.arange(8, device="cuda", dtype=torch.float32) + scale,
            {"value": torch.ones(8, device="cuda"), "counter": torch.tensor(0, device="cuda")},
            torch.arange(8, device="cuda").roll(int(scale)),
        ),
        {},
    )


def run_backend(backend_class, operation="add_"):
    device = torch.device("cuda:0")
    model = StatefulAccumulator(operation).eval().to(device)
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
                model, f"stateful_{operation}", spec, make_sample_store([recorded], root), device, root / "build"
            )
            # Exercise the structural contract on new inputs across three calls; this is not an accuracy test.
            stateful_validation = StatefulValidation(root / "held-out", spec)
            stateful_validation.record(StatefulAccumulator(operation), [sample(2.75)])
            stateful_validation.check(backend.infer, device, name=f"{backend_class.__name__} held-out")
            state = backend.to_dict()
            backend.deactivate()
            restored = backend_class.from_dict(StatefulAccumulator(operation).to(device), state)
            restored.activate()
            stateful_validation.check(restored.infer, device, name=f"{backend_class.__name__} checkpoint reload")
        finally:
            for candidate in (backend, restored):
                if candidate is not None and candidate.is_active:
                    candidate.deactivate()


@pytest.mark.parametrize("operation", MUTATION_OPERATIONS)
def test_stateful_inductor_aot(operation):
    run_backend(TorchInductorAotBackend, operation)


def test_stateful_inductor_jit():
    run_backend(TorchInductorJitBackend)


@pytest.mark.parametrize("operation", MUTATION_OPERATIONS)
def test_stateful_torch_tensorrt_aot(operation):
    run_backend(TorchTensorRTAotBackend, operation)


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
        operations = (
            MUTATION_OPERATIONS if backend_type in (TorchInductorAotBackend, TorchTensorRTAotBackend) else ("add_",)
        )
        for operation in operations:
            run_backend(backend_type, operation)
