# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = []
# scope = "always"
# ///

import gc
from logging import DEBUG, basicConfig

import torch
import torch.nn as nn

from aitune.torch import Module, OneBackendStrategy, tune
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig
from aitune.torch.dataloader import DataLoaderFactory
from aitune.torch.module_registry import MODULE_REGISTRY

FEATURES = 16


class NestedInputModel(nn.Module):
    """Small TensorRT-compatible model with a list of tensor inputs."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(FEATURES, FEATURES)

    def forward(self, x: torch.Tensor, residuals: list[torch.Tensor]) -> torch.Tensor:
        """Combine the primary input with residual tensors before projection."""
        return self.projection(x + residuals[0] - residuals[1])


def _make_dataset(device: torch.device) -> list[dict]:
    return [
        {
            "x": torch.randn(FEATURES, device=device),
            "residuals": [
                torch.randn(FEATURES, device=device),
                torch.randn(FEATURES, device=device),
            ],
        }
        for _ in range(4)
    ]


def _collate_nested_inputs(samples: list[dict]) -> dict:
    return {
        "x": torch.stack([sample["x"] for sample in samples]),
        "residuals": [
            torch.stack([sample["residuals"][index] for sample in samples])
            for index in range(len(samples[0]["residuals"]))
        ],
    }


def _tune_and_validate(batch_sizes: list[int]) -> None:
    device = torch.device("cuda")
    model = NestedInputModel().eval().to(device)
    dataset = _make_dataset(device)
    validation_inputs = _collate_nested_inputs(dataset[: max(batch_sizes)])

    with torch.no_grad():
        expected = model(**validation_inputs)

    strategy = OneBackendStrategy(TensorRTBackend(TensorRTBackendConfig(use_dynamo=True)))
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(False)
    shape_mode = "static" if len(batch_sizes) == 1 else "dynamic"
    tuned_model = Module(model, f"nested-input-model-{shape_mode}", strategy=strategy)

    try:
        tune(
            tuned_model,
            DataLoaderFactory(dataset, collate_fn=_collate_nested_inputs),
            batch_sizes=batch_sizes,
            max_num_batches_per_batch_size=1,
            device=device,
            ignore_failing_modules=False,
        )
        actual = tuned_model(**validation_inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    finally:
        tuned_model.deactivate()
        MODULE_REGISTRY.clear()
        del tuned_model, model, dataset, validation_inputs, expected
        gc.collect()
        torch.cuda.empty_cache()


def test_tune_nested_static_and_dynamic_shapes_with_onnx_dynamo():
    """Tune nested tensor inputs through the TensorRT ONNX-Dynamo backend."""
    _tune_and_validate([2])
    _tune_and_validate([2, 4])


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    test_tune_nested_static_and_dynamic_shapes_with_onnx_dynamo()
