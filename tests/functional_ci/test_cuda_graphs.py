# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT CUDA graph eligibility and reuse when sequence lengths change."""

from unittest.mock import patch

import pytest
import torch

import aitune.torch as ait
from aitune.torch.backend import TensorRTBackend, TensorRTProfile
from aitune.torch.dynamic_shapes import DynamicShapes
from tests.functional_ci.backends import tensorrt_backend
from tests.functional_ci.lifecycle import TENSORRT_FP32, pinned_strategy, selected_backend
from tests.functional_ci.models import IN_FEATURES, make_input, make_mlp

LENGTHS = (16, 32, 64, 24)


@pytest.mark.matrix("models.dynamic-non-batch/tensorrt")
def test_dynamic_sequence_excludes_cuda_graph_capture(device: torch.device):
    """An explicit sequence range runs normally with CUDA graphs enabled, including unrecorded lengths."""
    backend = tensorrt_backend(use_cuda_graphs=True, use_dynamo=False, opset_version=20)
    dynamic_shapes: DynamicShapes = {"input": (1, ait.DynamicDim("tokens", min=16, opt=32, max=64), IN_FEATURES)}
    reference = make_mlp(device)
    tuned = ait.Module(make_mlp(device), "sequence", strategy=pinned_strategy(backend), dynamic_shapes=dynamic_shapes)
    recorded = make_input(device, batch_size=32).reshape(1, 32, IN_FEATURES)
    ait.tune(tuned, list(recorded.unbind(0)), batch_sizes=[1], device=device)

    actual = selected_backend(tuned, TensorRTBackend)
    assert isinstance(actual, TensorRTBackend)
    assert actual.key() == backend.key()
    graphs = actual._cuda_graphs
    assert not graphs.static_profile_indices
    with (
        patch.object(graphs, "_capture", wraps=graphs._capture) as capture,
        torch.no_grad(),
    ):
        for seed in (5, 6):
            for length in LENGTHS:
                value = make_input(device, batch_size=length, seed=seed).reshape(1, length, IN_FEATURES)
                TENSORRT_FP32.assert_close(tuned(value), reference(value))
                assert not graphs.profiles
                assert graphs.active is None
                assert not graphs.capture_failed
        capture.assert_not_called()


@pytest.mark.matrix("models.dynamic-non-batch/tensorrt")
def test_sequence_profiles_capture_and_reuse_cuda_graphs(device: torch.device):
    """Each static sequence profile captures once and reuses its graph after switching to other lengths."""
    profiles = [
        TensorRTProfile().add_input_shape(
            "input", (1, length, IN_FEATURES), (1, length, IN_FEATURES), (1, length, IN_FEATURES)
        )
        for length in LENGTHS
    ]
    backend = tensorrt_backend(use_cuda_graphs=True, use_dynamo=False, opset_version=20, profiles=profiles)
    reference = make_mlp(device)
    tuned = ait.Module(make_mlp(device), "sequence-profiles", strategy=pinned_strategy(backend))
    recorded = [make_input(device, batch_size=length) for length in LENGTHS]
    ait.tune(tuned, recorded, batch_sizes=[1], device=device)

    actual = selected_backend(tuned, TensorRTBackend)
    assert isinstance(actual, TensorRTBackend)
    assert actual.key() == backend.key()
    # Tuning may already run inference; start the assertions with an empty capture cache.
    actual.deactivate()
    actual.activate()
    graphs = actual._cuda_graphs
    assert graphs.static_profile_indices == set(range(len(LENGTHS)))
    assert not graphs.profiles
    captured = {}
    with (
        patch.object(graphs, "_capture", wraps=graphs._capture) as capture,
        torch.no_grad(),
    ):
        for seed in (5, 6):
            for index, length in enumerate(LENGTHS):
                value = make_input(device, batch_size=length, seed=seed).reshape(1, length, IN_FEATURES)
                TENSORRT_FP32.assert_close(tuned(value), reference(value))
                assert not graphs.capture_failed
                assert graphs.active is graphs.profiles[index]
                assert graphs.active.graph is not None
                if seed == 5:
                    captured[index] = graphs.active.graph
                    assert capture.call_count == index + 1
                else:
                    assert graphs.active.graph is captured[index]
                    assert capture.call_count == len(LENGTHS)
        assert set(graphs.profiles) == set(range(len(LENGTHS)))
