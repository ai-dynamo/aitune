# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT CUDA graph capture and reuse across static profiles."""

import pytest
import torch

import aitune.torch as ait
from aitune.torch.backend import TensorRTBackend, TensorRTProfileMode
from aitune.torch.config import config
from tests.functional_ci.backends import BackendCase, tensorrt_backend, with_backends
from tests.functional_ci.lifecycle import assert_matches_eager, pinned_strategy, selected_backend
from tests.functional_ci.models import make_input, make_mlp


@pytest.mark.matrix("other.cuda-graphs-fp32/functional-ci")
@with_backends("cuda_graphs")
def test_cuda_graphs(case: BackendCase, device: torch.device, monkeypatch: pytest.MonkeyPatch):
    """Record static profiles, then alternate between graph replay and uncaptured shapes."""
    static_batches = (1, 3, 5)
    monkeypatch.setattr(config, "max_num_samples_stored", len(static_batches))
    backend = tensorrt_backend(
        use_dynamo=False,
        use_cuda_graphs=True,
        opset_version=20,
        profiles=TensorRTProfileMode.SAMPLES_USED,
    )
    tuned = ait.Module(make_mlp(device), "cuda-graphs", strategy=pinned_strategy(backend))
    ait.tune(
        tuned,
        list(make_input(device, batch_size=5).unbind(0)),
        batch_sizes=list(static_batches),
        max_num_batches_per_batch_size=1,
        device=device,
        ignore_failing_modules=False,
    )

    selected = selected_backend(tuned, TensorRTBackend)
    assert isinstance(selected, TensorRTBackend)
    graphs = selected._cuda_graphs
    fallback_index = len(static_batches)
    assert len(selected._trt_optimization_profiles) == fallback_index + 1
    assert graphs.static_profile_indices == set(range(fallback_index))

    captured = {}
    reference = make_mlp(device)
    for batch_size in static_batches:
        assert_matches_eager(tuned, reference, device, case.tolerance, batch_sizes=(batch_size,), seeds=(11,))
        assert graphs.active is not None
        graph = graphs.active.graph
        assert graph is not None
        assert all(graph is not previous for previous in captured.values())
        captured[batch_size] = graph

    assert graphs.profiles.keys() == graphs.static_profile_indices
    # Change the input values and alternate static and dynamic profiles to detect stale inputs or graph selection.
    for batch_size in (5, 2, 1, 4, 3):
        assert_matches_eager(tuned, reference, device, case.tolerance, batch_sizes=(batch_size,), seeds=(12,))
        if batch_size in captured:
            assert graphs.active is not None
            assert graphs.active.graph is captured[batch_size]
        else:
            assert graphs.active is None
            assert selected._context is not None
            assert selected._context.active_optimization_profile == fallback_index
    assert graphs.profiles.keys() == graphs.static_profile_indices
