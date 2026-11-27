# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU restoration costs including input mutations and preserved aliases."""

import pytest
import torch

from aitune.torch.module.forward_effects_spec import ForwardEffectsCapture
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.benchmarks.micro.reporting import record_timings


class CacheModel(torch.nn.Module):
    def __init__(self, mutation, alias_count):
        super().__init__()
        self.mutation = mutation
        self.alias_count = alias_count

    def forward(self, x, state):
        for cache in state:
            if self.mutation == "copy":
                cache["tensor"].copy_(x)
            elif self.mutation == "replace":
                cache["tensor"] = x + 1
        return {"value": x * 2, "aliases": state[: self.alias_count], "metadata": []}


@pytest.mark.parametrize("gc_enabled", [False, True], ids=["gc-off", "gc-on"])
@pytest.mark.parametrize("alias_count", [0, 1, 8])
@pytest.mark.parametrize("mutation", ["none", "copy", "replace"])
def test_output_restoration(mutation, alias_count, gc_enabled, record_property):
    """Time GraphSpec restoration including mutation handling and live alias lookup.

    The local model records the contract outside timing; this measures CPU restoration alone,
    without compilation, tensor allocation, or model execution in the measured region.
    """
    model = CacheModel(mutation, alias_count)
    signature = ForwardSignature.from_callable(model.forward)
    value = torch.ones(16)
    state = [{"tensor": torch.zeros_like(value)} for _ in range(8)]
    arguments = signature.normalize((value, state), {}).arguments
    input_spec = SampleMetadata.from_inputs(arguments)
    capture = ForwardEffectsCapture.before_call(arguments)
    output = model(value, state)
    output_spec = SampleMetadata.from_outputs(output)
    effects = capture.after_call(arguments, output, output_spec)
    graph = GraphSpec("reconstruction-baseline", input_spec, output_spec, signature, effects)
    flat_outputs = output_spec.flatten_tensors(output) + tuple(
        mutation.locator.get_value(arguments) for mutation in graph.extra_tensor_mutations()
    )
    runtime_state = [{"tensor": torch.zeros_like(value)} for _ in range(8)]
    runtime_arguments = {"x": value, "state": runtime_state}
    original_tensors = tuple(cache["tensor"] for cache in runtime_state)

    def check_restoration():
        restored = graph.restore_outputs(flat_outputs, runtime_arguments)
        assert restored["value"] is flat_outputs[0]
        assert type(restored) is dict
        assert type(restored["aliases"]) is list
        assert len(restored["aliases"]) == alias_count
        for index, cache in enumerate(restored["aliases"]):
            assert cache is runtime_state[index]
        for index, cache in enumerate(runtime_state):
            expected = value + 1 if mutation == "replace" else value if mutation == "copy" else torch.zeros_like(value)
            torch.testing.assert_close(cache["tensor"], expected)
            if mutation != "replace":
                assert cache["tensor"] is original_tensors[index]
        return restored

    first = check_restoration()
    first["metadata"].append("changed")
    record_timings(
        {
            "restore_outputs": lambda: graph.restore_outputs(flat_outputs, runtime_arguments),
            "locator_restore_outputs": lambda: effects.restore_outputs(flat_outputs, runtime_arguments, output_spec),
        },
        gc_enabled,
        record_property,
    )
    second = check_restoration()
    assert second["metadata"] == effects.output_template["metadata"] == []
    assert second is not first
    # The same recorded graph must resolve aliases from each call's arguments. A prepared
    # strategy must never keep the previous caller's containers or mutation targets alive.
    previous_state = runtime_state
    runtime_state = [{"tensor": torch.zeros_like(value)} for _ in range(8)]
    runtime_arguments["state"] = runtime_state
    original_tensors = tuple(cache["tensor"] for cache in runtime_state)
    third = check_restoration()
    for index, cache in enumerate(third["aliases"]):
        assert cache is not previous_state[index]
