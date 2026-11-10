# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for adapting structured module calls to flat tensors."""

import copy
import functools

import torch

from aitune.torch.libs.exporting import TensorExportAdapter
from aitune.torch.module.forward_contract import ForwardContract, InputMutation, InputMutationKind
from aitune.torch.utils.tensor import format_tensor_name
from tests.utilities.helpers import make_graph_spec


def _make_adapter(module, sample):
    call_sample = copy.deepcopy(sample)
    with torch.no_grad():
        output = module(*call_sample[0], **call_sample[1])
    graph_spec = make_graph_spec(module.forward, sample, output, strict=True)
    return TensorExportAdapter(module, sample, graph_spec), graph_spec


def test_adapter_exports_nested_inputs_and_outputs_as_flat_tensors():
    class NestedModule(torch.nn.Module):
        def forward(self, x, state):
            return {"sum": x + state["bias"], "nested": (x * 2,)}

    module = NestedModule()
    x = torch.ones(2)
    bias = torch.full((2,), 3.0)
    sample = ((x, {"bias": bias, "label": "constant"}), {})
    adapter, graph_spec = _make_adapter(module, sample)
    arguments = graph_spec.forward_signature.normalize(*sample).arguments

    flat_inputs = graph_spec.input_spec.flatten_tensors(arguments)
    exported = torch.export.export(adapter, flat_inputs, strict=True)
    flat_outputs = exported.module()(*flat_inputs)

    torch.testing.assert_close(flat_outputs, (x + bias, x * 2))


def test_adapter_preserves_keyword_call_layout_for_wrapped_modules():
    def require_kwargs(forward):
        @functools.wraps(forward)
        def wrapper(self, *args, **kwargs):
            if args:
                raise TypeError("arguments must remain keyword arguments")
            return forward(self, **kwargs)

        return wrapper

    class KeywordModule(torch.nn.Module):
        @require_kwargs
        def forward(self, x):
            return x + 1

    module = KeywordModule()
    x = torch.ones(2)
    adapter, _ = _make_adapter(module, ((), {"x": x}))

    torch.testing.assert_close(adapter(x), (x + 1,))


def test_adapter_isolates_mutable_non_tensor_inputs_between_calls():
    class IncrementState(torch.nn.Module):
        def forward(self, x, state):
            state[0] += 1
            return x * state[0]

    module = IncrementState()
    x = torch.ones(2)
    adapter, _ = _make_adapter(module, ((x, [0]), {}))

    first = adapter(x)
    second = adapter(x)

    torch.testing.assert_close(first[0], x)
    torch.testing.assert_close(second[0], x)


def test_adapter_appends_mutated_tensor_state_to_graph_outputs():
    class UpdateState(torch.nn.Module):
        def forward(self, x, state):
            state[0].add_(x)
            return x * 2

    module = UpdateState()
    x = torch.ones(2)
    state = torch.zeros(2)
    sample = ((x, [state]), {})
    adapter, graph_spec = _make_adapter(module, sample)
    state_locator = graph_spec.input_spec.tensor_data[1][0]
    graph_spec.forward_contract = ForwardContract(
        output_template=None,
        input_mutations=(InputMutation(state_locator, InputMutationKind.COPY_TENSOR, output_index=1),),
    )
    adapter = TensorExportAdapter(module, sample, graph_spec)

    flat_outputs = adapter(x, state)
    runtime_state = torch.zeros(2)
    arguments = graph_spec.forward_signature.normalize((x, [runtime_state]), {}).arguments
    result = graph_spec.restore_outputs(flat_outputs, arguments)

    torch.testing.assert_close(result, x * 2)
    assert arguments["state"][0] is runtime_state
    torch.testing.assert_close(runtime_state, x)
    assert graph_spec.output_names() == (
        format_tensor_name((), "output"),
        format_tensor_name(("state", 0), "state"),
    )
