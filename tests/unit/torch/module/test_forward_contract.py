# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for recording forward input mutations."""

from copy import deepcopy

import numpy as np
import torch

from aitune.torch.module.forward_contract import (
    ForwardContract,
    ForwardContractCapture,
    InputMutation,
    InputMutationKind,
    TensorMutationPolicy,
)
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.sample_metadata import SampleMetadata


def _capture(module, args):
    signature = ForwardSignature.from_callable(module.forward)
    arguments = signature.normalize(args, {}).arguments
    capture = ForwardContractCapture.before_call(arguments, deepcopy(arguments))
    output = module(*args)
    output_spec = SampleMetadata.from_outputs(output, strict=True)
    return capture.after_call(arguments, output, output_spec)


def test_capture_classifies_supported_leaf_mutations():
    class StatefulModule(torch.nn.Module):
        def forward(self, x, inplace, replaced, step):
            inplace[0].add_(x)
            replaced[0] = x + 1
            step[0] += 1
            return {"result": x * 2, "label": "recorded"}

    contract = _capture(
        StatefulModule(),
        (torch.ones(2), [torch.zeros(2)], [torch.zeros(2)], [0]),
    )

    mutations = {mutation.locator.path: mutation for mutation in contract.input_mutations}
    assert mutations[("inplace", 0)].kind == InputMutationKind.COPY_TENSOR
    assert mutations[("inplace", 0)].output_index == 1
    assert mutations[("replaced", 0)].kind == InputMutationKind.REPLACE_TENSOR
    assert mutations[("replaced", 0)].output_index == 2
    assert mutations[("step", 0)].kind == InputMutationKind.REPLACE_CONSTANT
    assert mutations[("step", 0)].value == 1
    assert contract.output_template == {"result": None, "label": "recorded"}


def test_capture_reuses_a_mutated_tensor_returned_as_output():
    class ReturnState(torch.nn.Module):
        def forward(self, state):
            state[0].add_(1)
            return state[0]

    contract = _capture(ReturnState(), ([torch.zeros(2)],))

    assert len(contract.input_mutations) == 1
    mutation = contract.input_mutations[0]
    assert mutation.kind == InputMutationKind.COPY_TENSOR
    assert mutation.output_index == 0


def test_capture_and_restore_numpy_leaf_replacement_with_incompatible_shapes():
    class ReplaceArray(torch.nn.Module):
        def forward(self, x, state):
            state[0] = np.ones(3)
            return x + 1

    contract = _capture(ReplaceArray(), (torch.zeros(2), [np.zeros(2)]))

    assert len(contract.input_mutations) == 1
    mutation = contract.input_mutations[0]
    assert mutation.locator.path == ("state", 0)
    assert mutation.kind == InputMutationKind.REPLACE_CONSTANT
    np.testing.assert_array_equal(mutation.value, np.ones(3))

    arguments = {"x": torch.zeros(2), "state": [np.zeros(2)]}
    output = torch.ones(2)
    output_spec = SampleMetadata.from_outputs(output)
    result = contract.restore_outputs((output,), arguments, output_spec)

    torch.testing.assert_close(result, output)
    np.testing.assert_array_equal(arguments["state"][0], np.ones(3))
    assert arguments["state"][0] is not mutation.value


def test_capture_detects_inference_tensor_mutation_without_a_version_counter():
    class UpdateState(torch.nn.Module):
        def forward(self, x, state):
            state.add_(x)
            return x

    with torch.inference_mode():
        contract = _capture(UpdateState(), (torch.ones(2), torch.zeros(2)))

    assert [mutation.kind for mutation in contract.input_mutations] == [InputMutationKind.COPY_TENSOR]


def test_restore_applies_input_mutations_and_rebuilds_output():
    arguments = {
        "copied": [torch.zeros(2)],
        "replaced": [torch.zeros(2)],
        "step": [0],
    }
    input_spec = SampleMetadata.from_inputs(arguments, strict=True)
    locators = {locator.path: locator for locator, _ in (*input_spec.tensor_data, *input_spec.other_data)}
    output_spec = SampleMetadata.from_outputs({"result": torch.zeros(2), "label": "recorded"}, strict=True)
    replacement = torch.full((2,), 3.0)
    contract = ForwardContract(
        output_template={"result": None, "label": "recorded"},
        input_mutations=(
            InputMutation(locators[("copied", 0)], InputMutationKind.COPY_TENSOR, output_index=1),
            InputMutation(locators[("replaced", 0)], InputMutationKind.REPLACE_TENSOR, output_index=2),
            InputMutation(locators[("step", 0)], InputMutationKind.REPLACE_CONSTANT, value=1),
        ),
    )
    original_copied = arguments["copied"][0]

    result = contract.restore_outputs((torch.ones(2), torch.full((2,), 2.0), replacement), arguments, output_spec)

    assert result["label"] == "recorded"
    torch.testing.assert_close(result["result"], torch.ones(2))
    assert arguments["copied"][0] is original_copied
    torch.testing.assert_close(original_copied, torch.full((2,), 2.0))
    assert arguments["replaced"][0] is replacement
    assert arguments["step"] == [1]


def test_restore_can_replace_a_tensor_normally_copied_into_the_input():
    original = torch.zeros(2)
    arguments = {"state": [original]}
    locator = SampleMetadata.from_inputs(arguments).tensor_data[0][0]
    output_spec = SampleMetadata.from_outputs(torch.zeros(2))
    replacement = torch.ones(2)
    contract = ForwardContract(
        output_template=None,
        input_mutations=(InputMutation(locator, InputMutationKind.COPY_TENSOR, output_index=1),),
    )

    contract.restore_outputs(
        (torch.zeros(2), replacement),
        arguments,
        output_spec,
        tensor_mutation_policies={1: TensorMutationPolicy.REPLACE_INPUT},
    )

    assert arguments["state"][0] is replacement
    assert arguments["state"][0] is not original


def test_restore_preserves_an_input_container_returned_in_mixed_outputs():
    class ReturnState(torch.nn.Module):
        def forward(self, x, state):
            state[0].add_(x)
            return {"result": x * 2, "state": state}

    module = ReturnState()
    recorded_args = (torch.ones(2), [torch.zeros(2)])
    contract = _capture(module, recorded_args)
    output_spec = SampleMetadata.from_outputs({"result": torch.zeros(2), "state": [torch.zeros(2)]}, strict=True)
    arguments = {"x": torch.full((2,), 3.0), "state": [torch.zeros(2)]}
    original_state = arguments["state"]
    original_state_tensor = original_state[0]

    result = contract.restore_outputs(
        (torch.full((2,), 6.0), torch.full((2,), 3.0)),
        arguments,
        output_spec,
    )

    assert result["state"] is original_state
    assert result["state"][0] is original_state_tensor
    torch.testing.assert_close(result["state"][0], torch.full((2,), 3.0))
    torch.testing.assert_close(result["result"], torch.full((2,), 6.0))


def test_restore_preserves_a_tensor_returned_directly_from_inputs():
    class ReturnInput(torch.nn.Module):
        def forward(self, x):
            return x

    contract = _capture(ReturnInput(), (torch.zeros(2),))
    output_spec = SampleMetadata.from_outputs(torch.zeros(2))
    input_tensor = torch.ones(2)

    result = contract.restore_outputs((torch.full((2,), 7.0),), {"x": input_tensor}, output_spec)

    assert result is input_tensor
