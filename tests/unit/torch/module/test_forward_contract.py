# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for recording forward input mutations."""

from copy import deepcopy

import torch

from aitune.torch.module.forward_contract import ForwardContractCapture, InputMutationKind
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


def test_capture_detects_inference_tensor_mutation_without_a_version_counter():
    class UpdateState(torch.nn.Module):
        def forward(self, x, state):
            state.add_(x)
            return x

    with torch.inference_mode():
        contract = _capture(UpdateState(), (torch.ones(2), torch.zeros(2)))

    assert [mutation.kind for mutation in contract.input_mutations] == [InputMutationKind.COPY_TENSOR]
