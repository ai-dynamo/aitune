# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrap structured module calls for exporters that accept flat tensors."""

from typing import Any

import torch
import torch.nn as nn
from torch.utils import _pytree

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.sample_store import Sample


def _prepare_argument_reconstruction(
    arguments: dict[str, Any],
    template: dict[str, Any],
    input_spec: SampleMetadata,
) -> tuple[_pytree.TreeSpec | None, tuple[Any, ...], tuple[int, ...]]:
    """Prepare a pytree recipe that strict Torch Export can trace."""
    leaves, tree_spec = _pytree.tree_flatten(arguments)
    template_leaves, template_tree_spec = _pytree.tree_flatten(template)
    flat_tensors = input_spec.flatten_tensors(arguments)
    tree_tensors = tuple(leaf for leaf in leaves if isinstance(leaf, torch.Tensor))

    tensors_match = len(tree_tensors) == len(flat_tensors) and all(
        tree_tensor is flat_tensor for tree_tensor, flat_tensor in zip(tree_tensors, flat_tensors, strict=True)
    )
    template_matches = (
        tree_spec == template_tree_spec
        and len(leaves) == len(template_leaves)
        and all(
            template_leaf is None if isinstance(leaf, torch.Tensor) else template_leaf is leaf
            for leaf, template_leaf in zip(leaves, template_leaves, strict=True)
        )
    )
    if not tensors_match or not template_matches:
        return None, (), ()

    tensor_leaf_indices = tuple(index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor))
    return tree_spec, tuple(template_leaves), tensor_leaf_indices


class TensorExportAdapter(nn.Module):
    """Expose a structured module call as flat tensor inputs and outputs."""

    def __init__(
        self,
        module: nn.Module,
        sample: Sample,
        graph_spec: GraphSpec,
    ):
        """Build reconstruction rules from one recorded module call."""
        super().__init__()
        self.module = module
        self.training = module.training
        forward_inputs = graph_spec.forward_signature.normalize(*sample)
        arguments = forward_inputs.arguments
        arguments_template = graph_spec.input_spec.without_tensors(arguments)
        reconstruction = _prepare_argument_reconstruction(arguments, arguments_template, graph_spec.input_spec)

        self._arguments_template = arguments_template
        self._call_layout = graph_spec.forward_signature.capture_call_layout(*sample)
        self._input_spec = graph_spec.input_spec
        self._output_spec = graph_spec.output_spec
        self._state_output_locators = tuple(mutation.locator for mutation in graph_spec.extra_tensor_mutations())
        self._argument_tree_spec, self._argument_leaf_template, self._tensor_leaf_indices = reconstruction

    def forward(self, *flat_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the module with rebuilt inputs and return flat tensor outputs."""
        arguments = self._reconstruct_arguments(flat_inputs)
        args, kwargs = self._call_layout.restore(arguments)
        output = self.module(*args, **kwargs)
        user_outputs = self._output_spec.flatten_tensors(output)
        state_outputs = tuple(locator.get_value(arguments) for locator in self._state_output_locators)
        return user_outputs + state_outputs

    def _reconstruct_arguments(self, flat_inputs: tuple[torch.Tensor, ...]) -> dict[str, Any]:
        """Rebuild normalized arguments without sharing mutable containers between calls."""
        if self._argument_tree_spec is None:
            return self._input_spec.unflatten_tensors(flat_inputs, self._arguments_template)

        argument_leaves = list(self._argument_leaf_template)
        for index, tensor in zip(self._tensor_leaf_indices, flat_inputs, strict=True):
            argument_leaves[index] = tensor
        return _pytree.tree_unflatten(argument_leaves, self._argument_tree_spec)
