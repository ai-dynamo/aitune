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


class TensorExportAdapter(nn.Module):
    """Adapt a structured module call to flat tensor inputs and outputs in recorded order.

    Non-tensor arguments come from the recorded sample. Compatible inputs use a prepared PyTree
    recipe to keep reconstruction simple for strict Torch Export. Otherwise, recorded locators
    reconstruct the arguments, including custom objects supported by Locator but unknown to PyTree.
    This fallback preserves reconstruction support; export still depends on the module and exporter.
    """

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
        self._argument_tree_spec, self._argument_leaf_template, self._tensor_leaf_indices = reconstruction

    def forward(self, *flat_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the module with rebuilt inputs and return flat tensor outputs."""
        arguments = self._reconstruct_arguments(flat_inputs)
        args, kwargs = self._call_layout.restore(arguments)
        output = self.module(*args, **kwargs)
        return self._output_spec.flatten_tensors(output)

    def _reconstruct_arguments(self, flat_inputs: tuple[torch.Tensor, ...]) -> dict[str, Any]:
        """Rebuild normalized arguments with fresh containers for each call.

        A None TreeSpec means the PyTree compatibility checks failed. The locator fallback deep-copies
        the template and inserts tensors at recorded paths, preserving support for custom input objects
        without requiring global PyTree registration. The compatible path rebuilds PyTree containers
        around the saved non-tensor leaves and current tensor inputs.
        """
        if self._argument_tree_spec is None:
            return self._input_spec.unflatten_tensors(flat_inputs, self._arguments_template)

        argument_leaves = list(self._argument_leaf_template)
        for index, tensor in zip(self._tensor_leaf_indices, flat_inputs, strict=True):
            argument_leaves[index] = tensor
        return _pytree.tree_unflatten(argument_leaves, self._argument_tree_spec)


def _prepare_argument_reconstruction(
    arguments: dict[str, Any],
    template: dict[str, Any],
    input_spec: SampleMetadata,
) -> tuple[_pytree.TreeSpec | None, tuple[Any, ...], tuple[int, ...]]:
    """Prepare a PyTree recipe when it agrees with the recorded tensor paths.

    Tensor order and identity must match, and the template must have the same tree structure,
    with tensors replaced by None and non-tensor leaves retaining their identity. This lets forward
    rebuild arguments without discovering their structure or deep-copying the template while tracing.

    Return a None TreeSpec on mismatch so the adapter uses locator reconstruction instead.
    For example, Locator may traverse a custom object that PyTree treats as an opaque leaf.
    """
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
