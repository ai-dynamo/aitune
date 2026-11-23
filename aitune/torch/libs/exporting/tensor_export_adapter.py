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

    Flattening is the explicit interface shared by export and backend execution. It separates
    tensor computation from Python containers, whose structure and identity must be restored outside
    the compiled runner. User tensor outputs come first, followed by changed input tensors that the
    forward effects specification needs for state restoration and that are absent from the user outputs.

    Non-tensor arguments come from the recorded sample. Compatible inputs use a prepared PyTree
    recipe to keep reconstruction simple for strict Torch Export. Otherwise, recorded locators
    reconstruct the arguments, including custom objects supported by Locator but unknown to PyTree.
    This fallback preserves reconstruction support; export still depends on the module and exporter.

    Example:
        For a module whose forward returns ``{"result": (x + state["bias"]) * state["scale"]}``,
        preparation from ``module(x=x, state={"bias": bias, "scale": 2})`` records this interface::

            Flat inputs:   (x, bias)
            Rebuilt call:  module(x=x, state={"bias": bias, "scale": 2})
            Flat outputs:  (result,)

        Each invocation supplies current tensors in the recorded order. The scale stays 2 because it
        comes from the sample, and the call stays keyword-based. This example has no input mutations.
        Backend restoration rebuilds the caller's Python output structure from the user tensor results.

    Mutation outputs:
        For ``cache[0] = x + 1; return x * 2`` with ``cache=[None]`` in the sample, flat inputs are
        ``(x,)`` and flat outputs are ``(x * 2, x + 1)``. Appending the updated cache tensor keeps an
        otherwise unreturned assignment visible outside the compiled graph. The backend applies it to
        the caller's cache before returning the user result. A changed tensor already returned by the
        module uses its existing output slot, so no extra result is needed for that mutation.

    Reconstruction rules:
        PyTree is used only when tensor identity and order agree with SampleMetadata and replacing
        tensors with None preserves the tree structure and non-tensor leaf identity. Matching only
        tensor counts could silently associate a flat input with the wrong argument. On mismatch,
        locators reconstruct from a copied template instead. Rebuilt containers isolate ordinary
        list and dict changes between calls; input tensors are references, so tensor writes can still
        affect supplied tensors. Custom-object reconstruction does not guarantee traceability.
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
        self._state_output_locators = tuple(mutation.locator for mutation in graph_spec.extra_tensor_mutations())
        self._argument_tree_spec, self._argument_leaf_template, self._tensor_leaf_indices = reconstruction

    def forward(self, *flat_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return user tensor outputs followed by the contract's extra mutation tensors.

        Extra results are read from reconstructed arguments after forward. They carry state changes
        to backend restoration even when the original module's return value omits that state.
        """
        arguments = self._reconstruct_arguments(flat_inputs)
        args, kwargs = self._call_layout.restore(arguments)
        output = self.module(*args, **kwargs)
        user_outputs = self._output_spec.flatten_tensors(output)
        # These locators are recorded exclusively for tensor mutations.
        state_outputs: tuple[Any, ...] = tuple(locator.get_value(arguments) for locator in self._state_output_locators)
        return user_outputs + state_outputs

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
