# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record the Python behavior that must surround a flat compiled module call.

For example, a module can update a cache without returning it::

    def forward(x, state):
        state["cache"] = state["cache"] + 1
        return {"value": x + state["cache"]}

The compiled interface returns (value, updated_cache). ForwardEffectsSpec tells the backend to assign
updated_cache to the caller's state and return {"value": value}. If forward also returns state,
restoration reuses the caller's original container.

Recording captures the specification once per graph and saves it with the graph for checkpoint reload.
It describes observed leaf changes, not arbitrary Python code.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from aitune.torch.module.locator import Locator
from aitune.torch.module.sample_metadata import SampleMetadata


class InputMutationKind(str, Enum):
    """How a recorded input leaf changed during forward.

    COPY_TENSOR means the same tensor was written to in place with unchanged metadata. The name refers
    to default restoration: copy the compiled result back into the existing input tensor. It does not
    mean the recorded forward call copied the tensor; a backend policy can also select replacement.
    REPLACE_TENSOR means a new tensor appeared or tensor metadata changed, requiring assignment rather
    than value copying.
    REPLACE_CONSTANT stores an observed non-tensor value; it does not evaluate an update on future inputs.
    """

    COPY_TENSOR = "copy_tensor"
    REPLACE_TENSOR = "replace_tensor"
    REPLACE_CONSTANT = "replace_constant"


class TensorMutationPolicy(str, Enum):
    """How a backend result should update a changed input tensor.

    PRESERVE_INPUT copies COPY_TENSOR results into the existing tensor. REPLACE_INPUT installs the
    backend's result at the input path instead; retained references to the old tensor may stay unchanged.
    REPLACE_TENSOR mutations always assign the result, regardless of this policy.

    Mutation kind describes the recorded model behavior; this policy selects backend restoration.
    AOT Inductor and Torch-TensorRT AOT keep the default PRESERVE_INPUT so references to an in-place
    mutation target remain valid. ONNX Runtime and standalone TensorRT choose REPLACE_INPUT to use
    returned buffers as the current state rather than copy values back into old leaves. That choice
    preserves the containing object, but provides weaker tensor identity guarantees. Output-buffer
    ownership alone does not require replacement.

    For ``old_cache = state["cache"]``, preservation leaves ``state["cache"] is old_cache`` true
    and updates its values. Replacement installs the result at ``state["cache"]``; old_cache may
    still refer to the previous tensor. Backends must document this choice at their restoration entry point.
    """

    PRESERVE_INPUT = "preserve_input"
    REPLACE_INPUT = "replace_input"


@dataclass(frozen=True)
class InputMutation:
    """One observable input change made by a module call.

    Attributes:
        locator: Path into normalized arguments where restoration applies the change.
        kind: Whether to copy tensor values, replace a tensor, or assign a recorded non-tensor value.
        output_index: Index in user tensor outputs followed by extra mutation outputs; None for constants.
        value: Recorded replacement for a non-tensor leaf, copied on restoration; unused for tensor changes.
    """

    locator: Locator
    kind: InputMutationKind
    output_index: int | None = None
    value: Any = None


@dataclass(frozen=True)
class OutputInputAlias:
    """An output path that returns the same object as an input path.

    Restoration reads input_locator from the current arguments and inserts that object at output_locator.
    This records Python identity, not storage sharing between distinct tensor views. A container alias
    covers its descendants, so a returned cache is restored as one object rather than rebuilt leaf by leaf.
    """

    output_locator: Locator
    input_locator: Locator


@dataclass(frozen=True)
class ForwardEffectsSpec:
    """Recorded output structure and input changes for one graph.

    Example:
        If forward assigns ``cache[0] = x + 1`` and returns ``x * 2``, compiled execution returns
        ``(x * 2, x + 1)``. The mutation points to output index 1. Restoration writes that tensor
        into the current cache and returns only index 0. If forward already returns the changed tensor,
        its user output index is reused. If it returns the cache container, an alias preserves that identity.

    Assumptions:
        The recorded mutation paths and alias relationships must apply to later calls using this graph.
        Capture does not describe deleted leaves or arbitrary side effects on opaque objects. Non-tensor
        replacements are recorded values: capturing ``step[0] += 1`` from 0 stores 1, not an increment
        operation. Strict matching can reject different starting values but cannot make that update dynamic.

    Attributes:
        output_template: Python output structure with tensor and aliased-object placeholders.
        input_mutations: Input leaf changes and the compiled output indices or constants that restore them.
        output_input_aliases: Outputs that must reuse current input objects after their state is restored.
    """

    output_template: Any
    input_mutations: tuple[InputMutation, ...] = ()
    output_input_aliases: tuple[OutputInputAlias, ...] = ()

    def extra_tensor_mutations(self, user_output_count: int) -> tuple[InputMutation, ...]:
        """Return tensor changes represented by additional graph outputs.

        User tensor outputs occupy indices below user_output_count. Changes absent from those outputs
        occupy the remaining indices, in capture order, so an unreturned cache update reaches the caller.
        """
        return tuple(
            mutation
            for mutation in self.input_mutations
            if mutation.output_index is not None and mutation.output_index >= user_output_count
        )

    def restore_outputs(
        self,
        flat_outputs: Any,
        arguments: dict[str, Any],
        output_spec: SampleMetadata,
        tensor_mutation_policies: Mapping[int, TensorMutationPolicy] | None = None,
    ) -> Any:
        """Apply input changes and rebuild the structured module output.

        Restore mutations first so outputs that alias inputs see the current state. Only user outputs
        are reconstructed; extra mutation tensors remain internal. Fresh output containers come from
        the template, while aliased outputs reuse input objects.

        Args:
            flat_outputs: User tensor results followed by extra mutation results in recorded order.
                A single tensor is also accepted for a one-result interface.
            arguments: Live normalized arguments from this inference call. Recorded paths update their
                tensors and nested containers; replacing a top-level binding only changes this dictionary.
            output_spec: Metadata defining user tensor output paths and order, excluding extra results.
            tensor_mutation_policies: Optional policies keyed by flat output index. Defaults to copying
                COPY_TENSOR results into existing tensors. Backends can instead install their output buffers.

        Returns:
            The structured user output with recorded input/output identities preserved.
        """
        flat_outputs = (flat_outputs,) if isinstance(flat_outputs, torch.Tensor) else tuple(flat_outputs)
        user_output_count = len(output_spec.tensor_data)
        expected_output_count = user_output_count + len(self.extra_tensor_mutations(user_output_count))
        if len(flat_outputs) != expected_output_count:
            raise ValueError(f"Expected {expected_output_count} flat outputs, got {len(flat_outputs)}.")

        for mutation in self.input_mutations:
            # Mutation paths are below the normalized argument dictionary, whose identity stays unchanged.
            if mutation.kind == InputMutationKind.REPLACE_CONSTANT:
                mutation.locator.set_value(arguments, deepcopy(mutation.value))
                continue

            if mutation.output_index is None:
                raise RuntimeError(f"Tensor mutation at {mutation.locator} has no corresponding flat output.")
            value = flat_outputs[mutation.output_index]
            policy = (tensor_mutation_policies or {}).get(
                mutation.output_index,
                TensorMutationPolicy.PRESERVE_INPUT,
            )
            if mutation.kind == InputMutationKind.COPY_TENSOR and policy == TensorMutationPolicy.PRESERVE_INPUT:
                target = mutation.locator.get_value(arguments)
                if not isinstance(target, torch.Tensor):
                    raise TypeError(
                        f"Tensor mutation target {mutation.locator} resolved to {type(target).__name__}, not a tensor."
                    )
                if target is not value:
                    target.copy_(value)
            else:
                mutation.locator.set_value(arguments, value)

        return output_spec.unflatten_tensors(
            flat_outputs[:user_output_count],
            self.output_template,
            tuple(
                (alias.output_locator, alias.input_locator.get_value(arguments)) for alias in self.output_input_aliases
            ),
        )


@dataclass(frozen=True)
class _TensorMetadata:
    """Tensor properties that cannot be reproduced by copying values."""

    shape: tuple[int, ...]
    stride: tuple[int, ...] | None
    storage_offset: int | torch.SymInt | None
    layout: torch.layout
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _InputLeafSnapshot:
    """Pre-call evidence for one leaf, retaining the live value for identity comparison.

    Tensor metadata and a version number detect changes without saving tensor values. If no version
    counter is readable, tensor_value holds an independent pre-call tensor for exact value comparison.
    Non-tensor leaves retain their original reference; mutations inside opaque leaf objects are not copied.
    """

    value: Any
    tensor_version: int | None
    tensor_metadata: _TensorMetadata | None = None
    tensor_value: torch.Tensor | None = None


@dataclass(frozen=True)
class ForwardEffectsCapture:
    """Compare normalized inputs before and after one module call.

    RecordingModule calls before_call for a new graph, executes the original module, and passes the
    same live arguments to after_call. Capture observes that execution rather than running another call.

    Tensor identity and metadata distinguish replacement from value changes. Version counters detect
    writes to ordinary tensors. Tensors created inside torch.inference_mode() have no version counter,
    even when later inspected outside the context, so those tensors require saved values instead.
    Entering inference_mode alone does not remove counters from tensors created outside it.
    Snapshot allocation and value comparison add recording cost; restoration does not repeat capture.
    """

    _before: dict[Locator, _InputLeafSnapshot]

    @classmethod
    def before_call(
        cls,
        arguments: dict[str, Any],
        snapshot_arguments: dict[str, Any] | None = None,
    ) -> ForwardEffectsCapture:
        """Capture input leaves immediately before forward executes.

        Args:
            arguments: Live normalized inputs that forward will receive. Keep the original references
                so after_call can distinguish assignment from changes to an existing tensor.
            snapshot_arguments: Optional independent pre-call copy with the same recorded paths.
                RecordingModule already copies samples; reusing that copy avoids another tensor clone
                when a version counter is unavailable. It must remain unchanged until after_call.
                Without a usable snapshot tensor, capture clones the live tensor as needed.

        Returns:
            Pre-call evidence to compare against the live arguments after forward.
        """
        return cls({
            locator: _snapshot_leaf(locator, value, snapshot_arguments)
            for locator, value in Locator.find_leaves(arguments)
        })

    def after_call(
        self,
        arguments: dict[str, Any],
        output: Any,
        output_spec: SampleMetadata,
    ) -> ForwardEffectsSpec:
        """Capture output structure, input changes, and input/output aliases after forward.

        Output structure and aliases are captured even when no input leaf changed. Compiled execution
        still needs them to rebuild the Python output and reuse input objects returned by forward.

        Args:
            arguments: The same normalized argument dictionary captured before the call, now containing
                forward's changes. Only paths present after the call are considered; deletions are not recorded.
            output: Actual structured result from that call, used to find tensor results and input aliases.
            output_spec: Metadata for output, defining the user tensor order before mutation results are added.

        Returns:
            A specification containing the output template, input mutations, and input/output aliases.
            Tensor mutations reuse user tensor output indices where possible; changed tensors absent
            from those outputs receive subsequent indices so compiled execution can return their values
            solely to restore input state.
        """
        output_tensors = output_spec.flatten_tensors(output)
        output_input_aliases = _find_output_input_aliases(arguments, output)
        aliased_output_values = tuple(alias.output_locator.get_value(output) for alias in output_input_aliases)
        output_indices_by_identity = {id(tensor): index for index, tensor in enumerate(output_tensors)}
        next_output_index = len(output_tensors)
        mutations = []

        for locator, value in Locator.find_leaves(arguments):
            kind = _mutation_kind(self._before.get(locator), value)
            if kind is None:
                continue
            if kind == InputMutationKind.REPLACE_CONSTANT:
                mutations.append(InputMutation(locator, kind, value=deepcopy(value)))
                continue

            output_index = output_indices_by_identity.get(id(value))
            if output_index is None:
                output_index = next_output_index
                next_output_index += 1
            mutations.append(InputMutation(locator, kind, output_index=output_index))

        return ForwardEffectsSpec(
            output_template=output_spec.without_tensors(output, aliased_output_values),
            input_mutations=tuple(mutations),
            output_input_aliases=output_input_aliases,
        )


def _find_output_input_aliases(arguments: dict[str, Any], output: Any) -> tuple[OutputInputAlias, ...]:
    """Find output tensors and containers that are also input objects.

    Record an aliased output container once and skip its descendants: reusing that input container
    also preserves the identity of its contents. This is what selecting the largest aliased values
    means. Distinct tensor views that share storage are not treated as the same object.
    """
    input_nodes = {
        id(value): locator
        for locator, value in Locator.find_nodes(arguments)
        if isinstance(value, torch.Tensor) or Locator.is_traversable(value)
    }
    aliases = []
    for output_locator, value in Locator.find_nodes(output, is_output=True):
        if any(alias.output_locator.is_prefix_of(output_locator) for alias in aliases):
            continue
        input_locator = input_nodes.get(id(value))
        if input_locator is not None:
            aliases.append(OutputInputAlias(output_locator, input_locator))
    return tuple(aliases)


def _tensor_version(tensor: torch.Tensor) -> int | None:
    """Read a version counter, or return None when unavailable.

    In particular, tensors created inside torch.inference_mode() do not expose a counter. The caller
    then uses saved tensor values; the active execution context alone does not determine availability.
    """
    try:
        return tensor._version
    except Exception:
        return None


def _tensor_metadata(tensor: torch.Tensor) -> _TensorMetadata:
    """Read metadata used to distinguish copying from replacement.

    Some tensor layouts do not support stride or storage offset queries. If either query raises
    RuntimeError, store None for both fields so snapshots can still compare the remaining metadata.
    None marks unavailable metadata, not a zero stride or offset.
    """
    try:
        stride = tuple(tensor.stride())
        storage_offset = tensor.storage_offset()
    except RuntimeError:
        stride = None
        storage_offset = None
    return _TensorMetadata(
        shape=tuple(tensor.shape),
        stride=stride,
        storage_offset=storage_offset,
        layout=tensor.layout,
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _snapshot_leaf(
    locator: Locator,
    value: Any,
    snapshot_arguments: dict[str, Any] | None,
) -> _InputLeafSnapshot:
    """Save the identity and mutation evidence for one input leaf.

    Use a version counter when available. Otherwise reuse an independent tensor from snapshot_arguments,
    or clone the live tensor, because retaining its reference alone would also observe later writes.
    """
    if not isinstance(value, torch.Tensor):
        return _InputLeafSnapshot(value, None)

    version = _tensor_version(value)
    tensor_value = None
    if version is None:
        if snapshot_arguments is not None:
            candidate = locator.get_value(snapshot_arguments)
            tensor_value = candidate if isinstance(candidate, torch.Tensor) else None
        if tensor_value is None or tensor_value is value:
            with torch.no_grad():
                tensor_value = value.detach().clone()
    return _InputLeafSnapshot(
        value=value,
        tensor_version=version,
        tensor_metadata=_tensor_metadata(value),
        tensor_value=tensor_value,
    )


def _tensor_values_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensors exactly when a version counter is unavailable.

    Floating-point and complex comparisons use zero tolerance and treat matching NaNs as equal.
    Return False if comparison raises: equality could not be established, so mutation capture
    conservatively records a tensor write.
    """
    try:
        if left.dtype.is_floating_point or left.dtype.is_complex:
            return bool(torch.allclose(left, right, rtol=0, atol=0, equal_nan=True))
        return torch.equal(left, right)
    except Exception:
        return False


def _values_equal(left: Any, right: Any) -> bool:
    """Compare non-tensor leaves by identity, then by equality.

    Identical objects are accepted without inspecting their internal state. Otherwise, equality must
    return a Python bool. Exceptions and other result types, such as elementwise comparison arrays,
    return False so mutation capture conservatively records a replacement.
    """
    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        # Leaf types may reject comparison, for example NumPy arrays with incompatible shapes.
        # Record a replacement when equality cannot be established.
        return False
    return result if isinstance(result, bool) else False


def _mutation_kind(previous: _InputLeafSnapshot | None, value: Any) -> InputMutationKind | None:
    """Return the restoration kind for a detected input leaf change.

    Args:
        previous: Pre-call snapshot for this input path, or None if the path is new.
        value: Post-call value at the same input path.

    Returns:
        COPY_TENSOR when the same tensor retains its metadata but a version change or value
        comparison indicates a write. Also returned when unchanged values cannot be established
        because no saved tensor value is available or comparison fails.
        REPLACE_TENSOR for a new tensor object or changed tensor metadata.
        REPLACE_CONSTANT for a new or changed non-tensor value; restoration assigns the recorded
        value rather than replaying the operation that produced it.
        None when no change is detected and no mutation record is needed. This does not describe
        output structure or input/output aliases, which are captured separately.

    COPY_TENSOR describes the default restoration action, not an operation performed by forward.
    A backend's tensor mutation policy may select replacement instead of copying values back.
    """
    if isinstance(value, torch.Tensor):
        if previous is None or not isinstance(previous.value, torch.Tensor) or previous.value is not value:
            return InputMutationKind.REPLACE_TENSOR
        if previous.tensor_metadata != _tensor_metadata(value):
            return InputMutationKind.REPLACE_TENSOR
        current_version = _tensor_version(value)
        if previous.tensor_version is not None and current_version is not None:
            return InputMutationKind.COPY_TENSOR if previous.tensor_version != current_version else None
        if previous.tensor_value is None or not _tensor_values_equal(previous.tensor_value, value):
            return InputMutationKind.COPY_TENSOR
        return None
    if previous is None or not _values_equal(previous.value, value):
        return InputMutationKind.REPLACE_CONSTANT
    return None
