# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save eager results to disk to check input changes, outside exported graphs."""

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from aitune.torch.distributed import coordinator
from aitune.torch.module.forward_contract import InputMutationKind, TensorMutationPolicy
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.locator import Locator
from aitune.torch.module.sample_store import Sample, SampleStore
from aitune.torch.task.correctness import CorrectnessInputMutationError


@contextmanager
def _preserve_buffers(module: torch.nn.Module):
    """Restore the model's buffers after recording expected results."""
    saved_buffers = []
    saved_values = {}
    for child in module.modules():
        saved_buffers.append((child, dict(child._buffers), set(child._non_persistent_buffers_set)))
        for buffer in child._buffers.values():
            if buffer is not None and id(buffer) not in saved_values:
                saved_values[id(buffer)] = (buffer, buffer.detach().clone())
    try:
        yield
    finally:
        with torch.no_grad():
            for original, saved in saved_values.values():
                if original.shape != saved.shape:
                    original.resize_(saved.shape)
                if not torch.equal(original, saved):
                    original.copy_(saved)
            for child, buffers, non_persistent in saved_buffers:
                child._buffers.clear()
                child._buffers.update(buffers)
                child._non_persistent_buffers_set.clear()
                child._non_persistent_buffers_set.update(non_persistent)


def _identity_nodes(arguments: Any) -> dict:
    """Keep references to tensors and containers so we can detect replacements."""
    return {
        locator: value
        for locator, value in Locator.find_nodes(arguments)
        if isinstance(value, torch.Tensor) or Locator.is_traversable(value)
    }


def _alias_groups(nodes: dict, *, storage: bool = False) -> dict:
    """Find paths that point to the same object or share tensor storage."""
    first_path = {}
    groups = {}
    for path, value in nodes.items():
        if storage:
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
                continue
            identity = (str(value.device), value.untyped_storage()._cdata)
        else:
            identity = id(value)
        groups[path] = first_path.setdefault(identity, path)
    return groups


def _snapshot(arguments: dict, output: Any, before: dict) -> dict:
    """Copy call results to CPU and record which objects share data."""
    values = {"inputs": arguments, "outputs": output}
    nodes = _identity_nodes(values)
    # Equal values are not enough: a copy must also keep the original input object.
    nodes.update({("before", path): value for path, value in before.items()})
    leaves = dict(Locator.find_leaves(values, include_mutation_fields=True))
    return {
        "leaves": {
            path: value.detach().to("cpu", copy=True) if isinstance(value, torch.Tensor) else deepcopy(value)
            for path, value in leaves.items()
        },
        "aliases": _alias_groups(nodes),
        "storage_aliases": _alias_groups(nodes, storage=True),
        "devices": {path: str(value.device) for path, value in leaves.items() if isinstance(value, torch.Tensor)},
    }


def _compare(expected: dict, actual: dict, name: str, replaced_inputs: tuple[Locator, ...] = ()) -> None:
    """Check values, shapes, dtypes, devices, and shared objects."""
    # Engine-owned state may replace recorded tensor leaves. Keep checking all
    # post-call aliases, including output-to-input aliases and shared containers.
    ignored = {("before", locator) for locator in replaced_inputs}
    for field in ("aliases", "storage_aliases", "devices"):
        expected_field = {path: value for path, value in expected[field].items() if path not in ignored}
        actual_field = {path: value for path, value in actual[field].items() if path not in ignored}
        if expected_field != actual_field:
            raise CorrectnessInputMutationError(f"{name}: {field} differ from eager execution")
    if expected["leaves"].keys() != actual["leaves"].keys():
        raise CorrectnessInputMutationError(f"{name}: input/output structure differs from eager execution")
    for path, value in expected["leaves"].items():
        other = actual["leaves"][path]
        try:
            if isinstance(value, torch.Tensor):
                atol = 1e-2 if value.dtype in (torch.float16, torch.bfloat16) else 1e-3
                torch.testing.assert_close(other, value, atol=atol, rtol=0, equal_nan=False)
            elif type(other) is not type(value) or other != value:
                raise AssertionError("non-tensor value differs")
        except (AssertionError, TypeError, ValueError) as error:
            raise CorrectnessInputMutationError(f"{name}: {path} differs from eager execution: {error}") from error


class StatefulValidation:
    """Compare repeated backend calls with the original model.

    Each sample starts a new series of calls that reuse the same input state. Expected results
    are saved before compilation and loaded one call at a time to limit memory use.
    These checks do not change the exported graph. They cover state passed as inputs,
    not hidden Python attributes inside the model.
    """

    def __init__(self, root: Path, graph_spec: GraphSpec, *, steps: int = 3):
        """Configure at least two consecutive calls per recorded sample."""
        if steps < 2:
            raise ValueError("Stateful validation requires at least two calls")
        self.root = root
        self.graph_spec = graph_spec
        self.steps = steps
        self.samples = SampleStore.create(root, "inputs")

    def record(self, module: torch.nn.Module, samples: Iterable[Sample]) -> None:
        """Record expected results without changing the caller's inputs, buffers, or random state."""
        with torch.no_grad():
            for index, sample in enumerate(samples):
                with coordinator.raise_if_any_rank_fails(f"Preparing stateful reference sample {index}"):
                    self.samples.append(sample)
                    args, kwargs = deepcopy(sample)
                devices = _cuda_devices(module, sample)
                with _preserve_buffers(module), torch.random.fork_rng(devices=devices):
                    self._record_sequence(module, args, kwargs, index)

    def _record_sequence(self, module: torch.nn.Module, args: tuple, kwargs: dict, index: int) -> None:
        """Save each result before the next call can change it."""
        for step in range(self.steps):
            with coordinator.raise_if_any_rank_fails(f"Stateful reference sample {index}, call {step + 1}"):
                arguments = self.graph_spec.forward_signature.normalize(args, kwargs).arguments
                before = _identity_nodes(arguments)
                output = module(*args, **kwargs)
                torch.save(_snapshot(arguments, output, before), self.root / f"expected-{index}-{step}.pt")

    def check(
        self,
        infer: Callable,
        device: torch.device | None,
        *,
        name: str,
        tensor_mutation_policy: TensorMutationPolicy = TensorMutationPolicy.PRESERVE_INPUT,
    ) -> None:
        """Compare each call with its saved eager results, including changes to input state."""
        replaced_inputs = tuple(
            mutation.locator
            for mutation in self.graph_spec.forward_contract.input_mutations
            if mutation.kind == InputMutationKind.COPY_TENSOR
            and tensor_mutation_policy == TensorMutationPolicy.REPLACE_INPUT
        )
        with torch.no_grad():
            for index, (args, kwargs) in enumerate(self.samples.iter_samples(device)):
                for step in range(self.steps):
                    label = f"{name}, sample {index}, call {step + 1}"
                    with coordinator.raise_if_any_rank_fails(label):
                        arguments = self.graph_spec.forward_signature.normalize(args, kwargs).arguments
                        before = _identity_nodes(arguments)
                        output = infer(*args, **kwargs)
                        expected = torch.load(self.root / f"expected-{index}-{step}.pt", weights_only=False)
                        _compare(expected, _snapshot(arguments, output, before), label, replaced_inputs)


def _cuda_devices(module: torch.nn.Module, sample: Sample) -> list[int]:
    """Find CUDA devices used by the model or its inputs."""
    tensors = list(module.parameters()) + list(module.buffers())
    tensors.extend(value for _, value in Locator.find_leaves(sample) if isinstance(value, torch.Tensor))
    return sorted({tensor.device.index for tensor in tensors if tensor.is_cuda})
