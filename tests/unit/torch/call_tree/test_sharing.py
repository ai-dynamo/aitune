# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``CallTree.shared_weights_of`` on recorded toy models."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from aitune.torch.call_tree.sharing import shared_weight_groups
from tests.toy_models.call_tree_models import (
    DelegatingGlueToy,
    GlueToy,
    SequentialHeadToy,
    SequentialSubclassToy,
    VlaToy,
    vla_inputs,
)
from tests.unit.torch.call_tree.helpers import record


class _Root(torch.nn.Module):
    """A root holding the unit under test and a sibling, with typed attributes."""

    def __init__(self, unit: torch.nn.Linear, other: torch.nn.Module | None = None):
        super().__init__()
        self.unit = unit
        self.other = other if other is not None else torch.nn.Identity()


def _groups(groups):
    return [(g.params, g.held_by, g.read_by, g.moved_by) for g in groups]


def _vla():
    model = VlaToy()
    tokens, image = vla_inputs()
    return model, record(model, lambda: model(tokens, image))


def test_tied_head_shares_the_embedding_read_outside_it():
    model, tree = _vla()

    (group,) = tree.shared_weights_of(model.lm_head)

    assert (group.params, group.held_by) == (["llm.embed.weight"], ["lm_head"])
    assert group.read_by == ["forward > llm.embed.forward"]
    assert group.bytes == model.llm.embed.weight.numel() * model.llm.embed.weight.element_size()


def test_a_module_sharing_with_another_wrapped_module_lists_it_as_a_reader():
    model, tree = _vla()

    (group,) = tree.shared_weights_of(model.llm)

    assert group.params == ["llm.embed.weight"]
    assert group.read_by == ["forward > llm.embed.forward", "forward > lm_head.forward"]


def test_a_unit_without_shared_weights_has_no_groups():
    model, tree = _vla()

    assert tree.shared_weights_of(model.dit) == []
    assert tree.shared_weights_of("forward > dit_forward") == []


def test_embedding_looked_up_outside_the_module():
    model = GlueToy()
    tokens = torch.randint(0, 16, (2,))
    tree = record(model, lambda: model(tokens))

    assert _groups(tree.shared_weights_of(model.prefix)) == [
        (["prefix.embed.weight"], ["prefix"], ["forward > prefix.embed.forward"], [])
    ]


def test_to_from_unrecorded_code_moves_the_whole_module():
    model = SequentialSubclassToy()
    x = torch.randn(2, 8)
    tree = record(model, lambda: model.encode(x))

    (group,) = tree.shared_weights_of(model)

    assert (group.held_by, group.read_by, group.moved_by) == (["root"], [], ["encode"])
    assert group.params == ["0"]  # the whole Sequential subclass at .0
    assert group.bytes == sum(p.numel() * p.element_size() for p in model.parameters()) > 0


def test_a_list_is_one_unit_named_by_its_parts():
    model, tree = _vla()

    (group,) = tree.shared_weights_of([model.llm, model.lm_head])

    assert group.held_by == ["llm + lm_head"]
    assert group.read_by == ["forward > llm.embed.forward"]


def test_groups_come_with_the_ids_of_their_parameters():
    model, tree = _vla()

    ((group, ids),) = shared_weight_groups(tree, model.lm_head)

    assert ids == {id(model.llm.embed.weight)}
    assert group.params == ["llm.embed.weight"]


def test_unknown_units_raise():
    model, tree = _vla()

    with pytest.raises(ValueError, match="No recorded call"):
        tree.shared_weights_of("forward > missing.forward")
    with pytest.raises(ValueError, match="has no recorded calls"):
        tree.shared_weights_of(torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="names no calls"):
        tree.shared_weights_of([])
    with pytest.raises(TypeError):
        tree.shared_weights_of([1])  # type: ignore[list-item]


def test_a_module_owns_all_its_parameters_even_when_forward_delegates_to_one_child():
    model = DelegatingGlueToy()
    tokens = torch.randint(0, 16, (2,))
    tree = record(model, lambda: model(tokens))

    assert _groups(tree.shared_weights_of(model.holder)) == [
        (["holder.embed.weight"], ["holder"], ["forward > holder.embed.forward"], [])
    ]


def test_an_exact_container_stands_for_the_outermost_calls_inside_it():
    model = SequentialHeadToy()
    tokens = torch.randint(0, 16, (2,))
    tree = record(model, lambda: model(tokens))

    assert _groups(tree.shared_weights_of(model.head)) == [
        (["embed.weight"], ["head"], ["forward > embed.forward"], [])
    ]


def test_direct_read_outside_any_module_call_in_the_inference_function():
    """F.linear(x, unit.head.weight) at inference level reads head.weight from outside."""
    model = GlueToy()
    tokens = torch.randint(0, 16, (2,))
    tree = record(model, lambda: (model(tokens), F.linear(torch.randn(2, 8), model.prefix.embed.weight))[0])

    groups = tree.shared_weights_of(model.prefix)

    assert [g.params for g in groups] and any("embed.weight" in p for g in groups for p in g.params)
    assert all(g.read_by for g in groups)


def test_read_inside_another_modules_forward_via_a_stored_reference():
    """A module reaching a non-submodule weight through a stored list is reported by its forward path."""
    unit = torch.nn.Linear(8, 8)

    class Other(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._refs = [unit]  # a list: unit is not a submodule

        def forward(self, x):
            return x * self._refs[0].weight.mean()

    model = _Root(unit, Other())
    x = torch.randn(2, 8)
    tree = record(model, lambda: (model.unit(x), model.other(x))[0])

    ((group, ids),) = shared_weight_groups(tree, unit)
    assert group.params == ["unit.weight"] and ids == {id(unit.weight)}
    assert group.read_by == ["other.forward"]


def test_caller_of_the_unit_reads_its_weight_directly():
    """An ancestor of the unit's node reading the weight counts as outside."""

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unit = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.unit(x) @ self.unit.weight  # direct read inside own forward

    model = Outer()

    tree = record(model, lambda: model(torch.randn(2, 8)))

    ((group, _),) = shared_weight_groups(tree, model.unit)
    assert group.params == ["unit.weight"] and group.read_by == ["forward"]


def test_no_over_reporting_single_param_read():
    """Only the actually-read param of a two-param module is reported."""
    unit = torch.nn.Linear(8, 8)
    other = torch.nn.Linear(8, 8)
    model = _Root(unit, other)
    tree = record(model, lambda: (model.unit(torch.randn(2, 8)), F.linear(torch.randn(2, 8), unit.weight))[0])

    groups = tree.shared_weights_of(unit)
    assert [g.params for g in groups] == [["unit.weight"]]


def _standalone_unit():
    """A model whose called unit owns weights nobody else reads (a sibling unit for contrast)."""
    return _Root(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)), torch.randn(2, 8)


def test_outside_device_lookup_reports_the_tensor_only():
    """next(unit.parameters()).device counts as a device access for that tensor alone."""
    model, x = _standalone_unit()

    def call():
        model.unit(x)
        return next(model.unit.parameters()).device

    tree = record(model, call)
    groups = tree.shared_weights_of(model.unit)

    assert len(groups) == 1
    assert groups[0].params == ["unit.weight"]
    assert groups[0].device_by == ["test_outside_device_lookup_reports_the_tensor_only.<locals>.call"]
    assert groups[0].read_by == [] and groups[0].moved_by == []


def test_dtype_and_shape_lookups_do_not_count():
    model, x = _standalone_unit()
    tree = record(model, lambda: (model.unit(x), model.unit.weight.dtype, model.unit.weight.shape)[0])

    assert tree.shared_weights_of(model.unit) == []


def test_read_inside_torch_compiled_code_is_found_via_inspect():
    """ait.inspect(call_tree=True) records reads inside torch.compile(fullgraph=True) glue code."""
    import aitune.torch as ait

    model, x = _standalone_unit()

    @torch.compile(fullgraph=True)
    def glue():
        return F.linear(x, model.unit.weight)

    def call(*_args):
        model.unit(x)
        return glue()

    info = ait.inspect(
        model,
        [(x,)],
        inference_function=call,
        number_of_iterations=2,
        warmup_iterations=1,
        min_depth=0,
        max_depth=8,
        call_tree=True,
    )

    assert info.call_tree is not None
    ((group, _),) = shared_weight_groups(info.call_tree, model.unit)
    assert group.params == ["unit.weight"]
    assert group.read_by == ["test_read_inside_torch_compiled_code_is_found_via_inspect.<locals>.glue"]


def test_to_from_outside_is_moved_not_read():
    """An outside .to() of the unit shows only in moved_by."""
    model = SequentialSubclassToy()
    x = torch.randn(2, 8)
    tree = record(model, lambda: model.encode(x))

    (group,) = tree.shared_weights_of(model)
    assert group.moved_by == ["encode"] and group.read_by == [] and group.device_by == []


def test_buffer_read_outside_is_reported():
    unit = torch.nn.Linear(8, 8)
    unit.register_buffer("scale", torch.ones(8))
    model = _Root(unit)
    tree = record(model, lambda: model.unit.get_buffer("scale").sum() + model.unit(torch.randn(2, 8)).sum())

    groups = tree.shared_weights_of(unit)
    assert [g.params for g in groups] == [["unit.scale"]]


class _ExtensionHead(torch.nn.Module):
    """Reads its weight in code torch functions do not see, like a C++ or CUDA extension kernel."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        with torch._C.DisableTorchFunction():
            return F.linear(x, self.weight)


def test_a_weight_an_outside_extension_kernel_reads_is_kept():
    unit = torch.nn.Linear(8, 8)
    model = _Root(unit, _ExtensionHead(unit.weight))
    tree = record(model, lambda: model.other(model.unit(torch.randn(2, 8))))

    ((group, ids),) = shared_weight_groups(tree, unit)
    assert group.params == ["unit.weight"] and ids == {id(unit.weight)}
    assert group.read_by == ["other.forward"]


def test_a_sparse_buffer_read_outside_is_tracked_by_identity():
    unit = torch.nn.Linear(8, 8)
    unit.register_buffer("mask", torch.eye(8).to_sparse())
    model = _Root(unit)

    def infer():
        return torch.sparse.mm(model.unit.get_buffer("mask"), model.unit(torch.randn(8, 8)))

    tree = record(model, infer)

    ((group, _),) = shared_weight_groups(tree, unit)
    assert group.params == ["unit.mask"]
    assert group.read_by == ["test_a_sparse_buffer_read_outside_is_tracked_by_identity.<locals>.infer"]
