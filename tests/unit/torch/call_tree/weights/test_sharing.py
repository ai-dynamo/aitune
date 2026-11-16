# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``CallTree.shared_weights_of`` on recorded toy models."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from aitune.torch.call_tree import shared_weights_of
from aitune.torch.call_tree.weights.sharing import shared_weights_with_ids
from tests.toy_models.call_tree_models import (
    DelegatingGlueToy,
    GlueToy,
    SequentialHeadToy,
    SequentialSubclassToy,
    VlaToy,
    vla_inputs,
)
from tests.unit.torch.call_tree.helpers import invisible_to_recorder, record


class _Root(torch.nn.Module):
    """A root module holding the unit under test as ``unit`` and an optional sibling as ``other``."""

    def __init__(self, unit: torch.nn.Linear, other: torch.nn.Module | None = None):
        super().__init__()
        self.unit = unit
        self.other = other if other is not None else torch.nn.Identity()


class _StoredRefReader(torch.nn.Module):
    """A module that reaches a weight it does not own through a stored list, and uses it in ``forward``."""

    def __init__(self, unit: torch.nn.Linear):
        super().__init__()
        self._refs = [unit]  # a list: unit is not a submodule

    def forward(self, x):
        return x * self._refs[0].weight.mean()


class _ExtensionHead(torch.nn.Module):
    """Reads its weight in code torch functions do not see, like a C++ or CUDA extension kernel."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        with invisible_to_recorder():
            return F.linear(x, self.weight)


class _ExtensionHeadWithBias(torch.nn.Module):
    """Reads a borrowed weight invisibly like ``_ExtensionHead``, and adds its own bias with a visible op."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight
        self.bias = torch.nn.Parameter(torch.zeros(8))

    def forward(self, x):
        with invisible_to_recorder():
            y = F.linear(x, self.weight)
        return y + self.bias


def test_tied_head_shares_the_embedding_read_outside_it():
    """``lm_head`` is tied to the embedding, so the embedding's call reads ``lm_head``'s weight from outside."""
    model = VlaToy()
    tokens, image = vla_inputs()
    tree = record(model, lambda: model(tokens, image))

    (weight,) = shared_weights_of(tree, model.lm_head)

    assert weight.name == "llm.embed.weight"
    assert weight.aliases == ["lm_head.weight"]
    assert weight.accesses == [("forward > llm.embed.forward", "read")]
    assert weight.bytes == model.llm.embed.weight.numel() * model.llm.embed.weight.element_size()


def test_a_module_sharing_with_another_wrapped_module_lists_it_as_a_reader():
    """The root runs ``llm.embed`` outside ``llm.forward``, and the tied ``lm_head`` reads it too: both are readers."""
    model = VlaToy()
    tokens, image = vla_inputs()
    tree = record(model, lambda: model(tokens, image))

    (weight,) = shared_weights_of(tree, model.llm)

    assert weight.name == "llm.embed.weight"
    assert weight.accesses == [("forward > llm.embed.forward", "read"), ("forward > lm_head.forward", "read")]


def test_a_list_is_one_unit_named_by_its_parts():
    """Passing several modules as one unit analyses them jointly."""
    model = VlaToy()
    tokens, image = vla_inputs()
    tree = record(model, lambda: model(tokens, image))

    (weight,) = shared_weights_of(tree, [model.llm, model.lm_head])

    assert weight.name == "llm.embed.weight"
    assert weight.accesses == [("forward > llm.embed.forward", "read")]


def test_embedding_looked_up_outside_the_module():
    """GlueToy calls the prefix's embedding before the prefix itself, so the embed call is an outside read."""
    model = GlueToy()
    tokens = torch.randint(0, 16, (2,))
    tree = record(model, lambda: model(tokens))

    (weight,) = shared_weights_of(tree, model.prefix)

    assert weight.name == "prefix.embed.weight"
    assert weight.accesses == [("forward > prefix.embed.forward", "read")]


def test_a_forward_path_owns_the_weights_of_children_it_does_not_run():
    """``holder.forward`` runs only ``body``, but offloading it moves ``embed`` too."""
    model = DelegatingGlueToy()
    tokens = torch.randint(0, 16, (2,))
    tree = record(model, lambda: model(tokens))

    (by_module,) = shared_weights_of(tree, model.holder)
    (by_path,) = shared_weights_of(tree, "forward > holder.forward")

    assert by_path.name == by_module.name == "holder.embed.weight"
    assert by_path.accesses == by_module.accesses == [("forward > holder.embed.forward", "read")]


def test_an_exact_container_stands_for_the_outermost_calls_inside_it():
    """An exact ``nn.Sequential`` has no recorded call of its own, so the head is the calls of its children."""
    model = SequentialHeadToy()
    tokens = torch.randint(0, 16, (2,))
    tree = record(model, lambda: model(tokens))

    (weight,) = shared_weights_of(tree, model.head)

    assert weight.name == "embed.weight"
    assert weight.accesses == [("forward > embed.forward", "read")]


def test_read_inside_another_modules_forward_via_a_stored_reference():
    """A module that keeps the unit in a plain list and reads its weight is reported as the reader."""
    unit = torch.nn.Linear(8, 8)
    model = _Root(unit, _StoredRefReader(unit))
    x = torch.randn(2, 8)

    def infer():
        unit(x)
        model.other(x)

    tree = record(model, infer)

    ((weight, tensor_id),) = shared_weights_with_ids(tree, unit)
    assert weight.name == "unit.weight"
    assert weight.accesses == [("other.forward", "read")]
    assert tensor_id == id(unit.weight)


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

    (weight,) = shared_weights_of(tree, model.unit)
    assert weight.name == "unit.weight"
    assert weight.accesses == [("forward", "read")]


def test_direct_read_outside_any_module_call_in_the_inference_function():
    """The inference function reads the prefix's embedding weight directly, so it is reported as a reader."""
    model = GlueToy()

    def infer():
        model(torch.randint(0, 16, (2,)))
        F.linear(torch.randn(2, 8), model.prefix.embed.weight)

    tree = record(model, infer)

    (weight,) = shared_weights_of(tree, model.prefix)

    assert weight.name == "prefix.embed.weight"
    assert weight.accesses == [("forward > prefix.embed.forward", "read"), (infer.__qualname__, "read")]


def test_only_the_parameter_read_outside_is_reported():
    """Code outside the unit reads only its weight, so the weight is reported and the bias is not."""
    unit = torch.nn.Linear(8, 8)
    model = _Root(unit)

    def infer():
        unit(torch.randn(2, 8))
        F.linear(torch.randn(2, 8), unit.weight)

    tree = record(model, infer)

    (weight,) = shared_weights_of(tree, unit)
    assert weight.name == "unit.weight"


def test_each_weight_lists_its_own_readers():
    """Each weight reports the outside code that reads it: the bias has its own reader, weight and scale another."""
    unit = torch.nn.Linear(8, 8)
    unit.register_buffer("scale", torch.ones(8))
    model = _Root(unit)

    def read_weight_and_scale():
        unit.weight.sum()
        unit.get_buffer("scale").sum()

    def read_bias():
        unit.bias.sum()

    def infer():
        unit(torch.randn(2, 8))
        read_weight_and_scale()
        read_bias()

    tree = record(model, infer)

    (bias, scale, weight) = shared_weights_of(tree, unit)

    assert bias.name == "unit.bias"
    assert bias.aliases == []
    assert bias.bytes == 8 * 4
    assert bias.accesses == [(read_bias.__qualname__, "read")]

    assert scale.name == "unit.scale"
    assert scale.aliases == []
    assert scale.bytes == 8 * 4
    assert scale.accesses == [(read_weight_and_scale.__qualname__, "read")]

    assert weight.name == "unit.weight"
    assert weight.aliases == []
    assert weight.bytes == 8 * 8 * 4
    assert weight.accesses == [(read_weight_and_scale.__qualname__, "read")]


def test_a_method_unit_does_not_report_its_own_forward_as_reader():
    """A method unit owns its module's weights, so the module's ``forward`` calling it is not an outside reader."""

    class Unit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(8))

        def forward(self, x):
            return self.step(x)

        def step(self, x):
            return x * self.w

    unit = Unit()
    tree = record(unit, lambda: unit(torch.randn(2, 8)), iterations=1)
    assert shared_weights_of(tree, "forward > step") == []


def test_a_registered_module_that_never_ran_reports_its_outside_readers():
    """A submodule that never ran is still analysed, and every read of its weight counts as outside."""
    unit = torch.nn.Linear(8, 8)
    model = _Root(unit, _StoredRefReader(unit))
    x = torch.randn(2, 8)
    tree = record(model, lambda: model.other(x))

    ((weight, tensor_id),) = shared_weights_with_ids(tree, unit)
    assert weight.name == "unit.weight"
    assert weight.accesses == [("other.forward", "read")]
    assert tensor_id == id(unit.weight)


def test_buffer_read_outside_is_reported():
    """A buffer read outside the unit is reported, the same as a parameter."""
    unit = torch.nn.Linear(8, 8)
    unit.register_buffer("scale", torch.ones(8))
    model = _Root(unit)

    def infer():
        unit.get_buffer("scale").sum()
        unit(torch.randn(2, 8)).sum()

    tree = record(model, infer)

    (weight,) = shared_weights_of(tree, unit)
    assert weight.name == "unit.scale"


def test_a_sparse_buffer_read_outside_is_tracked_by_identity():
    """A sparse buffer has no single storage, so it is tracked by object identity; an outside read is reported."""
    unit = torch.nn.Linear(8, 8)
    unit.register_buffer("mask", torch.eye(8).to_sparse())
    model = _Root(unit)

    def infer():
        return torch.sparse.mm(unit.get_buffer("mask"), unit(torch.randn(8, 8)))

    tree = record(model, infer)

    (weight,) = shared_weights_of(tree, unit)
    assert weight.name == "unit.mask"
    assert weight.accesses == [(infer.__qualname__, "read")]


def test_read_inside_torch_compiled_code_is_found_via_inspect():
    """``ait.inspect(call_tree=True)`` runs compiled glue code eagerly, so a read inside it is reported."""
    import aitune.torch as ait

    unit = torch.nn.Linear(8, 8)
    model = _Root(unit, torch.nn.Linear(8, 8))
    x = torch.randn(2, 8)

    @torch.compile(fullgraph=True)
    def glue():
        return F.linear(x, unit.weight)

    def call(*_args):
        unit(x)
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
    (weight,) = shared_weights_of(info.call_tree, unit)
    assert weight.name == "unit.weight"
    assert weight.accesses == [(glue.__qualname__, "read")]


def test_a_unit_without_shared_weights_reports_nothing():
    """A module whose weights nobody outside reads or moves reports an empty list."""
    model = VlaToy()
    tokens, image = vla_inputs()
    tree = record(model, lambda: model(tokens, image))

    assert shared_weights_of(tree, model.dit) == []
    assert shared_weights_of(tree, "forward > dit_forward") == []


def test_each_weight_comes_with_the_id_of_its_tensor():
    """``shared_weights_with_ids`` pairs each entry with the live tensor id of its weight."""
    model = VlaToy()
    tokens, image = vla_inputs()
    tree = record(model, lambda: model(tokens, image))

    ((weight, tensor_id),) = shared_weights_with_ids(tree, model.lm_head)

    assert tensor_id == id(model.llm.embed.weight)
    assert weight.name == "llm.embed.weight"


def test_outside_device_lookup_reports_the_tensor_only():
    """Asking for a parameter's device is reported as a ``"device"`` access for that parameter only, not a read."""
    unit = torch.nn.Linear(8, 8)
    model = _Root(unit, torch.nn.Linear(8, 8))
    x = torch.randn(2, 8)

    def call():
        unit(x)
        return next(unit.parameters()).device

    tree = record(model, call)

    (weight,) = shared_weights_of(tree, unit)
    assert weight.name == "unit.weight"
    assert weight.accesses == [(call.__qualname__, "device")]


@pytest.mark.parametrize(
    "lookup",
    [
        pytest.param(lambda w: w.device, id="device"),
        pytest.param(lambda w: w.is_cuda, id="is_cuda"),
        pytest.param(lambda w: w.get_device(), id="get_device"),
    ],
)
def test_device_lookups_are_device_accesses(lookup):
    """``.device``, ``is_cuda`` and ``get_device()`` on a weight all count as device lookups, not reads."""
    unit = torch.nn.Linear(8, 8)
    model = _Root(unit, torch.nn.Linear(8, 8))
    x = torch.randn(2, 8)

    def infer():
        unit(x)
        lookup(unit.weight)

    tree = record(model, infer)

    (weight,) = shared_weights_of(tree, unit)
    assert weight.name == "unit.weight"
    assert weight.accesses == [(lookup.__qualname__, "device")]


def test_a_device_lookup_inside_another_modules_forward_reports_that_call():
    """A device lookup inside another module's ``forward`` is reported as a ``"device"`` access by that call."""

    class DeviceReader(torch.nn.Module):
        def __init__(self, unit: torch.nn.Linear):
            super().__init__()
            self._refs = [unit]  # a list: unit is not a submodule

        def forward(self, x):
            _ = self._refs[0].weight.device
            return x

    unit = torch.nn.Linear(8, 8)
    model = _Root(unit, DeviceReader(unit))
    x = torch.randn(2, 8)
    tree = record(model, lambda: model.other(x))

    (weight,) = shared_weights_of(tree, unit)
    assert weight.name == "unit.weight"
    assert weight.accesses == [("other.forward", "device")]


def test_a_device_lookup_inside_the_unit_is_not_reported():
    """A device lookup inside the unit's own ``forward`` is inside the unit, so nothing is reported."""

    class Unit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(8))

        def forward(self, x):
            _ = self.w.device
            return x * self.w

    unit = Unit()
    tree = record(unit, lambda: unit(torch.randn(2, 8)), iterations=1)
    assert shared_weights_of(tree, unit) == []


def test_dtype_and_shape_lookups_do_not_count():
    """Reading ``dtype`` or ``shape`` is not reported, because an offloaded weight still answers them."""
    unit = torch.nn.Linear(8, 8)
    model = _Root(unit, torch.nn.Linear(8, 8))
    x = torch.randn(2, 8)

    def infer():
        unit(x)
        _ = unit.weight.dtype, unit.weight.shape

    tree = record(model, infer)

    assert shared_weights_of(tree, unit) == []


def test_an_outside_to_moves_the_whole_module_without_reading_it():
    """``encode`` calls ``.to()`` on the model from outside its ``forward``: reported as a move, not a read."""
    model = SequentialSubclassToy()
    x = torch.randn(2, 8)
    tree = record(model, lambda: model.encode(x))

    weights = shared_weights_of(tree, model)

    all_names = sorted(name for named in (model.named_parameters(), model.named_buffers()) for name, _ in named)
    assert [w.name for w in weights] == all_names
    for weight in weights:
        assert weight.accesses == [("encode", "move")]


def test_buffer_only_unit_moved_from_outside_is_reported():
    """Moving a module whose only weight is a buffer is reported as a ``"move"`` access."""

    class Scale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.ones(8))

        def forward(self, x):
            return x * self.get_buffer("scale")

    unit = Scale()

    def infer():
        unit(torch.randn(2, 8))
        unit.double()

    tree = record(unit, infer, iterations=1)
    (weight,) = shared_weights_of(tree, unit)
    assert weight.name == "scale"
    assert weight.accesses == [(infer.__qualname__, "move")]


def test_a_mover_called_both_inside_and_outside_the_unit_is_reported():
    """``mover.move`` runs both inside and outside the unit; the outside call is still reported as a mover."""

    class Mover(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._victim = []  # set later; a list so the victim stays owned by ``outer``

        def move(self):
            self._victim[0].float()

        def forward(self, x):
            return x

    class Outer(torch.nn.Module):
        def __init__(self, mover):
            super().__init__()
            self.victim = torch.nn.Linear(4, 4)
            self._mover = [mover]

        def forward(self, x):
            self._mover[0].move()
            return self.victim(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mover = Mover()
            self.outer = Outer(self.mover)
            self.mover._victim = [self.outer.victim]

        def forward(self, x):
            self.mover.move()  # the same ``mover.move`` label, outside the unit
            return self.outer(x)

    model = Model()
    x = torch.randn(2, 4)
    tree = record(model, lambda: model(x))

    (bias, weight) = shared_weights_of(tree, "forward > outer.forward")

    assert bias.name == "outer.victim.bias"
    assert bias.accesses == [("forward > mover.move", "move")]
    assert weight.name == "outer.victim.weight"
    assert weight.accesses == [("forward > mover.move", "move")]


def test_buffer_read_before_a_cast_during_recording_is_reported():
    """A cast during recording replaces the buffer object; the read made before the cast is still reported."""
    unit = torch.nn.Linear(8, 8)
    unit.register_buffer("scale", torch.ones(8))
    model = _Root(unit)

    def infer():
        unit.get_buffer("scale").sum()
        model.double()
        unit(torch.randn(2, 8).double()).sum()

    tree = record(model, infer, iterations=1)

    (bias, scale, weight) = shared_weights_of(tree, unit)

    assert bias.name == "unit.bias"
    assert bias.accesses == [(infer.__qualname__, "move")]
    assert scale.name == "unit.scale"
    assert scale.accesses == [(infer.__qualname__, "move"), (infer.__qualname__, "read")]
    assert weight.name == "unit.weight"
    assert weight.accesses == [(infer.__qualname__, "move")]


def test_buffer_read_is_reported_after_a_post_recording_cast():
    """After a cast between recording and the query, the entry reports the new buffer object's id."""
    unit = torch.nn.Linear(8, 8)
    unit.register_buffer("scale", torch.ones(8))
    model = _Root(unit)

    def infer():
        unit.get_buffer("scale").sum()
        unit(torch.randn(2, 8)).sum()

    tree = record(model, infer)

    model.double()

    ((weight, tensor_id),) = shared_weights_with_ids(tree, unit)
    assert weight.name == "unit.scale"
    assert weight.accesses == [(infer.__qualname__, "read")]
    assert tensor_id == id(unit.get_buffer("scale"))


def test_a_parameter_given_new_storage_without_apply_is_still_reported():
    """After ``param.data = ...`` gives the weight new storage, a later outside read is still reported."""
    unit = torch.nn.Linear(8, 8)
    model = _Root(unit)

    def infer():
        unit(torch.randn(2, 8))
        with invisible_to_recorder():
            unit.weight.data = unit.weight.data.clone()
        F.linear(torch.randn(2, 8), unit.weight)

    tree = record(model, infer, iterations=1)
    ((weight, tensor_id),) = shared_weights_with_ids(tree, unit)
    assert weight.name == "unit.weight"
    assert weight.accesses == [(infer.__qualname__, "read")]
    assert tensor_id == id(unit.weight)


def test_a_parameter_replaced_on_its_owner_is_still_reported():
    """After a new ``nn.Parameter`` replaces the weight on its owner module, as accelerate does, a read is still reported."""
    unit = torch.nn.Linear(8, 8)
    model = _Root(unit)

    def infer():
        unit(torch.randn(2, 8))
        with invisible_to_recorder():
            unit._parameters["weight"] = torch.nn.Parameter(unit.weight.detach().clone())
        F.linear(torch.randn(2, 8), unit.weight)

    tree = record(model, infer, iterations=1)
    ((weight, tensor_id),) = shared_weights_with_ids(tree, unit)
    assert weight.name == "unit.weight"
    assert weight.accesses == [(infer.__qualname__, "read")]
    assert tensor_id == id(unit.weight)


def test_a_weight_an_outside_extension_kernel_reads_is_kept():
    """A module holding the unit's weight and reading it in code the recorder can't see is still a reader."""
    unit = torch.nn.Linear(8, 8)
    model = _Root(unit, _ExtensionHead(unit.weight))
    tree = record(model, lambda: model.other(unit(torch.randn(2, 8))))

    ((weight, tensor_id),) = shared_weights_with_ids(tree, unit)
    assert weight.name == "unit.weight"
    assert weight.accesses == [("other.forward", "read")]
    assert tensor_id == id(unit.weight)


def test_a_weight_an_outside_extension_kernel_reads_alongside_visible_ops_is_kept():
    """The invisible read is still reported when the same module also runs ops the recorder can see."""
    unit = torch.nn.Linear(8, 8)
    model = _Root(unit, _ExtensionHeadWithBias(unit.weight))
    tree = record(model, lambda: model.other(unit(torch.randn(2, 8))))

    (weight,) = shared_weights_of(tree, unit)
    assert weight.name == "unit.weight"
    assert weight.accesses == [("other.forward", "read")]


def test_views_of_one_storage_each_count_their_own_bytes():
    """Each parameter viewing one storage reports its own ``numel * element_size`` bytes."""

    class TwoViews(torch.nn.Module):
        def __init__(self):
            super().__init__()
            backing = torch.randn(8, 8)
            self.a = torch.nn.Parameter(backing[:6])
            self.b = torch.nn.Parameter(backing[2:])

        def forward(self, x):
            return x

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.views = TwoViews()

        def forward(self, x):
            return x

    model = Model()

    def infer():
        model.views.a.sum()
        model.views.b.sum()

    tree = record(model, infer)

    (a, b) = shared_weights_of(tree, model.views)
    assert a.name == "views.a"
    assert a.bytes == 6 * 8 * 4
    assert a.accesses == [(infer.__qualname__, "read")]
    assert b.name == "views.b"
    assert b.bytes == 6 * 8 * 4
    assert b.accesses == [(infer.__qualname__, "read")]


def test_a_tied_weight_is_listed_once_with_its_alias():
    """A weight tied between the unit and another module is one entry, named by its first name, alias listed."""
    model = SequentialHeadToy()
    tokens = torch.randint(0, 16, (2,))
    tree = record(model, lambda: model(tokens))

    ((weight, tensor_id),) = shared_weights_with_ids(tree, model.head)
    assert tensor_id == id(model.embed.weight)
    assert weight.name == "embed.weight"
    assert weight.aliases == ["head.0.weight"]
    assert weight.bytes == model.embed.weight.numel() * model.embed.weight.element_size()


@pytest.mark.parametrize(
    ("unit", "error", "match"),
    [
        pytest.param("forward > missing.forward", ValueError, "No recorded call", id="unrecorded-path"),
        pytest.param(torch.nn.Linear(1, 1), ValueError, "not part of the recorded model", id="foreign-module"),
        pytest.param([], ValueError, "names no calls", id="empty-list"),
        pytest.param([1], TypeError, None, id="non-module-item"),  # type: ignore[list-item]
    ],
)
def test_unknown_units_raise(unit, error, match):
    """Units that name nothing the recording knows are rejected with a descriptive error."""
    model = VlaToy()
    tokens, image = vla_inputs()
    tree = record(model, lambda: model(tokens, image))

    with pytest.raises(error, match=match):
        shared_weights_of(tree, unit)
