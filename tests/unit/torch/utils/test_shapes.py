# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aitune.torch.utils.shapes."""

import pytest
import torch
import torch.nn as nn

from aitune.torch.dynamic_shapes import BatchDim, DynamicDim
from aitune.torch.module.locator import Locator, ObjectType
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.utils.shapes import (
    _create_nested_structure,
    _raise_on_locator_user_type,
    axis_dims_for_tensor,
    build_dynamic_shapes,
    make_batch_dim,
    prepare_export_sample,
)
from tests.utilities.helpers import make_graph_spec, update_input_spec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_input(x):
    return x


def _two_inputs(x, y):
    return x, y


def _arg_and_x(arg, x):
    return arg, x


def _arg_and_options(arg, x, opt_none=None, opt_dict=None):
    return arg, x, opt_none, opt_dict


def _mask_input(mask):
    return mask


def _input_metadata(x: torch.Tensor, batch_size: int) -> SampleMetadata:
    return SampleMetadata.from_inputs({"x": x}, batch_size=batch_size)


# ---------------------------------------------------------------------------
# _create_nested_structure
# ---------------------------------------------------------------------------


def test_create_nested_structure_dict_list():
    locator = Locator((("zw", ObjectType.DICT), (0, ObjectType.SEQUENCE)))
    assert _create_nested_structure(locator, last={}) == {"zw": [{}]}


def test_create_nested_structure_three_levels():
    locator = Locator((("zw", ObjectType.DICT), (0, ObjectType.SEQUENCE), ("t", ObjectType.DICT)))
    assert _create_nested_structure(locator, last={}) == {"zw": [{"t": {}}]}


def test_create_nested_structure_extend_existing_list():
    locator = Locator((("zw", ObjectType.DICT), (0, ObjectType.SEQUENCE)))
    root = _create_nested_structure(locator, last={})
    locator2 = Locator((("zw", ObjectType.DICT), (1, ObjectType.SEQUENCE)))
    root = _create_nested_structure(locator2, root=root, last={})
    assert root == {"zw": [{}, {}]}


def test_create_nested_structure_scalar_leaf():
    locator = Locator((("zw", ObjectType.DICT),))
    assert _create_nested_structure(locator, last=1) == {"zw": 1}


# ---------------------------------------------------------------------------
# _raise_on_locator_user_type
# ---------------------------------------------------------------------------


def test_raise_on_locator_user_type_raises_for_user_type():
    locator = Locator((("x", ObjectType.USER_TYPE),))
    with pytest.raises(ValueError, match="user types"):
        _raise_on_locator_user_type(locator)


def test_raise_on_locator_user_type_raises_for_dataclass():
    locator = Locator((("x", ObjectType.DATACLASS),))
    with pytest.raises(ValueError):
        _raise_on_locator_user_type(locator)


def test_raise_on_locator_user_type_ok_for_dict_and_sequence():
    locator = Locator((("x", ObjectType.DICT), (0, ObjectType.SEQUENCE)))
    _raise_on_locator_user_type(locator)  # must not raise


# ---------------------------------------------------------------------------
# build_dynamic_shapes tests (no GPU required)
# ---------------------------------------------------------------------------


def test_build_dynamic_shapes_static():
    """Single sample → no shape updates → all static dims → empty inner dict per arg.

    The all-empty result lets the caller decide whether to pass ``None`` or the dict
    itself to ``torch.export.export`` (AOT backends post-process to ``None``; ONNX
    keeps the dict so ``list(values())`` produces one entry per model arg).
    """
    args = (torch.randn(2, 32),)
    sample = (args, {})
    graph_spec = make_graph_spec(_single_input, sample, batch_size=2)
    result = build_dynamic_shapes(sample, graph_spec)
    assert result == {"x": {}}


def test_build_dynamic_shapes_batch_axis():
    """Two samples with different batch sizes → axis 0 detected as batch."""
    args1 = (torch.randn(1, 32),)
    args2 = (torch.randn(2, 32),)
    sample = (args1, {})
    graph_spec = make_graph_spec(_single_input, sample, batch_size=1)
    update_input_spec(graph_spec, (args2, {}), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec)

    assert result is not None
    assert len(result) == 1
    assert isinstance(result["x"], dict)
    assert 0 in result["x"], "axis 0 should be marked as batch dynamic"


def test_build_dynamic_shapes_dynamic_axis():
    """Same batch size but different sequence length → axis 1 detected as dynamic."""
    args1 = (torch.randn(2, 5),)
    args2 = (torch.randn(2, 7),)
    sample = (args1, {})
    graph_spec = make_graph_spec(_single_input, sample, batch_size=2)
    update_input_spec(graph_spec, (args2, {}), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec)

    assert result is not None
    assert isinstance(result["x"], dict)
    assert 1 in result["x"], "axis 1 should be marked as dynamic (non-batch)"


def test_build_dynamic_shapes_mixed_batch_and_dynamic():
    """Different batch size AND non-proportional axis → both batch and dim axes present."""
    args1 = (torch.randn(1, 5),)  # bs=1, seq=5
    args2 = (torch.randn(2, 7),)  # bs=2, seq=7 (not proportional → dim1)
    sample = (args1, {})
    graph_spec = make_graph_spec(_single_input, sample, batch_size=1)
    update_input_spec(graph_spec, (args2, {}), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec)

    assert result is not None
    assert 0 in result["x"], "axis 0 should be batch"
    assert 1 in result["x"], "axis 1 should be dynamic"


def test_build_dynamic_shapes_batch_multiplier():
    """Axis 0 = 2 * batch_size (CFG-style UNet) → batch Dim uses actual axis values [2, 4]."""
    args1 = (torch.randn(2, 10),)  # bs=1, axis_0 = 2
    args2 = (torch.randn(4, 10),)  # bs=2, axis_0 = 4
    sample = (args1, {})
    graph_spec = make_graph_spec(_single_input, sample, batch_size=1)
    update_input_spec(graph_spec, (args2, {}), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec)

    assert result is not None
    assert 0 in result["x"], "axis 0 should be marked as batch dynamic"
    batch_dim = make_batch_dim(graph_spec.input_spec.tensor_specs)
    assert batch_dim is not None
    assert batch_dim.min == 2, "batch Dim min must equal the actual minimum axis_0 value"
    assert batch_dim.max == 4, "batch Dim max must equal the actual maximum axis_0 value"


def test_build_dynamic_shapes_batch_multiplier_only_bs1_with_dynamic_spatial():
    """CFG-doubled batch axis that is always 2 (batch=1 only) + dynamic spatial dims.

    Mirrors the stable-diffusion case: batch is always 1 so axis_0 is always 2 (constant),
    but image height/width vary across recordings.  The batch axis must NOT be marked
    dynamic (min_val == max_val → static), while the spatial dims must still be symbolic.
    """
    # batch=1 only; axis_0 = 2 always; height/width vary
    args1 = (torch.randn(2, 4, 32, 32),)
    args2 = (torch.randn(2, 4, 64, 64),)
    sample = (args1, {})
    graph_spec = make_graph_spec(_single_input, sample, batch_size=1)
    update_input_spec(graph_spec, (args2, {}), batch_size=1)

    result = build_dynamic_shapes(sample, graph_spec)

    assert result is not None, "spatial dynamic dims should produce a non-None result"
    assert 0 not in result["x"], "axis 0 (always 2) must NOT be marked dynamic"
    assert 2 in result["x"], "height axis should be dynamic"
    assert 3 in result["x"], "width axis should be dynamic"


def test_build_dynamic_shapes_shared_dim_across_tensors():
    """Two args tensors sharing the same dynamic dim name get the same Dim object."""
    args1 = (torch.randn(2, 5), torch.randn(2, 5))
    args2 = (torch.randn(2, 7), torch.randn(2, 7))
    sample = (args1, {})
    graph_spec = make_graph_spec(_two_inputs, sample, batch_size=2)
    update_input_spec(graph_spec, (args2, {}), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec)

    assert result is not None
    assert result["x"][1] is result["y"][1], "shared dim name must map to the same Dim instance"


def _explicit_two_input_graph_spec():
    sample = ((torch.randn(2, 4), torch.randn(2, 4)), {})
    graph_spec = make_graph_spec(_two_inputs, sample, batch_size=2)
    graph_spec.dynamic_shapes = {
        "x": (BatchDim("batch", min=1, opt=2, max=8), DynamicDim("sequence", min=1, opt=4, max=16)),
        "y": (BatchDim("batch", min=1, opt=2, max=8), DynamicDim("sequence", min=1, opt=4, max=16)),
    }
    return sample, graph_spec


def test_build_dynamic_shapes_explicit_batch_dim_stays_a_named_shared_dim():
    """BatchDim keeps an explicit named Dim so tensors agreeing on batch share one symbol."""
    sample, graph_spec = _explicit_two_input_graph_spec()

    result = build_dynamic_shapes(sample, graph_spec)

    assert result["x"][0].__name__ == "batch"
    assert (result["x"][0].min, result["x"][0].max) == (1, 8)
    assert result["x"][0] is result["y"][0]


def test_build_dynamic_shapes_explicit_non_batch_dim_uses_dynamic_hint():
    """Non-batch DynamicDim becomes a Dim.DYNAMIC hint carrying the user's bounds.

    An explicit Dim would force torch.export to prove every size in the range satisfies
    the model's traced guards, which most models reject.
    """
    sample, graph_spec = _explicit_two_input_graph_spec()

    result = build_dynamic_shapes(sample, graph_spec)

    sequence = result["x"][1]
    assert isinstance(sequence, torch.export.dynamic_shapes._DimHint)
    assert (sequence.min, sequence.max) == (1, 16)


def test_build_dynamic_shapes_explicit_dims_use_named_dims_when_use_auto_false():
    """``use_auto=False`` keeps explicit named Dims for every axis, shared by name."""
    sample, graph_spec = _explicit_two_input_graph_spec()

    result = build_dynamic_shapes(sample, graph_spec, use_auto=False)

    assert result["x"][1].__name__ == "sequence"
    assert (result["x"][1].min, result["x"][1].max) == (1, 16)
    assert result["x"][1] is result["y"][1]


def test_build_dynamic_shapes_kwargs_are_covered():
    """Dynamic kwargs are now included in the returned shapes dict."""
    arg = torch.randn(2, 32)
    kw1 = {"x": torch.randn(1, 10)}
    kw2 = {"x": torch.randn(2, 10)}
    # kwargs tensor changes batch; args tensor stays static
    sample = ((arg,), kw1)
    graph_spec = make_graph_spec(_arg_and_x, sample, batch_size=1)
    update_input_spec(graph_spec, ((arg,), kw2), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec)
    assert result is not None
    assert "x" in result
    assert 0 in result["x"], "axis 0 of kwargs tensor should be marked as batch dynamic"


def test_build_dynamic_shapes_optional_none_kwargs_padded():
    """Optional kwargs that are None at recording time must still appear in dynamic_shapes.

    torch.export.export requires that every key present in the kwargs passed to it
    is covered in dynamic_shapes, even if the value is None (= no dynamic dims).
    This mirrors the failure mode observed with UNet-style models where optional
    arguments (timestep_cond, cross_attention_kwargs, …) are None at tune time.
    """
    arg = torch.randn(1, 10)
    # Only the mandatory tensor kwarg is recorded; the optional ones are None.
    kw_recorded = {"x": torch.randn(1, 10)}
    kw_recorded_bs2 = {"x": torch.randn(2, 10)}
    sample = ((arg,), kw_recorded)
    graph_spec = make_graph_spec(_arg_and_options, sample, batch_size=1)
    update_input_spec(graph_spec, ((arg,), kw_recorded_bs2), batch_size=2)

    # Actual kwargs dict passed to export includes a None-valued optional and a dict-valued optional.
    actual_kwargs = {"x": torch.randn(1, 10), "opt_none": None, "opt_dict": {"key": "val"}}
    result = build_dynamic_shapes(((arg,), actual_kwargs), graph_spec)

    assert result is not None
    assert "opt_none" in result, "optional None kwarg must be present in dynamic_shapes"
    assert "opt_dict" in result, "optional dict kwarg must be present in dynamic_shapes"
    assert result["opt_none"] is None, "None kwarg should map to None in dynamic_shapes"
    assert result["opt_dict"] == {}, "dict kwarg should map to {} in dynamic_shapes"


# ---------------------------------------------------------------------------
# _make_batch_dim tests
# ---------------------------------------------------------------------------


def test_make_batch_dim_returns_none_when_no_batch_axis():
    """Tensor specs without any batch axis → batch_dim is None."""
    meta = _input_metadata(torch.randn(2, 32), batch_size=2)
    # single sample → no update_shapes_seen → no batch* labels
    batch_dim = make_batch_dim(list(meta.tensor_specs))
    assert batch_dim is None


def test_make_batch_dim_creates_correct_range():
    """Two batch sizes → batch_dim spans [min_axis_val, max_axis_val]."""
    meta = _input_metadata(torch.randn(1, 32), batch_size=1)
    meta.update_shapes_seen(_input_metadata(torch.randn(4, 32), batch_size=4))

    batch_dim = make_batch_dim(list(meta.tensor_specs))

    assert batch_dim is not None
    assert batch_dim.min == 1
    assert batch_dim.max == 4


def test_make_batch_dim_returns_none_when_batch_axis_is_constant():
    """CFG-doubled batch: axis_0 = 2 * bs, but only bs=1 recorded → axis_0 always 2 → None."""
    # Both samples have bs=1 but axis_0=2 (multiplier=2, consistent integer)
    meta = _input_metadata(torch.randn(2, 10), batch_size=1)
    meta.update_shapes_seen(_input_metadata(torch.randn(2, 10), batch_size=1))

    batch_dim = make_batch_dim(list(meta.tensor_specs))

    # min_val == max_val == 2 → static → None
    assert batch_dim is None


# ---------------------------------------------------------------------------
# _axis_dims_for_tensor tests
# ---------------------------------------------------------------------------


def test_axis_dims_for_tensor_dynamic_axis_uses_dim_auto():
    """Non-batch varying axis must be assigned Dim.AUTO, not an explicit range Dim."""
    meta = _input_metadata(torch.randn(2, 5), batch_size=2)
    meta.update_shapes_seen(_input_metadata(torch.randn(2, 9), batch_size=2))

    tensor_spec = list(meta.tensor_specs)[0]
    axis_dims = axis_dims_for_tensor(tensor_spec, batch_dim=None, tensor_name="x")

    assert 1 in axis_dims, "axis 1 should be present"
    assert axis_dims[1] is torch.export.Dim.AUTO


def test_axis_dims_for_tensor_static_axis_excluded():
    """Axis where min_shape == max_shape must not appear in the result."""
    meta = _input_metadata(torch.randn(1, 32), batch_size=1)
    meta.update_shapes_seen(_input_metadata(torch.randn(2, 32), batch_size=2))

    tensor_spec = list(meta.tensor_specs)[0]
    batch_dim = make_batch_dim(list(meta.tensor_specs))
    axis_dims = axis_dims_for_tensor(tensor_spec, batch_dim, tensor_name="x")

    assert 1 not in axis_dims, "static axis 1 (always 32) must not be included"


def test_axis_dims_for_tensor_batch_axis_uses_shared_batch_dim():
    """Batch axis must reference the same batch_dim object passed in."""
    meta = _input_metadata(torch.randn(1, 32), batch_size=1)
    meta.update_shapes_seen(_input_metadata(torch.randn(2, 32), batch_size=2))

    tensor_spec = list(meta.tensor_specs)[0]
    batch_dim = make_batch_dim(list(meta.tensor_specs))
    axis_dims = axis_dims_for_tensor(tensor_spec, batch_dim, tensor_name="x")

    assert 0 in axis_dims
    assert axis_dims[0] is batch_dim


# ---------------------------------------------------------------------------
# _prepare_export_sample tests
# ---------------------------------------------------------------------------


def test_prepare_export_sample_expands_size1_dim_axis():
    """A dim* axis with hint=1 is doubled to 2 via torch.cat."""
    args1 = (torch.zeros(2, 1),)  # batch=2, seq=1
    args2 = (torch.zeros(2, 8),)  # batch=2, seq=8
    gs = make_graph_spec(_single_input, (args1, {}), batch_size=2)
    update_input_spec(gs, (args2, {}), batch_size=2)

    out_args, _ = prepare_export_sample((args1, {}), gs)

    assert out_args[0].shape[1] == 2, "size-1 dim* axis must be expanded to 2"


def test_prepare_export_sample_does_not_expand_already_large_dim_axis():
    """A dim* axis already > 1 is left unchanged."""
    args1 = (torch.zeros(2, 5),)
    args2 = (torch.zeros(2, 9),)
    gs = make_graph_spec(_single_input, (args1, {}), batch_size=2)
    update_input_spec(gs, (args2, {}), batch_size=2)

    out_args, _ = prepare_export_sample((args1, {}), gs)

    assert out_args[0].shape[1] == 5, "dim* axis already > 1 must not be changed"


def test_prepare_export_sample_expands_batch_axis_to_2():
    """Batch axis with hint=1 is expanded to 2 via make_batch."""
    args1 = (torch.zeros(1, 32),)
    args2 = (torch.zeros(4, 32),)
    gs = make_graph_spec(_single_input, (args1, {}), batch_size=1)
    update_input_spec(gs, (args2, {}), batch_size=4)

    out_args, _ = prepare_export_sample((args1, {}), gs)

    assert out_args[0].shape[0] == 2, "batch axis at hint=1 must be expanded to 2"


def test_prepare_export_sample_batch_already_at_2_unchanged():
    """Batch axis already at 2 (max=2) is not further expanded."""
    args1 = (torch.zeros(1, 32),)
    args2 = (torch.zeros(2, 32),)
    gs = make_graph_spec(_single_input, (args1, {}), batch_size=1)
    update_input_spec(gs, (args2, {}), batch_size=2)

    # pass the bs=2 sample — already at the hint target
    out_args, _ = prepare_export_sample((args2, {}), gs)

    assert out_args[0].shape[0] == 2


def test_prepare_export_sample_keyword_input_dim_axis_expanded():
    """A dim* axis passed by keyword is expanded in the normalized call."""
    kw1 = {"mask": torch.zeros(2, 1)}
    kw2 = {"mask": torch.zeros(2, 6)}
    gs = make_graph_spec(_mask_input, ((), kw1), batch_size=2)
    update_input_spec(gs, ((), kw2), batch_size=2)

    out_args, _ = prepare_export_sample(((), kw1), gs)

    assert out_args[0].shape[1] == 2, "size-1 dim* axis passed by keyword must be expanded"


# ---------------------------------------------------------------------------
# build_dynamic_shapes additional tests
# ---------------------------------------------------------------------------


def test_build_dynamic_shapes_dim_axis_uses_dim_auto():
    """Non-batch dynamic axis must be Dim.AUTO in the result dict."""
    args1 = (torch.randn(2, 5),)
    args2 = (torch.randn(2, 9),)
    sample = (args1, {})
    graph_spec = make_graph_spec(_single_input, sample, batch_size=2)
    update_input_spec(graph_spec, (args2, {}), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec)

    assert result is not None
    assert result["x"][1] is torch.export.Dim.AUTO


def test_build_dynamic_shapes_batch_axis_at_non_zero_position():
    """Batch axis at position 1 (not 0) is correctly identified and marked dynamic."""
    args1 = (torch.randn(5, 1, 32),)  # [seq=5, batch=1, feat=32]
    args2 = (torch.randn(5, 2, 32),)  # [seq=5, batch=2, feat=32]
    sample = (args1, {})
    graph_spec = make_graph_spec(_single_input, sample, batch_size=1)
    update_input_spec(graph_spec, (args2, {}), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec)

    assert result is not None
    assert 1 in result["x"], "axis 1 (batch) should be dynamic"
    assert 0 not in result["x"], "axis 0 (static seq) must not be dynamic"
    assert 2 not in result["x"], "axis 2 (static feat) must not be dynamic"


def test_build_dynamic_shapes_multiple_dim_axes():
    """Two independent dim* axes (e.g. height and width) both get Dim.AUTO."""
    args1 = (torch.randn(2, 32, 32),)  # [batch, h, w]
    args2 = (torch.randn(2, 48, 64),)  # h and w change independently
    sample = (args1, {})
    graph_spec = make_graph_spec(_single_input, sample, batch_size=2)
    update_input_spec(graph_spec, (args2, {}), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec)

    assert result is not None
    assert 1 in result["x"], "height axis should be dynamic"
    assert 2 in result["x"], "width axis should be dynamic"
    assert result["x"][1] is torch.export.Dim.AUTO
    assert result["x"][2] is torch.export.Dim.AUTO


def test_build_dynamic_shapes_static_axis_not_in_result():
    """Axes that never change across samples must not appear in the result."""
    args1 = (torch.randn(1, 32),)
    args2 = (torch.randn(2, 32),)
    sample = (args1, {})
    graph_spec = make_graph_spec(_single_input, sample, batch_size=1)
    update_input_spec(graph_spec, (args2, {}), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec)

    assert result is not None
    assert 0 in result["x"], "batch axis should be dynamic"
    assert 1 not in result["x"], "feat axis (always 32) must not be in result"


# ---------------------------------------------------------------------------
# build_dynamic_shapes with use_auto=False (explicit-Dim path used by ONNX)
# ---------------------------------------------------------------------------


def test_build_dynamic_shapes_use_auto_false_explicit_dim_for_dim_axis():
    """``use_auto=False`` returns explicit Dim for non-batch dynamic axes (not Dim.AUTO)."""
    args1 = (torch.randn(2, 5),)
    args2 = (torch.randn(2, 9),)
    sample = (args1, {})
    graph_spec = make_graph_spec(_single_input, sample, batch_size=2)
    update_input_spec(graph_spec, (args2, {}), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec, use_auto=False)

    assert result is not None
    dim = result["x"][1]
    assert dim is not torch.export.Dim.AUTO
    assert dim.min == 5
    assert dim.max == 9


def test_build_dynamic_shapes_use_auto_false_caches_dim_by_range_across_tensors():
    """Two tensors with the same (min, max) range share the same Dim instance under use_auto=False."""
    args1 = (torch.randn(2, 5), torch.randn(2, 5))
    args2 = (torch.randn(2, 9), torch.randn(2, 9))
    sample = (args1, {})
    graph_spec = make_graph_spec(_two_inputs, sample, batch_size=2)
    update_input_spec(graph_spec, (args2, {}), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec, use_auto=False)

    assert result is not None
    assert result["x"][1] is result["y"][1], "shared (min, max) range must yield a single Dim instance"


def test_build_dynamic_shapes_use_auto_true_default_still_uses_dim_auto():
    """Default ``use_auto=True`` keeps the Dim.AUTO path (regression for default behaviour)."""
    args1 = (torch.randn(2, 5),)
    args2 = (torch.randn(2, 9),)
    sample = (args1, {})
    graph_spec = make_graph_spec(_single_input, sample, batch_size=2)
    update_input_spec(graph_spec, (args2, {}), batch_size=2)

    result = build_dynamic_shapes(sample, graph_spec)

    assert result is not None
    assert result["x"][1] is torch.export.Dim.AUTO


# ---------------------------------------------------------------------------
# End-to-end torch.export regression tests (no GPU, no torch_tensorrt)
# ---------------------------------------------------------------------------


def test_export_succeeds_with_shared_batch_kwargs_jit_style():
    """Two kwargs sharing a runtime batch axis must export successfully under JIT-style labelling.

    Reproduces the failure mode that motivated the trt-aot fix: HF encoders take
    ``input_ids`` and ``attention_mask`` whose batch dims must always be equal at
    runtime.  In JIT mode, ``BATCH_SIZE_KEY`` is not set during recording, so the
    batch heuristic in ``TensorSpec.update_shapes_seen`` falls through to the
    ``dim*`` label rather than ``batch*``.

    Under torch_tensorrt's old per-``Input`` Dim construction this raised
    ``ConstraintViolationError`` because the two axes were declared as
    independent symbols.  ``axis_dims_for_tensor`` now uses ``Dim.AUTO`` for
    ``dim*`` axes, which lets ``torch.export`` discover and merge the equality
    during tracing.
    """

    class SharedBatchModel(nn.Module):
        def forward(self, input_ids, attention_mask):
            # Forces input_ids.size(0) == attention_mask.size(0) at runtime.
            return input_ids + attention_mask

    model = SharedBatchModel().eval()

    # JIT-style recording: no batch_size argument → multipliers are NaN →
    # axis 0 gets labelled "dim0" rather than "batch0" even though it varies
    # with batch.
    kw1 = {
        "input_ids": torch.zeros(1, 8, dtype=torch.long),
        "attention_mask": torch.zeros(1, 8, dtype=torch.long),
    }
    kw2 = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.zeros(2, 8, dtype=torch.long),
    }
    gs = make_graph_spec(model.forward, ((), kw1))
    update_input_spec(gs, ((), kw2))

    args, kwargs = prepare_export_sample(((), kw1), gs)
    dynamic_shapes = build_dynamic_shapes((args, kwargs), gs)

    # Sanity: both shared axes resolve to Dim.AUTO; the old per-Input construction
    # would have produced two independent explicit Dims here.
    assert dynamic_shapes is not None
    assert dynamic_shapes["input_ids"][0] is torch.export.Dim.AUTO
    assert dynamic_shapes["attention_mask"][0] is torch.export.Dim.AUTO

    # Must NOT raise ConstraintViolationError.
    exported = torch.export.export(model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes, strict=False)
    assert exported is not None


def test_export_succeeds_with_explicit_spatial_range_on_strided_model():
    """A user spatial range must export even when the model guards on spatial parity.

    Reproduces the ResNet stem (stride-2 conv + stride-2 maxpool), which derives the
    guard ``((H - 1) // 2) % 2 != 0``. Only about half the sizes in any range satisfy it,
    so an explicit ``Dim("spatial", 224, 256)`` raises ``ConstraintViolationError`` and
    tuning falls back to eager. The ``Dim.DYNAMIC`` hint keeps the user's bounds and lets
    torch.export enforce the guard at runtime instead.
    """

    class StridedSpatialModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=7, stride=2, padding=3)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.head = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            return self.head(self.pool(self.conv(x))).flatten(1)

    model = StridedSpatialModel().eval()
    sample = ((torch.randn(1, 3, 224, 224),), {})
    graph_spec = make_graph_spec(model.forward, sample, batch_size=1)
    graph_spec.dynamic_shapes = {
        "x": (
            BatchDim("batch", min=1, opt=1, max=2),
            3,
            DynamicDim("spatial", min=224, opt=224, max=256),
            DynamicDim("spatial", min=224, opt=224, max=256),
        )
    }

    args, kwargs = prepare_export_sample(sample, graph_spec)
    dynamic_shapes = build_dynamic_shapes((args, kwargs), graph_spec)

    exported = torch.export.export(model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes, strict=False)

    # The user's bounds survive; sizes outside them are rejected at runtime.
    assert exported.module()(torch.randn(2, 3, 256, 256)).shape == (2, 4)
    with pytest.raises(AssertionError, match="Guard failed"):
        exported.module()(torch.randn(2, 3, 300, 300))
