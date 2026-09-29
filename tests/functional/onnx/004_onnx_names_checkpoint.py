# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = []
# scope = "always"
# allow_failure = false
# [[pip_install]]
# packages = ["onnxruntime-gpu"]
# flags = ["--upgrade", "--index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"]
# ///

"""Preserve original ONNX names, shape ranges, and external artifacts through tuning and save/load.

Run: python -m pytest tests/functional/onnx/004_onnx_names_checkpoint.py -q -s
"""

from pathlib import Path

import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from aitune.torch import Module, OneBackendStrategy, load, save, tune
from aitune.torch.backend import ONNXRuntimeBackend, TensorRTBackend
from aitune.torch.dataloader import DataLoaderFactory, DynamicShapeDataset
from aitune.torch.module import OnnxModule
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.task.profiling import ProfilingConfig


def _create_model(path: Path) -> None:
    weight = numpy_helper.from_array(torch.full((3,), 2.0).numpy(), name="scale")
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["input.1", "scale"], ["scaled"]),
            helper.make_node("Add", ["scaled", "input/1"], ["sum:0"]),
            helper.make_node("Sub", ["input.1", "input/1"], ["difference.1"]),
        ],
        "named-arithmetic",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", 3]) for name in ["input.1", "input/1"]],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", 3]) for name in ["sum:0", "difference.1"]],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )


def _sample(layout: str, lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[tuple, dict]:
    if layout == "positional":
        return (lhs, rhs), {}
    if layout == "mixed":
        return (lhs,), {"input/1": rhs}
    # Reverse keyword order; these input names also collide if sanitized.
    return (), {"input/1": rhs, "input.1": lhs}


def _assert_metadata(graph: GraphSpec) -> None:
    assert {spec.name for spec in graph.input_spec.tensor_specs} == {"input.1", "input/1"}
    assert {spec.name for spec in graph.output_spec.tensor_specs} == {"sum:0", "difference.1"}
    for spec in graph.input_spec.tensor_specs:
        assert spec.min_shape == [1, 3]
        assert spec.max_shape == [4, 3]
        assert spec.get_batch_axis_multipliers() == {0: 1}


@pytest.mark.parametrize("backend_type", [ONNXRuntimeBackend, TensorRTBackend])
@pytest.mark.parametrize("layout", ["positional", "named", "mixed"])
@torch.inference_mode()
def test_onnx_names_checkpoint(tmp_path: Path, backend_type, layout: str, monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    path = tmp_path / "source.onnx"
    _create_model(path)
    source = OnnxModule(path)
    strategy = OneBackendStrategy(backend_type(), profiling_config=ProfilingConfig(batch_sizes=[1, 2, 4]))
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    module = Module(source, "named-onnx", strategy=strategy)
    lhs = torch.arange(12, device="cuda", dtype=torch.float32).reshape(4, 3)
    rhs = lhs.flip(0) + 1
    samples = DynamicShapeDataset(list(zip(lhs, rhs, strict=True)))

    def collate(batch):
        x, y = torch.utils.data.default_collate(batch)
        return _sample(layout, x, y)

    dataset = DataLoaderFactory(samples, collate_fn=collate)

    # A source ONNX file must bypass torch export for both backend implementations.
    def unexpected_export(*args, **kwargs):
        pytest.fail("Existing ONNX models must not be exported through torch")

    monkeypatch.setattr(torch.onnx, "export", unexpected_export)
    try:
        tune(module, dataset, batch_sizes=[1, 2, 4], device="cuda", ignore_failing_modules=False)
        (backend,) = module.module.backends.values()
        assert isinstance(backend, backend_type)
        _assert_metadata(backend._graph_spec)
        for batch in [1, 3, 4]:
            args, kwargs = _sample(layout, lhs[:batch], rhs[:batch])
            torch.testing.assert_close(
                module(*args, **kwargs),
                {"sum:0": lhs[:batch] * 2 + rhs[:batch], "difference.1": lhs[:batch] - rhs[:batch]},
            )

        checkpoint = tmp_path / "model.ait"
        save(module, checkpoint)
        module.deactivate()
        path.unlink()
        (tmp_path / "weights.bin").unlink()
        module = load(module, checkpoint)
        (backend,) = module.module.backends.values()
        assert isinstance(backend, backend_type)
        _assert_metadata(backend._graph_spec)
        for batch in [1, 3, 4]:
            args, kwargs = _sample(layout, lhs[:batch], rhs[:batch])
            actual = module(*args, **kwargs)
            assert all(value.is_cuda for value in actual.values())
            torch.testing.assert_close(
                actual, {"sum:0": lhs[:batch] * 2 + rhs[:batch], "difference.1": lhs[:batch] - rhs[:batch]}
            )
    finally:
        # Loaded backends are deployed; release their resources directly for test cleanup.
        if module.state.name == "TUNED":
            for backend in module.module.backends.values():
                backend._deactivate()
        source.deactivate()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
