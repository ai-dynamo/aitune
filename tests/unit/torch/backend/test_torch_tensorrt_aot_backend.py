# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch._export.verifier import SpecViolationError
from torch.export.graph_signature import ConstantArgument, TensorArgument

from aitune.torch.backend import ArtifactPath, torch_tensorrt_aot_backend
from aitune.torch.backend.torch_tensorrt_aot_backend import (
    TorchTensorRTAotBackend,
    TorchTensorRTAotBackendConfig,
    TorchTensorRTConfig,
    _restore_constant_inputs,
    _save_compiled_model,
    check_stage_shapes,
)
from aitune.torch.checkpoint.storage_tasks import torch_load_with_custom_types
from aitune.torch.dynamic_shapes import BatchDim
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.sample_store import Sample, SampleStore
from aitune.torch.utils.tensor import format_tensor_name
from tests.toy_models import ToyTorchModel
from tests.toy_models.torch_models import ToyTorchConditionalModel
from tests.utilities.helpers import make_recorded_graph_spec, make_sample_store, requires_cuda, update_input_spec


@dataclass
class TorchTensorRTTestConfig:
    workspace_size: int = 0
    device: int = 0


@dataclass
class TorchTensorRTTimingCacheTestConfig:
    timing_cache_path: str
    device: int = 0


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def module(self):
        return self


class FlatCacheModel(nn.Module):
    def __init__(self, *, reject_batch_one=False, truncate_output=False):
        super().__init__()
        self.reject_batch_one = reject_batch_one
        self.truncate_output = truncate_output

    def forward(self, x, cache):
        if self.reject_batch_one and x.shape[0] == 1:
            raise RuntimeError("u0 >= 2")
        cache.add_(x)
        output = x + cache
        return output[:1] if self.truncate_output else output, cache


def make_shape_validation_samples(tmp_path):
    model = FlatCacheModel()
    data = [((torch.ones(batch, 4), torch.zeros(batch, 4)), {}) for batch in (1, 2)]
    graph_spec = make_recorded_graph_spec(model, ((torch.ones(1, 4), torch.zeros(1, 4)), {}), name="0", strict=True)
    return graph_spec, make_sample_store(data, tmp_path)


@pytest.fixture
def model(torch_device) -> SimpleModel:
    return SimpleModel().to(torch_device).eval()


@pytest.fixture
def sample_data(torch_device, tmp_path) -> SampleStore:
    samples = [
        ((torch.randn(1, 10, device=torch_device),), {}),
        ((torch.randn(4, 10, device=torch_device),), {}),
        ((torch.randn(2, 10, device=torch_device),), {}),
    ]
    return make_sample_store(samples, tmp_path)


@pytest.fixture
def graph_spec(model, sample_data) -> GraphSpec:
    args, kwargs = sample_data[0]
    return make_recorded_graph_spec(model, (args, kwargs), name="0", strict=True)


@pytest.fixture
def backend(mocker) -> TorchTensorRTAotBackend:
    from aitune.torch.backend.torch_tensorrt_aot_backend import torch_tensorrt

    if torch_tensorrt is None:
        pytest.skip("torch_tensorrt is not available")

    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")  # always available

    return TorchTensorRTAotBackend(config=TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig()))


def _fake_torch_tensorrt_save(model, path, **kwargs):
    Path(path).write_bytes(b"fake")


def test_check_stage_shapes_preserves_recorded_inputs(tmp_path):
    graph_spec, samples = make_shape_validation_samples(tmp_path)
    expected = check_stage_shapes(FlatCacheModel(), graph_spec, samples, torch.device("cpu"), "export")

    actual = check_stage_shapes(FlatCacheModel(), graph_spec, samples, torch.device("cpu"), "reload", expected)

    assert actual == expected == [(torch.Size([1, 4]),) * 2, (torch.Size([2, 4]),) * 2]
    for args, _ in samples:
        assert torch.count_nonzero(args[1]) == 0


def test_check_stage_shapes_reports_runtime_restriction(tmp_path):
    graph_spec, samples = make_shape_validation_samples(tmp_path)

    with pytest.raises(RuntimeError, match=r"reload.*sample 0.*inputs.*1, 4.*u0 >= 2"):
        check_stage_shapes(FlatCacheModel(reject_batch_one=True), graph_spec, samples, torch.device("cpu"), "reload")


def test_check_stage_shapes_reports_output_shape_change(tmp_path):
    graph_spec, samples = make_shape_validation_samples(tmp_path)
    expected = check_stage_shapes(FlatCacheModel(), graph_spec, samples, torch.device("cpu"), "export")

    with pytest.raises(RuntimeError, match=r"compilation.*sample 1.*expected.*2, 4.*got.*1, 4"):
        check_stage_shapes(
            FlatCacheModel(truncate_output=True), graph_spec, samples, torch.device("cpu"), "compilation", expected
        )


def test_save_compiled_model_without_retracing(mocker, tmp_path: Path):
    mock_torch_tensorrt = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt")
    mock_torch_tensorrt.logging.errors.return_value = mocker.MagicMock()
    module = mocker.Mock(spec=nn.Module)

    _save_compiled_model(module, tmp_path / "model.pt", sample=((), {}), dynamic_shapes=None, pickle_protocol=5)

    mock_torch_tensorrt.dynamo.export.assert_called_once_with(module)
    mock_torch_tensorrt.save.assert_called_once_with(
        mock_torch_tensorrt.dynamo.export.return_value,
        (tmp_path / "model.pt").as_posix(),
        pickle_protocol=5,
    )


def test_save_compiled_model_uses_legacy_exporter_when_available(mocker, tmp_path: Path):
    mock_torch_tensorrt = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt")
    mock_torch_tensorrt.logging.errors.return_value = mocker.MagicMock()
    program = mocker.MagicMock()
    exporter_flags = []

    def export(gm: nn.Module, *, use_legacy_exporter: bool = False):  # Torch-TensorRT 2.11+ signature
        exporter_flags.append(use_legacy_exporter)
        return program

    mock_torch_tensorrt.dynamo.export = export

    _save_compiled_model(
        mocker.Mock(spec=nn.Module), tmp_path / "model.pt", sample=((), {}), dynamic_shapes=None, pickle_protocol=5
    )

    assert exporter_flags == [True]
    mock_torch_tensorrt.save.assert_called_once_with(program, (tmp_path / "model.pt").as_posix(), pickle_protocol=5)


@pytest.mark.parametrize("with_kwargs", [False, True], ids=["flat-tensors", "non-tensor-keyword"])
@pytest.mark.parametrize(
    "direct_save_error",
    [
        SpecViolationError("direct save failed"),
        RuntimeError("Only Tensors of floating point and complex dtype can require gradients"),
    ],
)
def test_save_compiled_model_retraces_after_direct_save_failure(
    mocker, tmp_path: Path, with_kwargs: bool, direct_save_error: Exception
):
    mock_torch_tensorrt = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt")
    mock_torch_tensorrt.logging.errors.return_value = mocker.MagicMock()
    mock_torch_tensorrt.save.side_effect = [direct_save_error, None]
    module = mocker.Mock(spec=nn.Module)
    sample = ((mocker.sentinel.input,), {"output_hidden_states": True} if with_kwargs else {})
    dynamic_shapes = {"x": {0: mocker.sentinel.batch}}
    mock_export = mocker.patch("torch.export.export")

    _save_compiled_model(module, tmp_path / "model.pt", sample, dynamic_shapes=dynamic_shapes, pickle_protocol=5)

    mock_export.assert_called_once_with(
        module, sample[0], sample[1] or None, dynamic_shapes=dynamic_shapes, strict=False
    )
    assert mock_torch_tensorrt.save.call_args_list == [
        mocker.call(
            mock_torch_tensorrt.dynamo.export.return_value,
            (tmp_path / "model.pt").as_posix(),
            pickle_protocol=5,
        ),
        mocker.call(mock_export.return_value, (tmp_path / "model.pt").as_posix(), pickle_protocol=5),
    ]


@pytest.mark.parametrize("direct_save_error", [OSError("disk full"), RuntimeError("unrelated runtime failure")])
def test_save_compiled_model_does_not_retrace_after_unrelated_error(
    mocker, tmp_path: Path, direct_save_error: Exception
):
    mock_torch_tensorrt = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt")
    mock_torch_tensorrt.logging.errors.return_value = mocker.MagicMock()
    mock_torch_tensorrt.save.side_effect = direct_save_error
    module = mocker.Mock(spec=nn.Module)

    with pytest.raises(type(direct_save_error), match=str(direct_save_error)):
        _save_compiled_model(module, tmp_path / "model.pt", sample=((), {}), dynamic_shapes=None, pickle_protocol=5)

    mock_torch_tensorrt.save.assert_called_once()


def test_restore_constant_inputs_makes_non_tensor_kwargs_serializable(tmp_path: Path):
    class ScaledModel(nn.Module):
        def forward(self, x: torch.Tensor, scale: float, mask: torch.Tensor | None = None) -> torch.Tensor:
            return x * scale if mask is None else x * scale * mask

    program = torch.export.export(ScaledModel(), (torch.ones(2),), {"scale": 2.0, "mask": None})
    # Torch-TensorRT declares every placeholder as a tensor input: https://github.com/pytorch/TensorRT/issues/4941
    for spec in program.graph_signature.input_specs:
        spec.arg = TensorArgument(name=spec.arg.name)
    with pytest.raises(KeyError, match="scale"):
        torch.export.save(program, tmp_path / "invalid.pt2")

    _restore_constant_inputs(program)

    assert [spec.arg for spec in program.graph_signature.input_specs] == [
        TensorArgument(name="x"),
        ConstantArgument(name="scale", value=2.0),
        ConstantArgument(name="mask", value=None),
    ]
    torch.export.save(program, tmp_path / "model.pt2")
    loaded = torch.export.load(tmp_path / "model.pt2").module()
    assert torch.equal(loaded(torch.ones(2), scale=2.0, mask=None), torch.full((2,), 2.0))


def test_save_compiled_model_retraces_non_tensor_inputs(mocker, tmp_path: Path):
    class StructuredModel(nn.Module):
        def forward(
            self, x: torch.Tensor, extras: dict[str, torch.Tensor], scale: float, mask: torch.Tensor | None
        ) -> torch.Tensor:
            output = (x + extras["bias"]) * scale
            return output if mask is None else output * mask

    batch = torch.export.Dim("batch", min=1, max=8)
    dynamic_shapes = {"x": {0: batch}, "extras": {"bias": {0: batch}}, "scale": None, "mask": None}
    sample = ((torch.ones(2, 4), {"bias": torch.ones(2, 4)}, 2.0, None), {})
    module = torch.export.export(StructuredModel(), sample[0], dynamic_shapes=dynamic_shapes).module()
    mock_torch_tensorrt = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt")
    mock_torch_tensorrt.logging.errors.return_value = mocker.MagicMock()
    mock_torch_tensorrt.dynamo.export.side_effect = SpecViolationError("direct save failed")
    mock_torch_tensorrt.save.side_effect = lambda program, path, **_: torch.export.save(program, path)

    _save_compiled_model(module, tmp_path / "model.pt2", sample, dynamic_shapes=dynamic_shapes, pickle_protocol=5)

    loaded = torch.export.load(tmp_path / "model.pt2").module()
    output = loaded(torch.ones(3, 4), {"bias": torch.ones(3, 4)}, 2.0, None)
    assert torch.equal(output, torch.full((3, 4), 4.0))


@pytest.fixture
def mock_torch_tensorrt(mocker, model: SimpleModel):
    mock_torch_tensorrt = mocker.Mock()
    mock_torch_tensorrt.dynamo.export.return_value = mocker.MagicMock()
    mock_torch_tensorrt.logging.warnings.return_value = mocker.MagicMock()
    mock_torch_tensorrt.logging.errors.return_value = mocker.MagicMock()
    mock_torch_tensorrt.dynamo.compile = mocker.Mock(return_value=model)
    mock_torch_tensorrt.save = mocker.Mock(side_effect=_fake_torch_tensorrt_save)
    mock_torch_tensorrt.load.return_value.module.return_value = model

    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt", mock_torch_tensorrt)
    return mock_torch_tensorrt


def _graph_spec_from_samples(model: nn.Module, samples: list[Sample]) -> GraphSpec:
    graph_spec = make_recorded_graph_spec(model, samples[0], name="0", strict=True)
    for args, kwargs in samples[1:]:
        output = model(*args, **kwargs)
        update_input_spec(graph_spec, (args, kwargs), strict=True)
        graph_spec.output_spec.update_shapes_seen(SampleMetadata.from_outputs(output))
    return graph_spec


@requires_cuda
def test_torch_tensorrt_aot_backend_config_key():
    """Test backend config with cache_dir."""

    config = TorchTensorRTAotBackendConfig()
    key1 = config.key()
    key2 = config.key()

    assert key1 == key2


@requires_cuda
def test_torch_tensorrt_aot_backend_config_describe():
    """Test backend config with cache_dir."""
    config = TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig())
    describe = config.describe()

    assert describe == "compile_config=TorchTensorRTConfig()"

    config = TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig(workspace_size=1))
    describe = config.describe()

    assert describe == "compile_config=TorchTensorRTConfig(workspace_size=1)"

    config = TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig(), pickle_protocol=1)
    describe = config.describe()

    assert describe == "compile_config=TorchTensorRTConfig(),pickle_protocol=1"


def test_torch_tensorrt_aot_description_omits_rank_local_device(mocker):
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_torch_tensorrt")
    backend = TorchTensorRTAotBackend(
        TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig(workspace_size=1, device=3))
    )

    assert "device=" not in backend.describe()
    assert "workspace_size=1" in backend.describe()


def test_torch_tensorrt_timing_cache_path_is_rank_isolated(mocker, tmp_path):
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_torch_tensorrt")
    timing_cache = tmp_path / "timing_cache.bin"
    isolated_cache = tmp_path / "timing_cache.rank-2-of-4.bin"
    resolve = mocker.patch(
        "aitune.torch.backend.torch_tensorrt_aot_backend.distributed_output_path",
        return_value=isolated_cache,
    )
    backend = TorchTensorRTAotBackend(
        TorchTensorRTAotBackendConfig(
            compile_config=TorchTensorRTTimingCacheTestConfig(timing_cache.as_posix()),
        )
    )

    settings = backend._compile_settings()

    assert settings["timing_cache_path"] == isolated_cache.as_posix()
    resolve.assert_called_once_with(timing_cache)
    assert backend._config.compile_config.timing_cache_path == timing_cache.as_posix()


def test_infer_restores_mutable_structured_inputs(mocker):
    class StatefulDecoder(nn.Module):
        def forward(self, x, cache):
            cache[0] = x + 1
            return x * 2

    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_torch_tensorrt")
    model = StatefulDecoder()
    x = torch.ones(2)
    graph_spec = make_recorded_graph_spec(model, ((x, [None]), {}), strict=True)
    backend = TorchTensorRTAotBackend()
    backend._graph_spec = graph_spec
    backend._opt_module = mocker.Mock(return_value=(x * 2, x + 1))
    cache = [None]

    result = backend._infer(x, cache)

    backend._opt_module.assert_called_once_with(x)
    torch.testing.assert_close(result, x * 2)
    torch.testing.assert_close(cache[0], x + 1)


@requires_cuda
def test_mock_build(
    mocker,
    mock_torch_tensorrt,
    backend: TorchTensorRTAotBackend,
    model: SimpleModel,
    graph_spec: GraphSpec,
    sample_data: SampleStore,
    torch_device: torch.device,
    tmp_path: Path,
):
    export_mock = mocker.spy(torch.export, "export")
    save_compiled = mocker.spy(torch_tensorrt_aot_backend, "_save_compiled_model")

    active_backend = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)

    assert active_backend is backend

    assert active_backend._device == torch_device
    assert active_backend.is_active

    export_mock.assert_called_once()
    locator, tensor_spec = graph_spec.input_spec.tensor_data[0]
    mock_torch_tensorrt.Input.assert_called_once_with(
        min_shape=tensor_spec.min_shape,
        opt_shape=tensor_spec.max_shape,
        max_shape=tensor_spec.max_shape,
        dtype=tensor_spec.dtype,
        name=format_tensor_name(locator.path, "input"),
    )
    mock_torch_tensorrt.dynamo.compile.assert_called_once()
    mock_torch_tensorrt.load.assert_called_once_with((tmp_path / "exported_model.pt2").as_posix())
    assert active_backend._opt_module is model
    mock_torch_tensorrt.logging.warnings.assert_called_once_with()
    warning_context = mock_torch_tensorrt.logging.warnings.return_value
    warning_context.__enter__.assert_called_once_with()
    warning_context.__exit__.assert_called_once()
    # Pin the pipeline shape: the ExportedProgram from torch.export.export must be
    # forwarded as the first positional arg to torch_tensorrt.dynamo.compile.
    assert mock_torch_tensorrt.dynamo.compile.call_args[0][0] is export_mock.spy_return
    assert save_compiled.call_args.args[2] is export_mock.spy_return.example_inputs


@pytest.mark.parametrize("validate_stage_shapes", [False, True])
@pytest.mark.parametrize(
    ("dynamic_shapes", "expected_opt", "expected_max"),
    [
        pytest.param(None, 4, 4, id="inferred"),
        pytest.param(
            {"x": (BatchDim("batch", min=1, opt=2, max=8), 10)},
            2,
            8,
            id="explicit",
        ),
    ],
)
def test_mock_build_exports_bounded_dynamic_shapes(
    mocker,
    tmp_path: Path,
    dynamic_shapes,
    expected_opt,
    expected_max,
    validate_stage_shapes,
):
    model = SimpleModel().eval()
    sample_data = [
        ((torch.randn(1, 10),), {}),
        ((torch.randn(4, 10),), {}),
        ((torch.randn(2, 10),), {}),
    ]
    graph_spec = _graph_spec_from_samples(model, sample_data)
    if dynamic_shapes is not None:
        graph_spec.dynamic_shapes = dynamic_shapes
    mock_torch_tensorrt = mocker.Mock()
    mock_torch_tensorrt.dynamo.export.return_value = mocker.MagicMock()
    mock_torch_tensorrt.logging.warnings.return_value = mocker.MagicMock()
    mock_torch_tensorrt.logging.errors.return_value = mocker.MagicMock()
    mock_torch_tensorrt.dynamo.compile = mocker.Mock(return_value=model)
    mock_torch_tensorrt.save = mocker.Mock(side_effect=_fake_torch_tensorrt_save)
    mock_torch_tensorrt.load.return_value.module.return_value = model
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt", mock_torch_tensorrt)
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_torch_tensorrt")
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.get_cuda_device", return_value=0)
    mocker.patch.object(TorchTensorRTAotBackend, "_devices", ["cpu", "cuda"])

    backend = TorchTensorRTAotBackend(
        config=TorchTensorRTAotBackendConfig(
            compile_config=TorchTensorRTTestConfig(), validate_stage_shapes=validate_stage_shapes
        )
    )
    export_mock = mocker.spy(torch.export, "export")
    save_compiled = mocker.spy(torch_tensorrt_aot_backend, "_save_compiled_model")
    shape_checks = mocker.spy(torch_tensorrt_aot_backend, "check_stage_shapes")

    backend.build(
        model, graph_spec, make_sample_store(sample_data, tmp_path), device=torch.device("cpu"), cache_dir=tmp_path
    )

    assert [call.args[4] for call in shape_checks.call_args_list] == (
        ["torch.export", "TensorRT compilation", "serialization and reload"] if validate_stage_shapes else []
    )
    dynamic_shapes = export_mock.call_args.kwargs["dynamic_shapes"]
    assert save_compiled.call_args.args[2] is export_mock.spy_return.example_inputs
    assert save_compiled.call_args.args[3] is dynamic_shapes["flat_inputs"]
    reexported = torch.export.export(
        export_mock.spy_return.module(),
        export_mock.spy_return.example_inputs[0],
        dynamic_shapes=save_compiled.call_args.args[3],
        strict=False,
    )
    path = tmp_path / "reexported.pt2"
    torch.export.save(reexported, path)
    loaded = torch.export.load(path).module()
    value = torch.randn(3, 10)
    torch.testing.assert_close(loaded(value)[0], model(value))
    dim = dynamic_shapes["flat_inputs"][0][0]
    assert dim is not torch.export.Dim.AUTO
    assert dim.min == 1
    assert dim.max == expected_max
    mock_torch_tensorrt.Input.assert_called_once_with(
        min_shape=[1, 10],
        opt_shape=[expected_opt, 10],
        max_shape=[expected_max, 10],
        dtype=torch.float32,
        name=format_tensor_name("x", "input"),
    )
    mock_torch_tensorrt.logging.warnings.assert_called_once_with()
    warning_context = mock_torch_tensorrt.logging.warnings.return_value
    warning_context.__enter__.assert_called_once_with()
    warning_context.__exit__.assert_called_once()


def test_build_uses_placement_preserving_module_move(mocker, tmp_path):
    model = SimpleModel().eval()
    move_module = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.move_module_to_device")
    sample_data = [((torch.randn(1, 10),), {})]
    graph_spec = _graph_spec_from_samples(model, sample_data)
    mock_torch_tensorrt = mocker.Mock()
    mock_torch_tensorrt.dynamo.export.return_value = mocker.MagicMock()
    mock_torch_tensorrt.logging.warnings.return_value = mocker.MagicMock()
    mock_torch_tensorrt.logging.errors.return_value = mocker.MagicMock()
    mock_torch_tensorrt.dynamo.compile.return_value = model
    mock_torch_tensorrt.save.side_effect = _fake_torch_tensorrt_save
    mock_torch_tensorrt.load.return_value.module.return_value = model
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt", mock_torch_tensorrt)
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_torch_tensorrt")
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.get_cuda_device", return_value=0)
    mocker.patch.object(TorchTensorRTAotBackend, "_devices", ["cpu", "cuda"])
    backend = TorchTensorRTAotBackend(config=TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig()))

    backend.build(
        model, graph_spec, make_sample_store(sample_data, tmp_path), device=torch.device("cpu"), cache_dir=tmp_path
    )

    move_module.assert_called_once_with(model, torch.device("cpu"))


@requires_cuda
def test_mock_infer(
    mocker,
    model: SimpleModel,
    graph_spec: GraphSpec,
    sample_data: SampleStore,
    backend: TorchTensorRTAotBackend,
    torch_device: torch.device,
    tmp_path: Path,
):
    mocker.patch("torch.export.save")
    load = mocker.patch("torch.export.load", return_value=model)
    mocker.patch(
        "aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt.save",
        side_effect=_fake_torch_tensorrt_save,
    )
    mocker.patch(
        "aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt.dynamo.export",
        return_value=mocker.MagicMock(),
    )
    # Keep export/profile validation real and bypass TRT compilation with the original model.
    mocker.patch(
        "aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt.dynamo.compile",
        return_value=model,
    )
    loaded = mocker.Mock()
    loaded.module.return_value = model
    mocker.patch(
        "aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt.load",
        return_value=loaded,
    )

    backend = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)

    args, kwargs = sample_data[0]

    output = backend.infer(*args, **kwargs)
    assert output.shape == (1, 2)

    load.assert_not_called()


@requires_cuda
def test_build_with_custom_precisions(
    model: SimpleModel,
    graph_spec: GraphSpec,
    sample_data: SampleStore,
    mocker,
    mock_torch_tensorrt,
    torch_device: torch.device,
    tmp_path: Path,
):
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")

    config = TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig(workspace_size=1))
    backend = TorchTensorRTAotBackend(
        config=config,
    )

    backend = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    assert backend is not None

    mock_torch_tensorrt.dynamo.compile.assert_called_once()
    assert mock_torch_tensorrt.dynamo.compile.call_args[1]["workspace_size"] == 1


@requires_cuda
def test_full_run_simple_model(
    backend: TorchTensorRTAotBackend,
    model: SimpleModel,
    graph_spec: GraphSpec,
    sample_data: SampleStore,
    torch_device: torch.device,
    tmp_path: Path,
):
    backend = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)

    assert backend._exported_model_artifact == ArtifactPath(tmp_path, Path("exported_model.pt2"))
    assert backend._exported_model_artifact.path.exists()

    args, kwargs = sample_data[0]

    assert backend._opt_module is not None

    output = backend.infer(*args, **kwargs)
    assert output.shape == (1, 2)


@requires_cuda
def test_full_run_toy_model_linear(torch_device: torch.device, tmp_path: Path):
    model = ToyTorchModel().to(torch_device)
    samples = model.samples(batch_sizes=[2], device=torch_device)
    graph_spec = model.graph_spec(batch_sizes=[2], device=torch_device)

    backend = TorchTensorRTAotBackend()
    backend = backend.build(
        model, graph_spec, make_sample_store(samples, tmp_path), device=torch_device, cache_dir=tmp_path
    )
    args, kwargs = samples[0]

    assert backend._opt_module is not None

    output = backend.infer(*args, **kwargs)
    assert output.shape == (2, 5)


@pytest.mark.skip(reason="Currently not supported by Torch-TRT https://github.com/pytorch/TensorRT/issues/767")
@requires_cuda
def test_conditional_model(torch_device, tmp_path: Path):
    model = ToyTorchConditionalModel().to(torch_device)
    inputs = model.inputs(batch_sizes=[2], device=torch_device)
    samples = model.samples(batch_sizes=[2], device=torch_device, kwargs={"apply_relu": True})
    graph_spec = model.graph_spec(batch_sizes=[2], device=torch_device, kwargs={"apply_relu": True})

    backend = TorchTensorRTAotBackend()
    backend = backend.build(
        model, graph_spec, make_sample_store(samples, tmp_path), device=torch_device, cache_dir=tmp_path
    )

    assert backend._opt_module is not None

    output = backend.infer(inputs, apply_relu=True)
    assert output.shape == (2, 5)


@requires_cuda
def test_serialization(torch_device, tmp_path):
    model = ToyTorchModel().to(torch_device)
    samples = model.samples(batch_sizes=[2], device=torch_device)
    graph_spec = model.graph_spec(batch_sizes=[2], device=torch_device)

    backend = TorchTensorRTAotBackend()
    backend = backend.build(
        model, graph_spec, make_sample_store(samples, tmp_path), device=torch_device, cache_dir=tmp_path
    )

    state_dict = backend.to_dict()

    torch.save(state_dict, tmp_path / "state_dict.pth")
    loaded_backend = TorchTensorRTAotBackend.from_dict(model, torch_load_with_custom_types(tmp_path / "state_dict.pth"))

    loaded_backend.activate()
    args, kwargs = samples[0]
    torch.testing.assert_close(backend.infer(*args, **kwargs), loaded_backend.infer(*args, **kwargs))


# --- TorchTensorRTAotBackendConfig.from_dict ---


def test_tensorrt_aot_config_from_dict_defaults():
    config = TorchTensorRTAotBackendConfig.from_dict({})
    default = TorchTensorRTAotBackendConfig()
    assert config.pickle_protocol == default.pickle_protocol
    assert config.pickle_protocol == 5


def test_tensorrt_aot_config_from_dict_nested_compile_config_dict():
    data = {"compile_config": {"workspace_size": 2048}}
    config = TorchTensorRTAotBackendConfig.from_dict(data)
    assert isinstance(config.compile_config, TorchTensorRTConfig)
    assert config.compile_config.workspace_size == 2048


def test_tensorrt_aot_config_from_dict_compile_config_instance_passthrough():
    config_instance = TorchTensorRTAotBackendConfig().compile_config
    config = TorchTensorRTAotBackendConfig.from_dict({"compile_config": config_instance})
    assert config.compile_config is config_instance
