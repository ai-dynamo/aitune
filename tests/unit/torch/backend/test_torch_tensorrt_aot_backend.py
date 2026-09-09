# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn as nn
from torch._export.verifier import SpecViolationError

from aitune.torch.backend import ArtifactPath
from aitune.torch.backend.torch_tensorrt_aot_backend import (
    TorchTensorRTAotBackend,
    TorchTensorRTAotBackendConfig,
    TorchTensorRTConfig,
    _save_compiled_model,
)
from aitune.torch.checkpoint.storage_tasks import torch_load_with_custom_types
from aitune.torch.dynamic_shapes import BatchDim
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.sample_store import Sample
from aitune.torch.utils.tensor import format_tensor_name
from tests.toy_models import ToyTorchModel
from tests.toy_models.torch_models import ToyTorchConditionalModel
from tests.utilities.helpers import make_recorded_graph_spec, requires_cuda, update_input_spec


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


@pytest.fixture
def model(torch_device) -> SimpleModel:
    return SimpleModel().to(torch_device).eval()


@pytest.fixture
def sample_data(torch_device) -> list[Sample]:
    return [
        ((torch.randn(1, 10, device=torch_device),), {}),
        ((torch.randn(4, 10, device=torch_device),), {}),
        ((torch.randn(2, 10, device=torch_device),), {}),
    ]


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


def test_stateful_save_compiled_model_uses_public_dynamo_export(mocker, tmp_path):
    library = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt")
    library.logging.errors.return_value = mocker.MagicMock()
    module = mocker.Mock(spec=nn.Module)
    inputs = (torch.randn(3, 10),)
    shapes = ({0: torch.export.Dim("batch", min=2, max=8)},)

    _save_compiled_model(module, tmp_path / "model.pt2", inputs, shapes, 5, stateful=True)

    library.save.assert_called_once_with(
        module,
        (tmp_path / "model.pt2").as_posix(),
        output_format="exported_program",
        retrace=True,
        arg_inputs=inputs,
        dynamic_shapes=shapes,
        pickle_protocol=5,
    )


@pytest.mark.parametrize(
    "direct_error",
    [
        None,
        SpecViolationError("direct export failed"),
        RuntimeError("Only Tensors of floating point and complex dtype can require gradients"),
    ],
)
def test_stateless_save_retains_direct_save_fallback(mocker, tmp_path, direct_error):
    library = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt")
    library.logging.errors.return_value = mocker.MagicMock()
    library.save.side_effect = [direct_error, None]
    inputs = (torch.randn(3),)
    shapes = ({0: torch.export.Dim("batch", min=2, max=8)},)

    _save_compiled_model(mocker.Mock(spec=nn.Module), tmp_path / "model.pt2", inputs, shapes, 5)

    calls = library.save.call_args_list
    assert [call.kwargs["retrace"] for call in calls] == ([False, True] if direct_error else [False])
    assert "arg_inputs" not in calls[0].kwargs
    if direct_error:
        assert calls[1].kwargs["arg_inputs"] is inputs
        assert calls[1].kwargs["dynamic_shapes"] is shapes


@pytest.mark.parametrize("error", [OSError("disk full"), RuntimeError("unrelated failure")])
def test_stateless_save_does_not_retry_unrelated_failures(mocker, tmp_path, error):
    library = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt")
    library.logging.errors.return_value = mocker.MagicMock()
    library.save.side_effect = error
    with pytest.raises(type(error)) as caught:
        _save_compiled_model(mocker.Mock(spec=nn.Module), tmp_path / "model.pt2", (), None, 5)
    assert caught.value is error
    library.save.assert_called_once()


@pytest.mark.parametrize("use_legacy_exporter", [False, True])
def test_save_compiled_model_passes_explicit_exporter(mocker, tmp_path, use_legacy_exporter):
    def save(module, path, *, use_legacy_exporter=None, **kwargs):
        pass

    library = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt")
    library.logging.errors.return_value = mocker.MagicMock()
    library.save = mocker.create_autospec(save)

    _save_compiled_model(
        mocker.Mock(spec=nn.Module),
        tmp_path / "model.pt2",
        (),
        None,
        5,
        use_legacy_exporter=use_legacy_exporter,
    )

    assert library.save.call_args.kwargs["use_legacy_exporter"] is use_legacy_exporter
    assert library.save.call_args.kwargs["retrace"] is True


@pytest.mark.parametrize("use_legacy_exporter", [False, True])
def test_save_compiled_model_omits_unsupported_exporter_option(mocker, tmp_path, use_legacy_exporter):
    # **kwargs alone does not establish that an older release supports the option.
    def save(module, path, **kwargs):
        pass

    library = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt")
    library.logging.errors.return_value = mocker.MagicMock()
    library.save = mocker.create_autospec(save)

    _save_compiled_model(
        mocker.Mock(spec=nn.Module),
        tmp_path / "model.pt2",
        (),
        None,
        5,
        use_legacy_exporter=use_legacy_exporter,
    )

    library.save.assert_called_once()
    assert "use_legacy_exporter" not in library.save.call_args.kwargs
    assert library.save.call_args.kwargs["retrace"] is True


@pytest.mark.parametrize(
    "error",
    [
        SpecViolationError("operator 'aten.copy_.default' is not functional"),
        SpecViolationError("mutation output is missing val"),
        RuntimeError("Only Tensors of floating point and complex dtype can require gradients"),
        OSError("disk full"),
    ],
)
def test_stateful_save_propagates_upstream_failure_without_retry(mocker, tmp_path, error):
    library = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt")
    library.logging.errors.return_value = mocker.MagicMock()
    library.save.side_effect = error

    with pytest.raises(type(error)) as caught:
        _save_compiled_model(mocker.Mock(spec=nn.Module), tmp_path / "model.pt2", (), None, 5, stateful=True)

    assert caught.value is error
    library.save.assert_called_once()


@pytest.fixture
def mock_torch_tensorrt(mocker, model: SimpleModel):
    mock_torch_tensorrt = mocker.Mock()
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
    sample_data: list[Sample],
    torch_device: torch.device,
    tmp_path: Path,
):
    export_mock = mocker.spy(torch.export, "export")

    active_backend = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)

    assert active_backend is backend

    active_backend = cast(TorchTensorRTAotBackend, active_backend)
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
    mock_torch_tensorrt.logging.warnings.return_value = mocker.MagicMock()
    mock_torch_tensorrt.logging.errors.return_value = mocker.MagicMock()
    mock_torch_tensorrt.dynamo.compile = mocker.Mock(return_value=model)

    def save(module, path, **kwargs):
        if not kwargs["retrace"]:
            raise SpecViolationError("direct export failed")
        _fake_torch_tensorrt_save(module, path, **kwargs)

    mock_torch_tensorrt.save = mocker.Mock(side_effect=save)
    mock_torch_tensorrt.load.return_value.module.return_value = model
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt", mock_torch_tensorrt)
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_torch_tensorrt")
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.get_cuda_device", return_value=0)
    mocker.patch.object(TorchTensorRTAotBackend, "_devices", ["cpu", "cuda"])

    backend = TorchTensorRTAotBackend(config=TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig()))
    export_mock = mocker.spy(torch.export, "export")

    backend.build(model, graph_spec, sample_data, device=torch.device("cpu"), cache_dir=tmp_path)

    dynamic_shapes = export_mock.call_args.kwargs["dynamic_shapes"]
    save_kwargs = mock_torch_tensorrt.save.call_args.kwargs
    assert save_kwargs["dynamic_shapes"] is dynamic_shapes["flat_inputs"]
    assert save_kwargs["retrace"] is True
    assert save_kwargs["output_format"] == "exported_program"
    compile_inputs = mock_torch_tensorrt.dynamo.compile.call_args.kwargs["inputs"]
    assert save_kwargs["arg_inputs"] is compile_inputs
    assert compile_inputs == [mock_torch_tensorrt.Input.return_value]
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

    # Check the positional shape specification with real Torch Export. GPU tests
    # exercise Torch-TensorRT converting Input descriptors into example tensors.
    example_args, _example_kwargs = export_mock.spy_return.example_inputs
    reexported = torch.export.export(
        export_mock.spy_return.module(),
        example_args,
        dynamic_shapes=save_kwargs["dynamic_shapes"],
        strict=False,
    )
    path = tmp_path / "public-reexport.pt2"
    torch.export.save(reexported, path)
    loaded = torch.export.load(path).module()
    value = torch.randn(3, 10)
    torch.testing.assert_close(loaded(value)[0], model(value))


def test_build_uses_placement_preserving_module_move(mocker, tmp_path):
    model = SimpleModel().eval()
    move_module = mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.move_module_to_device")
    sample_data = [((torch.randn(1, 10),), {})]
    graph_spec = _graph_spec_from_samples(model, sample_data)
    mock_torch_tensorrt = mocker.Mock()
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

    backend.build(model, graph_spec, sample_data, device=torch.device("cpu"), cache_dir=tmp_path)

    move_module.assert_called_once_with(model, torch.device("cpu"))


@requires_cuda
def test_mock_infer(
    mocker,
    model: SimpleModel,
    graph_spec: GraphSpec,
    sample_data: list[Sample],
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
    # Keep export validation real and bypass TRT compilation with the original model.
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

    backend = cast(TorchTensorRTAotBackend, backend)

    output = backend.infer(*args, **kwargs)
    assert output.shape == (1, 2)

    load.assert_not_called()


@requires_cuda
def test_build_with_custom_precisions(
    model: SimpleModel,
    graph_spec: GraphSpec,
    sample_data: list[Sample],
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
    sample_data: list[Sample],
    torch_device: torch.device,
    tmp_path: Path,
):
    backend = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)

    assert backend._exported_model_artifact == ArtifactPath(tmp_path, Path("exported_model.pt2"))
    assert backend._exported_model_artifact.path.exists()

    args, kwargs = sample_data[0]

    backend = cast(TorchTensorRTAotBackend, backend)
    assert backend._opt_module is not None

    output = backend.infer(*args, **kwargs)
    assert output.shape == (1, 2)


@requires_cuda
def test_full_run_toy_model_linear(torch_device: torch.device, tmp_path: Path):
    model = ToyTorchModel().to(torch_device)
    samples = model.samples(batch_sizes=[2], device=torch_device)
    graph_spec = model.graph_spec(batch_sizes=[2], device=torch_device)

    backend = TorchTensorRTAotBackend()
    backend = backend.build(model, graph_spec, samples, device=torch_device, cache_dir=tmp_path)
    args, kwargs = samples[0]

    backend = cast(TorchTensorRTAotBackend, backend)
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
    backend = backend.build(model, graph_spec, samples, device=torch_device, cache_dir=tmp_path)

    assert backend._opt_module is not None

    output = backend.infer(inputs, apply_relu=True)
    assert output.shape == (2, 5)


@requires_cuda
def test_serialization(torch_device, tmp_path):
    model = ToyTorchModel().to(torch_device)
    samples = model.samples(batch_sizes=[2], device=torch_device)
    graph_spec = model.graph_spec(batch_sizes=[2], device=torch_device)

    backend = TorchTensorRTAotBackend()
    backend = backend.build(model, graph_spec, samples, device=torch_device, cache_dir=tmp_path)

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


@pytest.mark.parametrize("use_legacy_exporter", [None, False, True])
def test_serialization_exporter_config_round_trip(use_legacy_exporter):
    config = TorchTensorRTAotBackendConfig(use_legacy_exporter=use_legacy_exporter)
    restored = TorchTensorRTAotBackendConfig.from_dict(config.to_dict())
    assert restored.use_legacy_exporter is use_legacy_exporter
    assert config.key() == restored.key()
    if use_legacy_exporter is not None:
        assert config.key() != TorchTensorRTAotBackendConfig().key()
