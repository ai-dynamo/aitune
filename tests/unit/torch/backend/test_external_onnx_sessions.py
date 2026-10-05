# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve externally owned ONNX sessions while backend resources are built and released."""

from copy import deepcopy
from types import SimpleNamespace

import onnx
import pytest
import torch
from onnx import TensorProto, helper

from aitune.exceptions import AITuneUserInputError
from aitune.torch.backend import ONNXRuntimeBackend, TensorRTBackend, TensorRTBackendConfig
from aitune.torch.backend.backend import BackendState
from aitune.torch.backend.tensorrt.torch_quantization import TorchQuantizationConfig
from aitune.torch.module.onnx_module import OnnxModule
from aitune.torch.utils.module import (
    is_externally_managed_module,
    register_externally_managed_module,
    unregister_externally_managed_module,
)


@pytest.fixture
def onnx_path(tmp_path):
    path = tmp_path / "add.onnx"
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["y"])],
        "double",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), path)
    return path


@pytest.fixture(params=["external", "aitune", "copy"])
def source_registration(request, onnx_path):
    owner = OnnxModule(onnx_path)
    if request.param != "aitune":
        register_externally_managed_module(owner)
    # Copies are made before activation: live ORT sessions cannot be deep-copied.
    source = deepcopy(owner) if request.param == "copy" else owner
    sample = torch.ones(1, 3)
    torch.testing.assert_close(owner(sample)["y"], sample * 2)
    if source is not owner:
        torch.testing.assert_close(source(sample)["y"], sample * 2)
    registration = SimpleNamespace(
        source=source,
        owner=owner,
        session=source._session,
        owner_session=owner._session,
        device=source._device,
        external=request.param == "external",
        sample=sample,
    )
    assert is_externally_managed_module(source) == registration.external
    try:
        yield registration
    finally:
        unregister_externally_managed_module(owner)
        source.deactivate()
        if owner is not source:
            owner.deactivate()


@pytest.fixture
def graph_spec():
    return SimpleNamespace(output_spec=SimpleNamespace(tensor_data=[(None, SimpleNamespace(name="y"))]))


def _assert_source_session(registration):
    if registration.external:
        assert registration.source._session is registration.session
        assert registration.source._device == registration.device
    else:
        assert registration.source._session is None
    if registration.owner is not registration.source:
        assert registration.owner._session is registration.owner_session
        assert registration.owner._device == registration.device


def _assert_owner_can_release_session(registration):
    if registration.external:
        torch.testing.assert_close(registration.source(registration.sample)["y"], registration.sample * 2)
        assert registration.source._session is registration.session
        registration.source.deactivate()
        assert registration.source._session is None
        assert registration.source._device is None


def test_onnx_runtime_build_respects_source_session_ownership(source_registration, graph_spec, tmp_path, mocker):
    backend = ONNXRuntimeBackend()
    # Exercise real ORT sessions and cleanup with CPU providers; the production backend targets CUDA.
    mocker.patch.object(backend, "_get_execution_providers", return_value=["CPUExecutionProvider"])
    try:
        backend.build(source_registration.source, graph_spec, None, torch.device("cuda", 0), tmp_path)
        assert backend.state is BackendState.ACTIVE
        assert backend._session is not source_registration.session
        (output,) = backend._session.run(None, {"x": source_registration.sample.numpy()})
        torch.testing.assert_close(torch.from_numpy(output), source_registration.sample * 2)
        _assert_source_session(source_registration)
        backend.deactivate()
        assert backend._session is None
        _assert_source_session(source_registration)
        _assert_owner_can_release_session(source_registration)
    finally:
        backend._deactivate()


@pytest.mark.parametrize("failure", ["metadata", "activation"])
def test_onnx_runtime_build_failure_respects_source_session_ownership(
    source_registration, graph_spec, tmp_path, mocker, failure
):
    backend = ONNXRuntimeBackend()
    error = RuntimeError("ONNX backend build failed")
    if failure == "metadata":
        mocker.patch("aitune.torch.backend.onnx_runtime_backend.onnx.load", side_effect=error)
    else:
        mocker.patch.object(backend, "_activate", side_effect=error)
    with pytest.raises(RuntimeError, match="ONNX backend build failed"):
        backend.build(source_registration.source, graph_spec, None, torch.device("cuda", 0), tmp_path)
    _assert_source_session(source_registration)
    backend._deactivate()
    _assert_source_session(source_registration)


def test_tensorrt_build_respects_source_session_ownership(source_registration, graph_spec, tmp_path, mocker):
    backend = TensorRTBackend()
    mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.cuda_set_device")
    # Stub GPU engine construction/activation while retaining real source handling and backend cleanup.
    engine_path = tmp_path / "model.plan"
    engine_path.write_bytes(b"test-engine")
    mocker.patch.object(backend, "_build_standard", return_value=engine_path)
    mocker.patch.object(backend, "_activate", side_effect=lambda: setattr(backend, "_trt_runtime", object()))
    try:
        backend.build(source_registration.source, graph_spec, None, torch.device("cuda", 0), tmp_path)
        assert backend.state is BackendState.ACTIVE
        assert backend._trt_runtime is not None
        _assert_source_session(source_registration)
        backend.deactivate()
        assert backend._trt_runtime is None
        _assert_source_session(source_registration)
        _assert_owner_can_release_session(source_registration)
    finally:
        backend._deactivate()


def test_tensorrt_rejected_build_respects_source_session_ownership(source_registration, tmp_path, mocker):
    backend = TensorRTBackend(TensorRTBackendConfig(quantization_config=TorchQuantizationConfig()))
    mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.cuda_set_device")
    with pytest.raises(AITuneUserInputError, match="Torch quantization requires a Torch module"):
        backend.build(source_registration.source, None, None, torch.device("cuda", 0), tmp_path)
    _assert_source_session(source_registration)
    backend._deactivate()
    _assert_source_session(source_registration)
