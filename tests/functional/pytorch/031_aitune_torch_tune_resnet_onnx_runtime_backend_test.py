# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm", "coloredlogs", "flatbuffers", "numpy", "packaging", "protobuf", "sympy"]
# scope = "always"
#
# [[pip_install]]
# packages = ["onnxruntime-gpu"]
# flags = ["--upgrade", "--pre", "--index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"]
# ///


from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from tempfile import TemporaryDirectory

import onnxruntime
import timm
import torch

from aitune.torch import load, save, tune
from aitune.torch.backend.onnx_runtime_backend import ONNXRuntimeBackend, ONNXRuntimeBackendConfig
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

logger = getLogger(__name__)


def _check_exported_artifact(artifact, destination, data, expected_probs):
    artifact.model.export_files(destination)
    assert all((destination.parent / path).is_file() for path in artifact.model.additional_files)
    session = onnxruntime.InferenceSession(str(destination), providers=["CUDAExecutionProvider"])
    assert tuple(node.name for node in session.get_inputs()) == artifact.input_names
    assert tuple(node.name for node in session.get_outputs()) == artifact.output_names
    for batch_size in (1, 2):
        inputs = data.unsqueeze(0).repeat(batch_size, 1, 1, 1).cpu().numpy()
        outputs = session.run(list(artifact.output_names), {artifact.input_names[0]: inputs})
        actual = torch.softmax(torch.from_numpy(outputs[0]), dim=-1)
        expected = expected_probs.cpu().unsqueeze(0).expand(batch_size, -1)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def do_test(backend: ONNXRuntimeBackend):
    device = torch.device("cuda")

    model = timm.create_model("resnet18", pretrained=False)
    model.to(device)
    model.eval()
    data = torch.randn((3, 224, 224), device=device)

    with torch.no_grad():
        out = model(data.unsqueeze(0))
    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    strategy = OneBackendStrategy(backend)
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    module = Module(model, "functional-resnet18-onnx", strategy=strategy)
    tune(
        module,
        data,
        batch_sizes=[2, 1],
        dry_run=False,
        disable_external_logging=False,
        ignore_failing_modules=False,
    )

    out = module(data.unsqueeze(0))
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-3, atol=1e-3)

    artifact = module.artifact()
    assert artifact.max_batch_size == 2
    if backend._config.use_dynamo:
        assert artifact.model.additional_files, "Dynamo export must exercise external weights"

    with TemporaryDirectory(prefix="aitune-onnx-artifact-") as directory:
        root = Path(directory)
        _check_exported_artifact(artifact, root / "export" / "renamed.onnx", data, expected_probs)

        checkpoint = root / "model.ait"
        save(module, checkpoint)
        restored = load(timm.create_model("resnet18", pretrained=False).eval(), checkpoint)
        restored_artifact = restored.artifact()
        assert restored_artifact.model.path != artifact.model.path
        assert restored_artifact.inputs == artifact.inputs
        assert restored_artifact.outputs == artifact.outputs
        assert restored_artifact.runtime == artifact.runtime
        assert restored_artifact.model.format == artifact.model.format
        assert restored_artifact.model.additional_files == artifact.model.additional_files
        _check_exported_artifact(restored_artifact, root / "restored-export" / "model.onnx", data, expected_probs)


def test_tune_resnet_onnx_runtime():
    errors = []
    configs = [
        ONNXRuntimeBackendConfig(),
        ONNXRuntimeBackendConfig(use_dynamo=False),
    ]

    for config in configs:
        try:
            logger.info("Testing config: %s", config.describe() or "default")
            do_test(ONNXRuntimeBackend(config=config))
        except Exception as e:
            logger.error("Error with config %s: %s", config.describe(), e)
            errors.append(f"Error with config {config.describe()!r}: {e}")
        finally:
            MODULE_REGISTRY.clear()

    if errors:
        raise RuntimeError("There were some errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    test_tune_resnet_onnx_runtime()
