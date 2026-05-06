# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["transformers", "diffusers"]
# scope = "always"
#
# [[pip_install]]
# packages = ["onnxruntime-gpu"]
# flags = ["--upgrade", "--pre", "--index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"]
# ///


from logging import DEBUG, basicConfig, getLogger

import timm
import torch

from aitune.torch import tune
from aitune.torch.backend.onnx_runtime_backend import (
    ONNXExecutionProvider,
    ONNXRuntimeBackend,
    ONNXRuntimeBackendConfig,
)
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

logger = getLogger(__name__)


def do_test(backend: ONNXRuntimeBackend):
    device = torch.device("cuda")

    model = timm.create_model("resnet18", pretrained=False)
    model.to(device)
    model.eval()
    data = torch.randn((3, 224, 224), device=device)

    with torch.no_grad():
        out = model(data.unsqueeze(0))
    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    module = Module(
        model,
        "functional-resnet18-onnx-trt-ep",
        strategy=OneBackendStrategy(backend, validate_against_baseline=False),
    )
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


def test_tune_resnet_onnx_runtime_tensorrt_ep():
    """Tune resnet18 with ONNXRuntimeBackend using the TensorRT Execution Provider.

    Mirrors test 031 but replaces the default CUDA EP with the TensorRT EP
    (``["TensorrtExecutionProvider", "CUDAExecutionProvider"]``).  The TensorRT EP
    compiles ONNX subgraphs to TensorRT engines inside ONNX Runtime, providing an
    alternative acceleration path distinct from ``TensorRTBackend``.

    Both trace-based and dynamo-based ONNX export are exercised.

    Requires ``onnxruntime-gpu`` built with TensorRT support and a compatible
    TensorRT installation on the system.
    """
    errors = []
    configs = [
        ONNXRuntimeBackendConfig(execution_provider=ONNXExecutionProvider.TENSORRT),
        ONNXRuntimeBackendConfig(use_dynamo=False, execution_provider=ONNXExecutionProvider.TENSORRT),
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
    test_tune_resnet_onnx_runtime_tensorrt_ep()
