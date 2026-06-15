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

import timm
import torch

from aitune.torch import tune
from aitune.torch.backend.onnx_runtime_backend import ONNXRuntimeBackend
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

logger = getLogger(__name__)


def test_tune_resnet_onnx_runtime_dynamic_shapes():
    """Tune resnet18 with two data samples of different spatial shapes using ONNXRuntimeBackend.

    Passing (3, 224, 224) and (3, 256, 256) with batch_sizes=[1] means each sample
    is processed individually so tensors of different spatial sizes are never stacked.
    The recording module detects:
        axis 2 → dim2  (H: 224 vs 256)
        axis 3 → dim3  (W: 224 vs 256)

    For trace-based export these map to dynamic_axes so the same .onnx model handles
    both spatial sizes.  Inference is verified for both shapes against eager outputs.

    Note: resnet18 supports variable spatial input sizes via its adaptive average
    pooling layer.
    """
    device = torch.device("cuda")

    model = timm.create_model("resnet18", pretrained=False)
    model.to(device)
    model.eval()

    data_224 = torch.randn((3, 224, 224), device=device)
    data_256 = torch.randn((3, 256, 256), device=device)

    with torch.no_grad():
        expected_224 = torch.nn.functional.softmax(model(data_224.unsqueeze(0))[0], dim=0)
        expected_256 = torch.nn.functional.softmax(model(data_256.unsqueeze(0))[0], dim=0)

    try:
        strategy = OneBackendStrategy(ONNXRuntimeBackend())
        strategy.enable_performance_validation(False)
        strategy.enable_find_max_batch_size(False)
        module = Module(model, "functional-resnet18-onnx-dynamic", strategy=strategy)

        # batch_sizes=[1] prevents cross-sample stacking so the two spatial sizes
        # are seen as distinct shapes, triggering dynamic axis detection.
        tune(
            module,
            [data_224, data_256],
            batch_sizes=[1],
            dry_run=False,
            disable_external_logging=False,
            ignore_failing_modules=False,
        )

        actual_224 = torch.nn.functional.softmax(module(data_224.unsqueeze(0))[0], dim=0)
        torch.testing.assert_close(actual_224, expected_224, rtol=1e-3, atol=1e-3)

        actual_256 = torch.nn.functional.softmax(module(data_256.unsqueeze(0))[0], dim=0)
        torch.testing.assert_close(actual_256, expected_256, rtol=1e-3, atol=1e-3)

    finally:
        MODULE_REGISTRY.clear()


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    test_tune_resnet_onnx_runtime_dynamic_shapes()
