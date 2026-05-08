# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
# scope = "always"
# allow_failure = true
# ///


from logging import DEBUG, basicConfig, getLogger

import timm
import torch

from aitune.torch import tune
from aitune.torch.backend.torch_inductor_aot_backend import TorchInductorAotBackend
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

logger = getLogger(__name__)


def test_tune_resnet_torch_inductor_aot_dynamic_shapes():
    """Tune resnet18 with two data samples of different spatial shapes.

    Passing (3, 224, 224) and (3, 256, 256) with batch_sizes=[1] means each sample
    is fetched individually (no cross-sample stacking) and the recording module sees:
        (1, 3, 224, 224)
        (1, 3, 256, 256)

    After update_shapes_seen this yields two dynamic spatial axes:
        axis 2  → dim2   (H: 224 vs 256, non-proportional to batch)
        axis 3  → dim3   (W: 224 vs 256, non-proportional to batch)

    The AOT Inductor backend compiles the model with both axes marked as dynamic
    via torch.export.Dim instances.  Inference is then verified for both spatial
    sizes against the original eager outputs.

    Note: resnet18 supports variable spatial input sizes thanks to its adaptive
    average pooling layer.
    """
    device = torch.device("cuda")

    model = timm.create_model("resnet18", pretrained=False)
    model.to(device)
    model.eval()

    data_224 = torch.randn((3, 224, 224), device=device)
    data_256 = torch.randn((3, 256, 256), device=device)

    # Capture expected outputs before tuning (eager, no optimization)
    with torch.no_grad():
        expected_224 = torch.nn.functional.softmax(model(data_224.unsqueeze(0))[0], dim=0)
        expected_256 = torch.nn.functional.softmax(model(data_256.unsqueeze(0))[0], dim=0)

    try:
        strategy = OneBackendStrategy(TorchInductorAotBackend())
        strategy.enable_validate_against_baseline(False)
        strategy.enable_find_max_batch_size(False)
        module = Module(model, "functional-resnet18-dynamic", strategy=strategy)

        # batch_sizes=[1] ensures each sample is processed individually so that
        # tensors of different spatial sizes are never stacked together.
        # The two shapes (224 and 256) cause H and W axes to be detected as dynamic.
        tune(
            module,
            [data_224, data_256],
            batch_sizes=[1],
            dry_run=False,
            disable_external_logging=False,
        )

        # Verify correctness for 224×224
        actual_224 = torch.nn.functional.softmax(module(data_224.unsqueeze(0))[0], dim=0)
        torch.testing.assert_close(actual_224, expected_224, rtol=1e-3, atol=1e-4)

        # Verify correctness for 256×256 (different spatial size, same compiled model)
        actual_256 = torch.nn.functional.softmax(module(data_256.unsqueeze(0))[0], dim=0)
        torch.testing.assert_close(actual_256, expected_256, rtol=1e-3, atol=1e-4)

    finally:
        MODULE_REGISTRY.clear()


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    test_tune_resnet_torch_inductor_aot_dynamic_shapes()
