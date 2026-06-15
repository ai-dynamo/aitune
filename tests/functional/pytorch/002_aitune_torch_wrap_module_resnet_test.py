# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "always"
# ///


from logging import INFO, basicConfig

import timm
import torch

from aitune.torch import Module, OneBackendStrategy
from aitune.torch.backend import TorchInductorJitBackend


def test_resnet50():
    # given
    device = torch.device("cuda")

    model = timm.create_model("resnet50", pretrained=False)
    model.to(device)
    model.eval()
    data = torch.randn((2, 3, 224, 224), device=device)

    with torch.no_grad():
        out = model(data)
    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    # when
    module = Module(model, "functional-resnet50")

    # then - verify recording
    module(data)
    assert len(module.graph_specs) == 1

    # then - verify tuning
    strategy = OneBackendStrategy(TorchInductorJitBackend())
    strategy.enable_performance_validation(False)
    module.tune(device=device, strategy=strategy, dry_run=False)
    out = module(data)
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_resnet50()
