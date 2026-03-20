# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "always"
# ///


import timm
import torch

from aitune.torch.backend.torch_inductor_jit_backend import TorchInductorJitBackend
from aitune.torch.module.wrapper_module import Module
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy


def test_resnet50():
    # given
    device = torch.device("cuda")

    model = timm.create_model("resnet50", pretrained=False)
    model.to(device)
    model.eval()
    data = torch.randn((2, 3, 224, 224), device=device)

    def pre_hook(module, input):  # noqa: A002
        # this actually inject data into the model
        return data

    def post_hook(module, input, output):  # noqa: A002
        # this extract max detected element
        return torch.argmax(output, dim=1)

    model.register_forward_pre_hook(pre_hook)
    model.register_forward_hook(post_hook)

    with torch.no_grad():
        expected_arg_max = model()  # notice: not argument - it will be added by pre hook

    # when
    module = Module(model, "functional-resnet50")

    # then - verify recording
    module()  # notice: not argument - it will be added by pre hook
    assert len(module.graph_specs) == 1

    # then - verify tuning
    strategy = OneBackendStrategy(TorchInductorJitBackend())
    module.tune(device=device, strategy=strategy, dry_run=False)
    actual_arg_max = module(data)
    torch.testing.assert_close(actual_arg_max, expected_arg_max, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    test_resnet50()
