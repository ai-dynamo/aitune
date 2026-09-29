# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
# scope = "always"
# allow_failure = true
# ///


from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from tempfile import TemporaryDirectory

import timm
import torch

from aitune.records import DeploymentArtifact, RuntimeConfig
from aitune.torch import tune
from aitune.torch.backend.torch_inductor_aot_backend import TorchInductorAotBackend, TorchInductorAotBackendConfig
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

logger = getLogger(__name__)


def do_test(backend: TorchInductorAotBackend):
    # given
    device = torch.device("cuda")

    model = timm.create_model("resnet18", pretrained=False)
    model.to(device)
    model.eval()
    data = torch.randn((3, 224, 224), device=device)

    with torch.no_grad():
        expected_output = model(data.unsqueeze(0))
    expected_probs = torch.nn.functional.softmax(expected_output[0], dim=0)

    strategy = OneBackendStrategy(backend)
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    module = Module(model, "functional-resnet18", strategy=strategy)
    # when
    tune(module, data, batch_sizes=[2, 1], dry_run=False, disable_external_logging=False)
    # then - verify tuning
    out = module(data.unsqueeze(0))
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-3, atol=1e-4)

    artifact = module.artifact()
    assert isinstance(artifact, DeploymentArtifact)
    assert artifact.model.format == "pt2"
    assert artifact.model.metadata == {"structured_call": False}
    assert artifact.model.additional_files == ()
    assert artifact.runtime == RuntimeConfig(name="aotinductor")
    assert artifact.input_names == ("INPUT__0",)
    assert artifact.output_names == ("OUTPUT__0",)
    assert artifact.inputs[0].min_shape == (1, 3, 224, 224)
    assert artifact.inputs[0].max_shape == (2, 3, 224, 224)
    assert artifact.outputs[0].min_shape == (1, 1000)
    assert artifact.outputs[0].max_shape == (2, 1000)
    assert artifact.max_batch_size == 2

    with TemporaryDirectory() as directory:
        destination = Path(directory) / "renamed.pt2"
        assert artifact.model.export_files(destination) == destination
        assert tuple(Path(directory).iterdir()) == (destination,)
        runner = torch._inductor.aoti_load_package(str(destination))
        with torch.no_grad():
            for batch_size in (1, 2):
                inputs = data.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                torch.testing.assert_close(runner(inputs), expected_output.repeat(batch_size, 1), rtol=1e-3, atol=1e-4)
        del runner


def test_tune_resnet_torch_inductor_aot():
    errors = []
    configs = [
        TorchInductorAotBackendConfig(),
        TorchInductorAotBackendConfig(inductor_configs={"max_autotune": True}),
    ]

    for config in configs:
        try:
            logger.info("Testing config: %s", config.describe() or "default")
            do_test(TorchInductorAotBackend(config=config))
        except Exception as e:
            logger.error("Error with config %s: %s", config.describe(), e)
            errors.append(f"Error with config {config.describe()!r}: {e}")
        finally:
            MODULE_REGISTRY.clear()

    if errors:
        raise RuntimeError("There were some errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    test_tune_resnet_torch_inductor_aot()
