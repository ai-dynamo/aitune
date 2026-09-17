# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["torchvision"]
# scope = "always"
# allow_failure = false
# [[pip_install]]
# packages = ["onnxruntime-gpu"]
# flags = ["--upgrade", "--index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"]
# ///

"""Run pretrained ResNet-50 ONNX through OneBackendStrategy and ONNXRuntimeBackend.

Run with: python -m pytest tests/functional/onnx/001_onnx_resnet.py -q -s
"""

from pathlib import Path

import pytest
import torch
from torchvision.models import ResNet50_Weights, resnet50

from aitune.torch import Module, OneBackendStrategy, tune
from aitune.torch.backend import ONNXRuntimeBackend
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module import OnnxModule
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.utils.module import offload


@pytest.mark.parametrize("offload_device", ["cpu", "meta"])
@pytest.mark.parametrize("nested", [False, True])
@torch.inference_mode()
def test_onnx_resnet(tmp_path: Path, offload_device: str, nested: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    batch_sizes = [1, 2, 4]
    images = torch.randn(4, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    reference = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    path = tmp_path / "resnet50.onnx"
    torch.onnx.export(
        reference,
        (images[:1],),
        path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
        external_data=False,
    )
    source = OnnxModule(path)
    module = None
    try:
        requests = images.cuda()
        expected = {batch: source(images=requests[:batch])["logits"] for batch in batch_sizes}
        torch.testing.assert_close(expected[1].cpu(), reference(images[:1]), rtol=1e-2, atol=1e-2)
        del reference

        assert "CUDAExecutionProvider" in source._session.get_providers()
        target = torch.nn.ModuleDict({"child": source}) if nested else source
        offload(target, device=offload_device)
        assert source._session.get_providers() == ["CPUExecutionProvider"]
        with pytest.raises(AssertionError, match="call offload"):
            source(images=requests[:1])

        actual_cpu = source(images=images[:1])["logits"]
        assert actual_cpu.device.type == "cpu"

        torch.testing.assert_close(actual_cpu, expected[1].cpu(), rtol=1e-2, atol=1e-2)
        assert source._session.get_providers() == ["CPUExecutionProvider"]

        offload(target, device="cuda")
        assert "CUDAExecutionProvider" in source._session.get_providers()
        with pytest.raises(AssertionError, match="call offload"):
            source(images=images[:1])

        torch.testing.assert_close(source(images=requests[:1])["logits"], expected[1], rtol=1e-2, atol=1e-2)

        strategy = OneBackendStrategy(ONNXRuntimeBackend(), profiling_config=ProfilingConfig(batch_sizes=batch_sizes))
        # Both paths use ORT CUDA; this test checks integration, with no speedup requirement.
        strategy.enable_performance_validation(False)
        module = Module(source, "onnx-resnet50", strategy=strategy)
        tune(
            module,
            DynamicShapeDataset([{"images": image} for image in requests]),
            batch_sizes=batch_sizes,
            device="cuda",
            ignore_failing_modules=False,
        )
        (backend,) = module.module.backends.values()
        assert isinstance(backend, ONNXRuntimeBackend)
        assert backend._onnx_model_artifact.path == path
        (graph_spec,) = module.graph_specs
        (input_spec,) = graph_spec.input_spec.tensor_specs
        (output_spec,) = graph_spec.output_spec.tensor_specs
        assert input_spec.name == "images"
        assert output_spec.name == "logits"
        assert input_spec.min_shape == [1, 3, 224, 224]
        assert input_spec.max_shape == [max(batch_sizes), 3, 224, 224]
        assert source._session is None
        for batch in batch_sizes:
            actual = module(images=requests[:batch])["logits"]
            assert actual.is_cuda
            torch.testing.assert_close(actual, expected[batch], rtol=1e-2, atol=1e-2)
    finally:
        if module is not None:
            module.deactivate()
        source.deactivate()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
