# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["torchvision"]
# scope = "always"
# allow_failure = false
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
from aitune.torch.utils.memory import cleanup_memory
from aitune.torch.utils.module import offload


@pytest.mark.parametrize("offload_device", ["cpu", "meta"])
@torch.inference_mode()
def test_onnx_resnet(tmp_path: Path, offload_device: str) -> None:
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
    requests = images.cuda()
    expected = {batch: source(images=requests[:batch])["logits"] for batch in batch_sizes}
    torch.testing.assert_close(expected[1].cpu(), reference(images[:1]), rtol=1e-2, atol=1e-2)
    del reference

    assert "CUDAExecutionProvider" in source._session.get_providers()
    cleanup_memory()
    free_before, _ = torch.cuda.mem_get_info()
    # Include a parent module: offload must also switch nested ONNX sessions.
    offload(source, device=offload_device)
    assert source._session.get_providers() == ["CPUExecutionProvider"]
    free_after, _ = torch.cuda.mem_get_info()
    # ORT allocations are outside PyTorch's allocator; ResNet-50 weights exceed 90 MiB.
    assert free_after - free_before > 50 * 1024**2
    with pytest.raises(AssertionError, match="call offload"):
        source(images=requests[:1])

    actual_cpu = source(images=images[:1])["logits"]
    assert actual_cpu.device.type == "cpu"

    torch.testing.assert_close(actual_cpu, expected[1].cpu(), rtol=1e-2, atol=1e-2)
    assert source._session.get_providers() == ["CPUExecutionProvider"]

    offload(source, device="cuda")
    assert "CUDAExecutionProvider" in source._session.get_providers()
    with pytest.raises(AssertionError, match="call offload"):
        source(images=images[:1])

    torch.testing.assert_close(source(images=requests[:1])["logits"], expected[1], rtol=1e-2, atol=1e-2)

    strategy = OneBackendStrategy(ONNXRuntimeBackend(), profiling_config=ProfilingConfig(batch_sizes=batch_sizes))
    # Both paths use ORT CUDA; this test checks integration, with no speedup requirement.
    strategy.enable_performance_validation(False)
    module = Module(source, "onnx-resnet50", strategy=strategy)
    try:
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
        assert source._session is None
        for batch in batch_sizes:
            actual = module(images=requests[:batch])["logits"]
            assert actual.is_cuda
            torch.testing.assert_close(actual, expected[batch], rtol=1e-2, atol=1e-2)
    finally:
        module.deactivate()
        source.deactivate()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
