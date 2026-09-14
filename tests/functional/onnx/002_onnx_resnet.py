# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["torchvision"]
# scope = "always"
# allow_failure = false
# ///

"""Run pretrained ResNet-50 ONNX through MaxThroughputStrategy and ONNXRuntimeBackend.

Run with: python -m pytest tests/functional/onnx/002_onnx_resnet.py -q -s
"""

from pathlib import Path

import pytest
import torch
from torchvision.models import ResNet50_Weights, resnet50

from aitune.torch import MaxThroughputStrategy, Module, tune
from aitune.torch.backend import ONNXRuntimeBackend, TensorRTBackend, TensorRTBackendConfig
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module import OnnxModule
from aitune.torch.task.profiling import ProfilingConfig


@torch.inference_mode()
def test_onnx_resnet(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    batch_sizes = [1, 2, 4, 8]
    images = torch.randn(8, 3, 224, 224, generator=torch.Generator().manual_seed(0))
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

    strategy = MaxThroughputStrategy(
        [
            ONNXRuntimeBackend(),
            TensorRTBackend(TensorRTBackendConfig(workspace_size=1 << 30)),
        ],
        profiling_config=ProfilingConfig(batch_sizes=batch_sizes),
    )
    strategy.enable_find_max_batch_size(True)
    strategy.enable_performance_validation(True)
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
        assert isinstance(backend, (ONNXRuntimeBackend, TensorRTBackend))
        results = strategy.perf_validation_results
        assert len(results) == 2  # Both backends must build, validate, and finish profiling.

        best = max(results, key=lambda result: result.metric)
        assert best.passed
        assert best.backend_description == backend.describe()
        assert best.speedup >= 1.0

        selected = next(result for result in strategy.backend_results if result["backend"] == backend.describe())
        selected_batch = selected["selected_batch_size"]
        assert selected_batch in batch_sizes
        print(
            f"Selected {backend.describe()}, batch {selected_batch}; "
            f"baseline: {best.baseline_metric:.2f} images/s; "
            f"tuned: {best.metric:.2f} images/s; speedup: {best.speedup:.2f}x"
        )  # noqa: T201

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
