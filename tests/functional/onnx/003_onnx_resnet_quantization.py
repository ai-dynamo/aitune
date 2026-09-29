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

"""Compare INT8 max and entropy calibration on pretrained ONNX ResNet-50.

Synthetic inputs test execution and numerical drift, not ImageNet accuracy.
Run: python -m pytest tests/functional/onnx/003_onnx_resnet_quantization.py -q -s
"""

from pathlib import Path

import onnx
import pytest
import torch
from torchvision.models import ResNet50_Weights, resnet50

from aitune.torch import Module, OneBackendStrategy, config, tune
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig
from aitune.torch.backend.tensorrt.onnx_quantization import CalibrationMethod, ONNXQuantizationConfig
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module import OnnxModule
from aitune.torch.task.profiling import ProfilingConfig


@pytest.mark.parametrize("calibration_method", ["max", "entropy"])
@torch.inference_mode()
def test_onnx_resnet_quantization(tmp_path: Path, monkeypatch, calibration_method: CalibrationMethod) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    batch_sizes = [1, 2, 4, 8]
    monkeypatch.setattr(config, "max_num_samples_stored", 16)
    generator = torch.Generator().manual_seed(0)
    calibration = torch.randn(16, 3, 224, 224, generator=generator)
    requests = torch.randn(8, 3, 224, 224, generator=generator).cuda()
    reference = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    path = tmp_path / "resnet50.onnx"
    torch.onnx.export(
        reference,
        (calibration[:1],),
        path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
        external_data=False,
    )
    del reference

    source = OnnxModule(path)
    module = None
    try:
        expected = {batch: source(images=requests[:batch])["logits"] for batch in batch_sizes}
        strategy = OneBackendStrategy(
            TensorRTBackend(
                TensorRTBackendConfig(
                    workspace_size=1 << 30,
                    quantization_config=ONNXQuantizationConfig("int8", calibration_method=calibration_method),
                )
            ),
            profiling_config=ProfilingConfig(batch_sizes=batch_sizes),
        )
        # A quantization regression must exercise the quantized engine even if it is slower.
        strategy.enable_performance_validation(False)
        strategy.enable_find_max_batch_size(False)
        module = Module(source, f"onnx-resnet50-int8-{calibration_method}", strategy=strategy)
        tune(
            module,
            DynamicShapeDataset([{"images": image} for image in calibration.cuda()]),
            batch_sizes=batch_sizes,
            device="cuda",
            ignore_failing_modules=False,
        )
        (backend,) = module.module.backends.values()
        assert isinstance(backend, TensorRTBackend)

        quantized = onnx.load(backend._engine_artifact.root / "onnx_ptq" / "model.onnx")
        node_types = [node.op_type for node in quantized.graph.node]
        assert "QuantizeLinear" in node_types
        assert "DequantizeLinear" in node_types
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
            assert actual.is_cuda and torch.isfinite(actual).all()

            cosine = torch.nn.functional.cosine_similarity(actual, expected[batch]).min().item()
            relative_error = ((actual - expected[batch]).norm() / expected[batch].norm()).item()
            # Quantization allows bounded drift; compare logits, not only softmax probabilities.
            assert cosine >= 0.99
            assert relative_error <= 0.1
            print(f"INT8 {calibration_method}, batch {batch}: cosine={cosine:.5f}, relative L2={relative_error:.5f}")  # noqa: T201
    finally:
        if module is not None:
            module.deactivate()
        source.deactivate()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
