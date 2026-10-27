# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["torchvision", "monai"]
# scope = "always"
# allow_failure = false
# [[pip_install]]
# packages = ["onnxruntime-gpu"]
# flags = ["--upgrade", "--index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"]
# ///

"""Export three bring-up models, tune ONNX, report throughput, and save/load.

UNet configurations come from trtre/models/kpi/clara_models.py. EfficientNet-B0
uses torchvision pretrained weights instead of the NVIDIA torch.hub implementation
in scripts/001_efficientnet.py. UNets use seeded random weights, as in the registry.
Synthetic inputs check execution and numerical agreement, not task accuracy.
Run: python -m pytest tests/functional/onnx/005_onnx_bringup_models.py -q -s
Use --basetemp=/path/to/artifacts to choose where ONNX, reports, and checkpoints stay.
"""

import json
from pathlib import Path

import pytest
import torch
from monai.networks.nets import UNet
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from aitune.torch import MaxThroughputStrategy, Module, PerformanceValidationMode, load, save, tune
from aitune.torch.backend import ONNXRuntimeBackend, TensorRTBackend, TensorRTBackendConfig
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module import OnnxModule
from aitune.torch.task.profiling import ProfilingConfig


@pytest.mark.parametrize("model_name", ["efficientnet_b0", "2d_unet", "3d_unet"])
@torch.inference_mode()
def test_onnx_bringup_model(tmp_path: Path, model_name: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        if model_name == "efficientnet_b0":
            reference = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).eval()
            shape = (3, 256, 256)
        else:
            spatial_dims = 2 if model_name == "2d_unet" else 3
            reference = UNet(
                spatial_dims=spatial_dims,
                in_channels=1,
                out_channels=1 if spatial_dims == 2 else 2,
                channels=[16, 32, 64, 128, 256],
                strides=[2, 2, 2, 2],
                num_res_units=2,
                **({"norm": "batch"} if spatial_dims == 3 else {}),
            ).eval()
            shape = (1, *([128] * spatial_dims))

    batch_sizes = [1, 2] if model_name == "3d_unet" else [1, 2, 4]
    images = torch.randn(max(batch_sizes), *shape, generator=torch.Generator().manual_seed(0))

    path = tmp_path / f"{model_name}.onnx"
    torch.onnx.export(
        reference,
        (images[:1],),
        path,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
        external_data=False,
    )
    reference_output = reference(images[:1])
    del reference

    source = OnnxModule(path)
    module = None
    try:
        requests = images.cuda()
        expected = {batch: source(images=requests[:batch]) for batch in batch_sizes}
        torch.testing.assert_close(expected[1]["output"].cpu(), reference_output, rtol=1e-2, atol=1e-2)
        strategy = MaxThroughputStrategy(
            [
                ONNXRuntimeBackend(),
                TensorRTBackend(TensorRTBackendConfig(workspace_size=1 << 30)),
            ],
            profiling_config=ProfilingConfig(batch_sizes=batch_sizes),
        )
        strategy.enable_find_max_batch_size(False)
        strategy.enable_performance_validation(PerformanceValidationMode.DIAGNOSTIC)
        module = Module(source, model_name, strategy=strategy)
        tune(
            module,
            DynamicShapeDataset([{"images": image} for image in requests]),
            batch_sizes=batch_sizes,
            device="cuda",
            ignore_failing_modules=False,
        )
        (backend,) = module.module.backends.values()
        results = strategy.perf_validation_results
        assert len(results) == 2
        assert all(result["success"] for result in strategy.backend_results)
        assert all(result.metric > 0 and result.baseline_metric > 0 for result in results)
        assert max(results, key=lambda result: result.metric).backend_description == backend.describe()
        report = [
            {
                "backend": result.backend_description,
                "baseline_samples_per_second": result.baseline_metric,
                "tuned_samples_per_second": result.metric,
                "speedup": result.speedup,
                "selected": result.backend_description == backend.describe(),
            }
            for result in results
        ]
        (tmp_path / "performance.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{model_name}: {json.dumps(report, indent=2)}")
        for batch in batch_sizes:
            torch.testing.assert_close(module(images=requests[:batch]), expected[batch], rtol=1e-2, atol=1e-2)

        checkpoint = tmp_path / f"{model_name}.ait"
        save(module, checkpoint)
        assert checkpoint.stat().st_size > 0
        module.deactivate()
        module = load(module, checkpoint)
        for batch in batch_sizes:
            torch.testing.assert_close(module(images=requests[:batch]), expected[batch], rtol=1e-2, atol=1e-2)
        print(f"Artifacts: {tmp_path}")
    finally:
        if module is not None and module.state.name == "TUNED":
            for backend in module.module.backends.values():
                backend._deactivate()
        source.deactivate()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
