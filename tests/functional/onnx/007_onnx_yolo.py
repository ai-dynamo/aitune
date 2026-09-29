# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["huggingface-hub"]
# scope = "always"
# allow_failure = false
# [[pip_install]]
# packages = ["onnxruntime-gpu"]
# flags = ["--upgrade", "--index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"]
# ///

"""Download unchanged YOLOv10n ONNX to the HF cache, tune with TensorRT, and save/load.

Source: https://huggingface.co/onnx-community/yolov10n
The published graph requires batch 1 and 640x640 RGB inputs.
Deterministic generated input checks execution and numerical agreement, not detection accuracy.
Run: HF_ENDPOINT=https://huggingface.co python -m pytest tests/functional/onnx/007_onnx_yolo.py -q -s
Use --basetemp=/path/to/artifacts to retain the performance report and checkpoint.
"""

import json
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download

from aitune.torch import MaxThroughputStrategy, Module, PerformanceValidationMode, load, save, tune
from aitune.torch.backend import ONNXRuntimeBackend, TensorRTBackend, TensorRTBackendConfig
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module import OnnxModule
from aitune.torch.task.profiling import ProfilingConfig


def load_model():
    snapshot = snapshot_download(
        "onnx-community/yolov10n",
        revision="57657320425ee34056408a57ad9d29c4d4815bd8",
        allow_patterns=["onnx/model.onnx"],
    )
    path = Path(snapshot) / "onnx/model.onnx"
    return OnnxModule(path)


def sample_input() -> torch.Tensor:
    """Create a reproducible input without an external image asset."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand((1, 3, 640, 640), generator=generator).cuda()


def tune_and_save(source: OnnxModule, requests: torch.Tensor, tmp_path: Path):
    batch_sizes = [1]
    strategy = MaxThroughputStrategy(
        [
            ONNXRuntimeBackend(),
            TensorRTBackend(TensorRTBackendConfig(workspace_size=1 << 30, enable_tf32=False)),
        ],
        profiling_config=ProfilingConfig(batch_sizes=batch_sizes),
    )
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(PerformanceValidationMode.DIAGNOSTIC)
    module = Module(source, "yolov10n", strategy=strategy)
    try:
        tune(
            module,
            DynamicShapeDataset([{"images": image} for image in requests]),
            batch_sizes=batch_sizes,
            device="cuda",
            ignore_failing_modules=False,
        )
        (backend,) = module.module.backends.values()
        results = strategy.perf_validation_results
        assert isinstance(backend, TensorRTBackend)
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
        print(f"yolov10n: {json.dumps(report, indent=2)}")
        tuned_output = {name: value.clone() for name, value in module(images=requests).items()}

        checkpoint = tmp_path / "yolov10n.ait"
        save(module, checkpoint)
        assert checkpoint.stat().st_size > 0
        module.deactivate()
        return module, checkpoint, tuned_output
    except BaseException:
        if module.state.name == "TUNED":
            for backend in module.module.backends.values():
                backend._deactivate()
        raise


def inference(module, checkpoint, requests, tuned_output, expected):
    module = load(module, checkpoint)
    restored_output = module(images=requests)
    torch.testing.assert_close(restored_output, tuned_output)
    print(f"Load passed. Checkpoint: {checkpoint}")  # noqa: T201
    torch.testing.assert_close(restored_output, expected, rtol=1e-2, atol=1e-2)
    assert all(torch.isfinite(output).all() for output in restored_output.values())


@torch.inference_mode()
def test_onnx_yolo(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    source = load_model()
    module = None
    try:
        requests = sample_input()
        expected = source(images=requests)
        module, checkpoint, tuned_output = tune_and_save(source, requests, tmp_path)
        inference(module, checkpoint, requests, tuned_output, expected)
    finally:
        if module is not None and module.state.name == "TUNED":
            for backend in module.module.backends.values():
                backend._deactivate()
        source.deactivate()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
