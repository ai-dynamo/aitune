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
A pinned dog image checks detection agreement and checkpoint round-trip across runtimes.
Run: HF_ENDPOINT=https://huggingface.co python -m pytest tests/functional/onnx/007_onnx_yolo.py -q -s
Use --basetemp=/path/to/artifacts to retain the performance report and checkpoint.
"""

import json
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from PIL import Image

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
    image_path = Path(__file__).resolve().parents[3] / "examples/ResNet/dog.webp"
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((640, 640))
        image_tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(640, 640, 3)
    return image_tensor.permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255).cuda()


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


def test_detection_matching_ignores_row_order() -> None:
    expected = torch.tensor([[10, 10, 30, 30, 0.9, 1], [60, 60, 100, 100, 0.8, 2]])
    actual = torch.tensor([[60, 60, 100, 100, 0.79, 2], [10, 10, 31, 30, 0.91, 1]])

    assert _match_detections(actual, expected) == (2, 2, 2)


def inference(module, checkpoint, requests, tuned_output, expected):
    module = load(module, checkpoint)
    restored_output = module(images=requests)
    torch.testing.assert_close(restored_output, tuned_output)
    print(f"Load passed. Checkpoint: {checkpoint}")  # noqa: T201
    # output0 rows are [x1, y1, x2, y2, score, class_id]; box coordinates are in input pixels.
    assert restored_output.keys() == expected.keys()
    for name, restored in restored_output.items():
        reference = expected[name]
        assert restored.shape == reference.shape
        assert torch.isfinite(restored).all()
        assert torch.isfinite(reference).all()
    assert restored_output["output0"].shape == (1, 300, 6)
    matched, expected_count, restored_count = _match_detections(restored_output["output0"][0], expected["output0"][0])
    assert expected_count > 0, "Reference produced no confident detections"
    assert restored_count > 0, "TensorRT produced no confident detections"
    assert matched / expected_count >= 0.9
    assert matched / restored_count >= 0.9


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


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2:4] - boxes[:, :2]).clamp(min=0).prod(-1)


def _match_detections(
    actual: torch.Tensor,
    expected: torch.Tensor,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    score_tolerance: float = 0.05,
) -> tuple[int, int, int]:
    """Greedily match confident detections of the same class by IoU, highest IoU first."""
    actual = actual[actual[:, 4] >= confidence_threshold].cpu()
    expected = expected[expected[:, 4] >= confidence_threshold].cpu()
    top_left = torch.maximum(expected[:, None, :2], actual[None, :, :2])
    bottom_right = torch.minimum(expected[:, None, 2:4], actual[None, :, 2:4])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    iou = intersection / (_box_area(expected)[:, None] + _box_area(actual)[None, :] - intersection)
    same_class = expected[:, None, 5] == actual[None, :, 5]
    close_score = (expected[:, None, 4] - actual[None, :, 4]).abs() <= score_tolerance
    iou = iou.where(same_class & close_score & (iou >= iou_threshold), 0)
    matched = 0
    while iou.numel() and iou.max() > 0:
        expected_index, actual_index = divmod(int(iou.argmax()), iou.shape[1])
        iou[expected_index], iou[:, actual_index] = 0, 0
        matched += 1
    return matched, len(expected), len(actual)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
