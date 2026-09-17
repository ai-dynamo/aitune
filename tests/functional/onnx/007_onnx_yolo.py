# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["huggingface-hub", "pillow"]
# scope = "always"
# allow_failure = false
# [[pip_install]]
# packages = ["onnxruntime-gpu"]
# flags = ["--upgrade", "--index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"]
# ///

"""Download unchanged YOLOv10n ONNX to the HF cache, tune with TensorRT, and save/load.

Source: https://huggingface.co/onnx-community/yolov10n
The published graph requires batch 1 and 640x640 RGB inputs.
Uses the model card street image and compares detections with confidence >= 0.4.
Run: HF_ENDPOINT=https://huggingface.co python -m pytest tests/functional/onnx/007_onnx_yolo.py -q -s
Use --basetemp=/path/to/artifacts to retain the performance report and checkpoint.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageDraw

from aitune.torch import MaxThroughputStrategy, Module, PerformanceValidationMode, load, save, tune
from aitune.torch.backend import ONNXRuntimeBackend, TensorRTBackend, TensorRTBackendConfig
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module import OnnxModule
from aitune.torch.task.profiling import ProfilingConfig


def load_model():
    snapshot = snapshot_download(
        "onnx-community/yolov10n",
        revision="57657320425ee34056408a57ad9d29c4d4815bd8",
        allow_patterns=["onnx/model.onnx", "config.json", "preprocessor_config.json"],
    )
    path = Path(snapshot) / "onnx/model.onnx"
    config = json.loads((Path(snapshot) / "config.json").read_text())
    processor = json.loads((Path(snapshot) / "preprocessor_config.json").read_text())
    return OnnxModule(path), config, processor


def load_image(processor):
    image_path = hf_hub_download("Xenova/transformers.js-docs", "city-streets.jpg", repo_type="dataset")
    with Image.open(image_path) as original:
        image = original.convert("RGB")
    scale = processor["size"]["longest_edge"] / max(image.size)
    new_width, new_height = (int(round(size * scale, 2)) for size in image.size)
    resized = image.resize((new_width, new_height), resample=processor["resample"])
    pixels = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() * processor["rescale_factor"]
    # Transformers.js pads the bottom and right with zeros after rescaling.
    images = torch.zeros((1, 3, processor["pad_size"], processor["pad_size"]))
    images[0, :, :new_height, :new_width] = pixels

    return image, images.cuda(), (new_width, new_height)


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


def inference(module, checkpoint, requests, tuned_output, expected, image, resized_size, config):
    new_width, new_height = resized_size

    module = load(module, checkpoint)
    restored_output = module(images=requests)
    torch.testing.assert_close(restored_output, tuned_output)
    print(f"Load passed. Checkpoint: {checkpoint}")  # noqa: T201

    threshold = 0.4
    predictions = restored_output["output0"][0]
    predictions = predictions[predictions[:, 4] >= threshold]
    reference = expected["output0"][0]
    reference = reference[reference[:, 4] >= threshold]
    assert len(predictions) > 0
    torch.testing.assert_close(predictions, reference, rtol=1e-2, atol=1e-2)

    # draw detections on the image
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    xs, ys = image.width / new_width, image.height / new_height
    for xmin, ymin, xmax, ymax, score, class_id in predictions.tolist():
        box = (xmin * xs, ymin * ys, xmax * xs, ymax * ys)
        bbox = ", ".join(f"{value:.2f}" for value in box)
        label = config["id2label"][str(int(class_id))]
        print(f'Found "{label}" at [{bbox}] with score {score:.4f}.')  # noqa: T201
        draw.rectangle(box, outline="lime", width=3)

        caption = f"{label} {score:.2f}"
        text_width = draw.textlength(caption)
        position = (max(0, min(box[0], image.width - text_width - 4)), max(0, box[1] - 16))

        draw.rectangle(draw.textbbox(position, caption), fill="black")
        draw.text(position, caption, fill="lime")

    annotated_path = Path(__file__).parent / "city-streets-detections.png"
    annotated.save(annotated_path)
    print(f"Annotated image: {annotated_path}")


@torch.inference_mode()
def test_onnx_yolo(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    source, config, processor = load_model()
    module = None
    try:
        image, requests, resized_size = load_image(processor)
        expected = source(images=requests)
        module, checkpoint, tuned_output = tune_and_save(source, requests, tmp_path)
        inference(module, checkpoint, requests, tuned_output, expected, image, resized_size, config)
    finally:
        if module is not None and module.state.name == "TUNED":
            for backend in module.module.backends.values():
                backend._deactivate()
        source.deactivate()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
