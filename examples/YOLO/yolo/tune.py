# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune YOLOv10n from Torch or ONNX and save an AITune checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from aitune.torch import MaxThroughputStrategy, Module, PerformanceValidationMode, save, tune
from aitune.torch.backend import ONNXRuntimeBackend, TensorRTBackend, TensorRTBackendConfig
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module import OnnxModule
from aitune.torch.task.profiling import ProfilingConfig

ONNX_REVISION = "57657320425ee34056408a57ad9d29c4d4815bd8"


def _processor() -> dict:
    """Read the preprocessing settings used by the ONNX functional test."""
    config_path = hf_hub_download("onnx-community/yolov10n", "preprocessor_config.json", revision=ONNX_REVISION)
    return json.loads(Path(config_path).read_text())


def _onnx_source(path: Path | None) -> OnnxModule:
    """Load an existing ONNX graph or the pinned graph from the functional test."""
    model_path = path or Path(hf_hub_download("onnx-community/yolov10n", "onnx/model.onnx", revision=ONNX_REVISION))
    return OnnxModule(model_path)


def _torch_source(weights: str) -> torch.nn.Module:
    """Expose the model tensor, without Ultralytics' predictor or image postprocessing."""
    from ultralytics import YOLO

    model = YOLO(weights).model.eval().cuda()
    # The YOLOv10 detection head returns a single [batch, detections, 6] tensor in export mode.
    model.model[-1].export = True
    return model


def _image(processor: dict) -> torch.Tensor:
    """Use the model-card street image and preprocessing from the ONNX functional test."""
    image_path = hf_hub_download("Xenova/transformers.js-docs", "city-streets.jpg", repo_type="dataset")
    with Image.open(image_path) as original:
        image = original.convert("RGB")
    scale = processor["size"]["longest_edge"] / max(image.size)
    new_width, new_height = (int(round(size * scale, 2)) for size in image.size)
    resized = image.resize((new_width, new_height), resample=processor["resample"])
    pixels = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() * processor["rescale_factor"]
    images = torch.zeros((1, 3, processor["pad_size"], processor["pad_size"]))
    images[0, :, :new_height, :new_width] = pixels
    return images.cuda()


@torch.inference_mode()
def main() -> None:
    """Tune one source format and save its deployment-capable artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("torch", "onnx"), required=True)
    parser.add_argument("--target", choices=("triton",), default="triton")
    parser.add_argument("--onnx-path", type=Path, help="Existing YOLOv10n ONNX file; defaults to the pinned model")
    parser.add_argument("--torch-weights", default="yolov10n.pt", help="Ultralytics YOLOv10n weights")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults to this example's artifacts")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This prototype requires a CUDA GPU")
    if args.source == "torch" and args.onnx_path is not None:
        parser.error("--onnx-path can only be used with --source onnx")
    output_dir = args.output_dir or Path(__file__).resolve().parents[1] / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = _processor()
    source = _onnx_source(args.onnx_path) if args.source == "onnx" else _torch_source(args.torch_weights)
    input_name = "images" if args.source == "onnx" else "x"
    images = _image(processor)
    expected = source(**{input_name: images})

    strategy = MaxThroughputStrategy(
        [
            ONNXRuntimeBackend(),
            TensorRTBackend(TensorRTBackendConfig(workspace_size=1 << 30, enable_tf32=False)),
        ],
        profiling_config=ProfilingConfig(batch_sizes=[1]),
    )
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(PerformanceValidationMode.DIAGNOSTIC)
    module = Module(source, "yolov10n", strategy=strategy)
    try:
        tune(
            module,
            DynamicShapeDataset([{input_name: images[0]}]),
            batch_sizes=[1],
            device="cuda",
            ignore_failing_modules=False,
        )
        actual = module(**{input_name: images})
        predictions = actual["output0"][0] if args.source == "onnx" else actual[0]
        reference = expected["output0"][0] if args.source == "onnx" else expected[0]
        predictions = predictions[predictions[:, 4] >= 0.4]
        reference = reference[reference[:, 4] >= 0.4]
        if not len(reference):
            raise RuntimeError("The street image produced no detections above the 0.4 confidence threshold")
        torch.testing.assert_close(predictions, reference, rtol=1e-2, atol=1e-2)
        checkpoint = output_dir / "yolov10n.ait"
        save(module, checkpoint)
        print(f"AITune checkpoint: {checkpoint}")
    finally:
        if module.state.name == "TUNED":
            module.deactivate()
        if isinstance(source, OnnxModule):
            source.deactivate()


if __name__ == "__main__":
    main()
