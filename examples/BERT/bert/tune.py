# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune BERT from Torch or ONNX and save an AITune checkpoint."""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel

from aitune.torch import MaxThroughputStrategy, Module, PerformanceValidationMode, config, save, tune
from aitune.torch.backend import ONNXRuntimeBackend, TensorRTBackend, TensorRTBackendConfig, TensorRTProfile
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module import OnnxModule
from aitune.torch.task.profiling import ProfilingConfig

MODEL_NAME = "bert-base-uncased"
BATCH_SIZES = [1, 2, 4]
SEQUENCE_LENGTHS = [64, 128, 256]


def _tensorrt_profiles() -> list[TensorRTProfile]:
    """Cover recorded shapes and gaps; SAMPLES_USED cannot add a wide fallback."""
    profiles = []
    for sequence_length in SEQUENCE_LENGTHS:
        for batch_size in BATCH_SIZES:
            shape = (batch_size, sequence_length)
            profiles.append(TensorRTProfile().add_input_shape("input_ids", shape, shape, shape))
    # The Python runtime uses the first matching profile, so the broad profile must be last.
    profiles.append(
        TensorRTProfile().add_input_shape(
            "input_ids", (1, min(SEQUENCE_LENGTHS)), (2, 128), (max(BATCH_SIZES), max(SEQUENCE_LENGTHS))
        )
    )
    return profiles


def _bert_model() -> torch.nn.Module:
    """Load the same model and output contract as the ONNX functional test."""
    return AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager", return_dict=False).eval()


def _onnx_source(path: Path, tokens: torch.Tensor) -> OnnxModule:
    """Use an existing ONNX file, exporting the test model only when none was supplied."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            _bert_model(),
            (tokens[:1],),
            path,
            input_names=["input_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
                "pooler_output": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
            external_data=False,
        )
    return OnnxModule(path)


@torch.inference_mode()
def main() -> None:
    """Tune one source format and save its deployment-capable artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("torch", "onnx"), required=True)
    parser.add_argument("--target", choices=("triton",), default="triton")
    parser.add_argument("--onnx-path", type=Path, help="Existing BERT ONNX file with the test's input/output interface")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults to this example's artifacts")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This prototype requires a CUDA GPU")
    if args.source == "torch" and args.onnx_path is not None:
        parser.error("--onnx-path can only be used with --source onnx")
    if args.onnx_path is not None and not args.onnx_path.is_file():
        parser.error(f"ONNX model file does not exist: {args.onnx_path}")
    output_dir = args.output_dir or Path(__file__).resolve().parents[1] / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = AutoConfig.from_pretrained(MODEL_NAME).vocab_size
    generator = torch.Generator().manual_seed(0)
    tokens = {
        length: torch.randint(vocab_size, (max(BATCH_SIZES), length), generator=generator)
        for length in SEQUENCE_LENGTHS
    }
    source = (
        _bert_model().cuda()
        if args.source == "torch"
        else _onnx_source(args.onnx_path or output_dir / f"{MODEL_NAME}.onnx", tokens[128])
    )
    requests = {length: values.cuda() for length, values in tokens.items()}
    validation_batches = [*BATCH_SIZES, 3]
    expected = {
        (length, batch_size): source(input_ids=values[:batch_size])
        for length, values in requests.items()
        for batch_size in validation_batches
    }

    # Preserve one sample for each measured batch/sequence shape.
    config.max_num_samples_stored = len(BATCH_SIZES) * len(SEQUENCE_LENGTHS)

    strategy = MaxThroughputStrategy(
        [
            ONNXRuntimeBackend(),
            TensorRTBackend(TensorRTBackendConfig(workspace_size=1 << 30, profiles=_tensorrt_profiles())),
        ],
        profiling_config=ProfilingConfig(batch_sizes=BATCH_SIZES),
    )
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(PerformanceValidationMode.DIAGNOSTIC)
    module = Module(source, "bert", strategy=strategy)
    try:
        tune(
            module,
            DynamicShapeDataset([{"input_ids": values[0]} for values in requests.values()]),
            batch_sizes=BATCH_SIZES,
            device="cuda",
            ignore_failing_modules=False,
        )
        for (length, batch_size), baseline in expected.items():
            torch.testing.assert_close(module(input_ids=requests[length][:batch_size]), baseline, rtol=1e-2, atol=1e-2)
        checkpoint = output_dir / "bert.ait"
        save(module, checkpoint)
        print(f"AITune checkpoint: {checkpoint}")
    finally:
        if module.state.name == "TUNED":
            module.deactivate()
        if isinstance(source, OnnxModule):
            source.deactivate()


if __name__ == "__main__":
    main()
