# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native execution of exported deployment artifacts for functional CI."""

from collections.abc import Mapping
from pathlib import Path

import torch

from aitune.records import DeploymentArtifact


def run_artifact(
    artifact: DeploymentArtifact,
    exported_path: Path,
    inputs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Execute an exported artifact through its native runtime.

    Args:
        artifact: Deployment description whose format selects the native runtime.
        exported_path: Main model file written by ``artifact.model.export_files``.
        inputs: Input tensors keyed by ``artifact.input_names``.

    Returns:
        Output tensors keyed by ``artifact.output_names``.
    """
    if artifact.model.format == "pt2":
        return _run_pt2(artifact, exported_path, inputs)
    if artifact.model.format == "tensorrt_plan":
        return _run_tensorrt(artifact, exported_path, inputs)
    raise NotImplementedError(f"No native runner for artifact format {artifact.model.format!r}")


def _run_pt2(
    artifact: DeploymentArtifact,
    exported_path: Path,
    inputs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run an AOT Inductor ``.pt2`` package with positional inputs in artifact order."""
    ordered_inputs = [inputs[name] for name in artifact.input_names]
    device = ordered_inputs[0].device
    device_index = device.index if device.index is not None else 0
    runner = torch._inductor.aoti_load_package(str(exported_path), device_index=device_index)
    with torch.no_grad():
        result = runner(*ordered_inputs)
    outputs = result if isinstance(result, (list, tuple)) else (result,)
    return dict(zip(artifact.output_names, outputs, strict=True))


def _run_tensorrt(
    artifact: DeploymentArtifact,
    exported_path: Path,
    inputs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run an exported TensorRT plan through Polygraphy's ``TrtRunner``."""
    import tensorrt as trt
    from polygraphy.backend.trt import TrtRunner

    ordered_inputs = [(name, inputs[name]) for name in artifact.input_names]
    device = ordered_inputs[0][1].device
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(exported_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT engine from {exported_path}")
    feed = {name: tensor.detach().cpu().numpy() for name, tensor in ordered_inputs}
    with TrtRunner(engine) as runner:
        outputs = runner.infer(feed)
    return {name: torch.from_numpy(outputs[name]).to(device) for name in artifact.output_names}
