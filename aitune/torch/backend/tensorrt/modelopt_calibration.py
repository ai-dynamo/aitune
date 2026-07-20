# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA ModelOpt Calibration module for quantization and autocast."""

import logging

import numpy as np
import torch

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.utils.tensor import format_tensor_name

# Setup logger
logger = logging.getLogger(__name__)


def prepare_calibration_data(data: list[Sample], graph_spec: GraphSpec) -> dict[str, np.ndarray]:
    """Prepare calibration data in ModelOpt format from 1..N samples and graph_spec.

    Builds a single dict of arrays (one per ONNX input) by mapping each sample to
    ONNX input names and concatenating all samples along axis=0. So you get one
    "large batch" per input, e.g. 3 samples of shapes (2,3,224,224), (1,3,224,224),
    (4,3,224,224) → one array of shape (7, 3, 224, 224). ModelOpt requires this
    format (dict of arrays, not a list of batches).

    Tensors are moved to CPU and converted to numpy. Uses graph_spec.input_spec
    for ONNX input names and locators into normalized forward arguments.

    Args:
        data: List of 1..N Sample objects (args, kwargs); each can have any batch size.
        graph_spec: Graph specification whose input_spec defines ONNX input names
            and locators into normalized forward arguments.

    Returns:
        Single dict mapping ONNX input names to numpy arrays. Each value has
        shape (total_examples, ...) with total_examples = sum of batch sizes
        across samples.

    Raises:
        ValueError: If data is empty or graph_spec has no tensor inputs.
    """
    if not data:
        raise ValueError("Calibration data must not be empty when samples are provided.")

    tensor_data = graph_spec.input_spec.tensor_data
    if not tensor_data:
        raise ValueError(
            "GraphSpec has no tensor inputs; calibration data cannot be built. "
            "Ensure the model was recorded with at least one tensor input."
        )

    onnx_input_names = [format_tensor_name(locator.path, "input") for locator, _ in tensor_data]
    logger.info(
        "Preparing calibration data for %d samples with ONNX input names: %s",
        len(data),
        onnx_input_names,
    )

    per_sample = _collect_per_sample_dicts(data, graph_spec)
    if not per_sample:
        raise ValueError(
            "No valid calibration samples could be built from the provided data and graph_spec. "
            "Ensure samples contain tensors that match the graph_spec input structure."
        )

    per_sample = _filter_to_representative_max_shape(per_sample)
    result = _concatenate_per_sample(per_sample)
    total_examples = sum(next(iter(d.values())).shape[0] for d in per_sample)
    logger.info(
        "Prepared ModelOpt calibration data: %d samples, %d total examples",
        len(per_sample),
        total_examples,
    )
    for name, arr in result.items():
        logger.info("  Calibration input %r: shape %s", name, arr.shape)
    return result


def _collect_per_sample_dicts(
    data: list[Sample],
    graph_spec: GraphSpec,
) -> list[dict[str, np.ndarray]]:
    """Build list of per-sample dicts (ONNX name -> array), skip empty, raise on batch size 0."""
    per_sample: list[dict[str, np.ndarray]] = []
    for sample in data:
        input_dict = _sample_to_input_dict(sample, graph_spec)
        if not input_dict:
            logger.warning(
                "Sample produced no tensor inputs; skipping. Check that sample structure matches graph_spec."
            )
            continue
        if any(arr.shape[0] == 0 for arr in input_dict.values()):
            raise ValueError(
                "Calibration sample has batch size 0 (shape[0] == 0). Samples must have at least one batch element."
            )
        per_sample.append(input_dict)
    return per_sample


def _sample_to_input_dict(
    sample: Sample,
    graph_spec: GraphSpec,
) -> dict[str, np.ndarray]:
    """Convert one (args, kwargs) sample to a dict of ONNX input name -> numpy array."""
    args, kwargs = sample
    forward_inputs = graph_spec.forward_signature.normalize(args, kwargs)
    input_dict: dict[str, np.ndarray] = {}
    for locator, _ in graph_spec.input_spec.tensor_data:
        value = locator.get_value(forward_inputs.arguments)
        if not torch.is_tensor(value):
            logger.debug(
                "Skipping non-tensor value for input %s (type=%s); may be optional or nested.",
                locator.display_path,
                type(value).__name__,
            )
            continue
        input_dict[format_tensor_name(locator.path, "input")] = value.detach().cpu().numpy()
    return input_dict


def _sample_shape_rank(sample: dict[str, np.ndarray]) -> int:
    """Return a rank for this sample (larger = larger shape). Used to pick max-shape representative."""
    return max(arr.size for arr in sample.values())


def _shapes_match(sample: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> bool:
    """Return True if sample has the same non-batch shapes as reference for all keys.

    Compares shape[1:] only so different batch sizes (axis=0) are allowed and
    can be concatenated along axis=0.
    """
    if set(sample.keys()) != set(reference.keys()):
        return False
    return all(sample[name].shape[1:] == reference[name].shape[1:] for name in reference)


def _filter_to_representative_max_shape(
    per_sample: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    """Keep only samples whose non-batch shape matches the representative.

    The representative is the sample with the largest total number of elements.
    Matching is on shape[1:] only, so different batch sizes are kept and
    concatenated along axis=0. Samples with different non-batch shapes are
    dropped with a warning.
    """
    if len(per_sample) <= 1:
        return per_sample
    representative = max(per_sample, key=_sample_shape_rank)
    filtered = [d for d in per_sample if _shapes_match(d, representative)]
    dropped = len(per_sample) - len(filtered)
    if dropped > 0:
        logger.warning(
            "Calibration: using representative (max) shape; %d sample(s) with different shapes were dropped.",
            dropped,
        )
    return filtered


def _concatenate_per_sample(per_sample: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Merge list of per-sample dicts into one dict with arrays concatenated on axis=0."""
    result: dict[str, np.ndarray] = {}
    for name in per_sample[0]:
        arrays = [d[name] for d in per_sample if name in d]
        if arrays:
            result[name] = np.concatenate(arrays, axis=0)
    return result
