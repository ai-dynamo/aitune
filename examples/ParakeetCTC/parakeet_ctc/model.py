# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model utilities for ASR pipeline."""

from copy import deepcopy

import nemo.collections.asr
import nemo.core.neural_types.neural_type
import torch
from nemo.core.classes.common import typecheck

from aitune.torch import Module, TuneStrategy


def get_model(model_name: str = "nvidia/parakeet-ctc-0.6b") -> nemo.collections.asr.models.EncDecCTCModel:
    """Get a pretrained ASR model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        ASR model
    """
    # Note: Disable typechecking for nemo, as passing inputs between onnx and nemo fails
    typecheck.set_typecheck_enabled(False)

    # Note: Allowing nemo object to be un/serialized (torch.load) #
    torch.serialization.add_safe_globals([
        # pytype: disable=module-attr
        nemo.core.neural_types.neural_type.NeuralType,
        nemo.core.neural_types.elements.MelSpectrogramType,
        nemo.core.neural_types.axes.AxisType,
        nemo.core.neural_types.axes.AxisKind,
        nemo.core.neural_types.elements.LengthsType,
        nemo.core.neural_types.elements.SpectrogramType,
        nemo.core.neural_types.elements.IntType,
        nemo.core.neural_types.elements.AcousticEncodedRepresentation,
        # pytype: enable=module-attr
    ])

    asr_model = nemo.collections.asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)

    cfg = deepcopy(asr_model.decoding.cfg)
    cfg.strategy = "greedy_batch"
    asr_model.change_decoding_strategy(cfg)

    asr_model.eval()
    asr_model = asr_model.to("cuda")

    # Note: Move STFT window to GPU
    asr_model.preprocessor.featurizer.window = asr_model.preprocessor.featurizer.window.to("cuda")

    return asr_model


def wrap_pipeline(name: str, pipeline, strategy: TuneStrategy | None = None):
    """Wrap modules in the pipeline.

    Args:
        name: The name of the pipeline
        pipeline: The ASR pipeline

    Returns:
        ASR model
    """
    pipeline.encoder = Module(
        pipeline.encoder,
        name=f"{name}-encoder",
        strategy=deepcopy(strategy),
    )

    pipeline.decoder = Module(
        pipeline.decoder,
        name=f"{name}-decoder",
        strategy=deepcopy(strategy),
    )

    return pipeline
