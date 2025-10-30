# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model utilities for ASR pipeline."""

from copy import deepcopy

import nemo.collections.asr
import nemo.core.neural_types.neural_type
import torch
from nemo.core.classes.common import typecheck
from omegaconf import open_dict

import aitune.torch as ait


def get_model(model_name: str = "nvidia/parakeet-rnnt-1.1b"):
    """Get a pretrained ASR model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        ASR model
    """
    # Note: Disable typechecking for nemo, as passing inputs between onnx and nemo fails
    typecheck.set_typecheck_enabled(False)

    # Note: Allowing nemo object to be un/serialized (torch.load)
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

    asr_model = nemo.collections.asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_name)

    cfg = deepcopy(asr_model.decoding.cfg)
    with open_dict(cfg):
        cfg.greedy.use_cuda_graph_decoder = False
        cfg.strategy = "greedy_batch"

    asr_model.change_decoding_strategy(cfg)

    asr_model.eval()
    asr_model = asr_model.to("cuda")

    # Note: Move STFT window to GPU
    asr_model.preprocessor.featurizer.window = asr_model.preprocessor.featurizer.window.to("cuda")

    return asr_model


def wrap_pipeline(name: str, pipeline: nemo.collections.asr.models.EncDecRNNTBPEModel, strategy: ait.TuneStrategy):
    """Wrap modules in the pipeline.

    Args:
        name: The name of the pipeline
        pipeline: The ASR pipeline
        strategy: The tuning strategy

    Returns:
        ASR model
    """
    pipeline.encoder = ait.Module(
        pipeline.encoder,
        name=f"{name}-encoder",
        strategy=strategy,
    )

    pipeline.decoder.prediction["dec_rnn"] = ait.Module(
        pipeline.decoder.prediction["dec_rnn"],
        name=f"{name}-decoder-rnn",
        strategy=strategy,
    )

    return pipeline
