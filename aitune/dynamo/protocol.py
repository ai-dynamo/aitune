# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo request wiring and response conversion."""

import base64
import time
from collections.abc import Callable
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import torch

Modality = Literal["image", "video", "audio", "embedding"]
SUPPORTED_MODALITIES: frozenset[str] = frozenset({"image", "video", "audio", "embedding"})


def get_wiring(modality: str) -> tuple:
    """Return the Dynamo model input, model type, and request class for a modality."""
    from dynamo.llm import ModelInput, ModelType

    if modality == "embedding":
        from pydantic import BaseModel

        class EmbeddingRequest(BaseModel):
            model: str
            input: str | list[str] | list[list[str]]
            user: str | None = None
            dimensions: int | None = None

        return ModelInput.Text, ModelType.Embedding, EmbeddingRequest

    if modality == "image":
        from dynamo.common.protocols.image_protocol import NvCreateImageRequest

        return ModelInput.Text, ModelType.Images, NvCreateImageRequest

    if modality == "video":
        from dynamo.common.protocols.video_protocol import NvCreateVideoRequest

        return ModelInput.Text, ModelType.Videos, NvCreateVideoRequest

    if modality == "audio":
        try:
            from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest

            model_type = ModelType.Audios
        except (AttributeError, ImportError) as error:
            raise ImportError("The audio modality requires ai-dynamo-runtime>=1.1.0.") from error
        return ModelInput.Text, model_type, NvCreateAudioSpeechRequest

    raise ValueError(f"Unsupported modality {modality!r}. Valid types: {sorted(SUPPORTED_MODALITIES)}")


def get_register_model_kwargs() -> dict:
    """Return arguments required by the installed Dynamo version."""
    from dynamo.common import __version__ as dynamo_version
    from packaging.version import Version

    if Version(dynamo_version) < Version("1.3.0"):
        return {}

    from dynamo.llm import WorkerType

    return {"worker_type": WorkerType.Aggregated}


def pack_response(result: Any, modality: str, model: str) -> dict:
    """Convert an inference result to the Dynamo wire format."""
    if isinstance(result, dict):
        return result

    try:
        packer = _RESPONSE_PACKERS[modality]
    except KeyError as error:
        raise TypeError(f"Unsupported modality {modality!r}. Valid types: {sorted(SUPPORTED_MODALITIES)}") from error
    return packer(result, model)


def _pack_embedding_response(result: Any, model: str) -> dict:
    """Pack an embedding inference result."""
    if not isinstance(result, np.ndarray | torch.Tensor):
        raise TypeError(
            f"Cannot pack response of type {type(result).__name__!r} for modality 'embedding'. "
            "Return a dict or ndarray/Tensor."
        )
    if result.ndim == 0:
        raise ValueError("Embedding must be a 1D array, got 0D array.")
    if result.ndim > 2:
        raise ValueError(f"Embedding must be a 1D or 2D array, got {result.ndim}D array.")
    if result.ndim == 1:
        result = result.unsqueeze(0) if isinstance(result, torch.Tensor) else np.expand_dims(result, 0)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": row.tolist(), "index": index} for index, row in enumerate(result)
        ],
        "model": model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def _encode_media_response(result: Any, modality: str) -> str:
    """Validate and base64-encode a media inference result."""
    if not isinstance(result, bytes):
        raise TypeError(
            f"Cannot pack response of type {type(result).__name__!r} for modality {modality!r}. Return a dict or bytes."
        )
    return base64.b64encode(result).decode()


def _pack_image_response(result: Any, _model: str) -> dict:
    """Pack an image inference result."""
    encoded = _encode_media_response(result, "image")
    return {"created": int(time.time()), "data": [{"b64_json": encoded}]}


def _pack_video_response(result: Any, model: str) -> dict:
    """Pack a video inference result."""
    encoded = _encode_media_response(result, "video")
    return {
        "id": str(uuid4()),
        "object": "video",
        "model": model,
        "status": "completed",
        "progress": 100,
        "created": int(time.time()),
        "data": [{"b64_json": encoded, "output_format": "mp4"}],
    }


def _pack_audio_response(result: Any, model: str) -> dict:
    """Pack a text-to-speech inference result."""
    encoded = _encode_media_response(result, "audio")
    return {
        "id": str(uuid4()),
        "object": "audio.speech",
        "model": model,
        "status": "completed",
        "progress": 100,
        "created": int(time.time()),
        "data": [{"b64_json": encoded, "output_format": "wav"}],
    }


_RESPONSE_PACKERS: dict[str, Callable[[Any, str], dict]] = {
    "embedding": _pack_embedding_response,
    "image": _pack_image_response,
    "video": _pack_video_response,
    "audio": _pack_audio_response,
}
