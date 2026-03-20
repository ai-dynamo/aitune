# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AI Dynamo service with ParakeetRNNT model and batching.

This service implements batched audio transcription using batch decorator
to improve throughput by processing multiple audio files together.
"""

import asyncio
import logging
import os
import time
from pathlib import Path

import torch
import uvloop
import yaml
from aitune_examples_common.batching import batch, get_or_create_event_loop
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from nemo.collections.asr.parts.mixins.transcription import InternalTranscribeConfig, TranscribeConfig
from pydantic import BaseModel, Field

import aitune.torch as ait
from aitune.torch.backend import TensorRTBackend, TorchEagerBackend, TorchInductorJitBackend
from aitune.torch.config import aitune_cache_dir

from ..sample_data import ensure_sample_audio
from ..tune import get_model, tune_model

logger = logging.getLogger(__name__)


class AudioTranscriptionRequest(BaseModel):
    """Request model for audio transcription."""

    request_id: str | None = Field(default=None, description="Request ID")
    audio_path: str = Field(description="Path to audio file")


class AudioTranscriptionResponse(BaseModel):
    """Response model for audio transcription results."""

    request_id: str
    transcription: str
    transcription_time: float
    error: str | None = Field(default=None, description="Error message")

    @staticmethod
    def make_error_response(request_id: str, error: str) -> "AudioTranscriptionResponse":
        """Make an error response."""
        return AudioTranscriptionResponse(
            request_id=request_id,
            transcription="",
            transcription_time=0,
            error=error,
        )


class ParakeetRNNTBatchedBackend:
    """Backend service for ParakeetRNNT model inference with batching."""

    def __init__(self, config: dict) -> None:
        """Initialize the ParakeetRNNT backend with batching."""
        logger.info("Starting ParakeetRNNT backend with batching")

        self.pipeline = None
        self.model_name = config.get("Backend", {}).get("model_name", "nvidia/parakeet-rnnt-1.1b")
        self.max_batch_size = config.get("Backend", {}).get("max_batch_size", 8)
        self.batch_timeout = config.get("Backend", {}).get("batch_timeout", 0.5)  # seconds

        self.tuning_audio_path = config.get("Backend", {}).get("tuning_audio_path", str(ensure_sample_audio()))
        self.audio_storage_path = Path(config.get("Backend", {}).get("audio_storage_path", "/tmp/audio"))
        self.force_tune = config.get("Backend", {}).get("force_tune", False)
        self.audio_storage_path.mkdir(parents=True, exist_ok=True)

        self.tuned_model_path = config.get("Backend", {}).get("tuned_model_path")
        if self.tuned_model_path is None:
            self.tuned_model_path = aitune_cache_dir() / f"{self.model_name.replace('/', '_')}.pt"
        else:
            self.tuned_model_path = Path(self.tuned_model_path)

        logging.getLogger("aitune").setLevel(logging.INFO)

        self.setup_batching()

    def setup_batching(self):
        """Setup batching for the backend."""
        self.handle_batch = batch(max_batch_size=self.max_batch_size, batch_wait_timeout_s=self.batch_timeout)(
            self.handle_batch
        )

    async def initialize_model(self):
        """Tune the model on start."""
        logger.info("Tuning model on start")

        strategy = ait.FirstWinsStrategy(
            backends=[
                TensorRTBackend(),
                TorchInductorJitBackend(),
                TorchEagerBackend(),
            ]
        ).enable_find_max_batch_size(enable=False)

        # Load model
        if not self.tuned_model_path.exists() or self.force_tune:
            # Tune model
            tune_model(
                self.model_name,
                self.tuning_audio_path,
                self.tuned_model_path,
                strategy=strategy,
                batch_sizes=list(range(1, self.max_batch_size + 1)),
            )

        logger.info("Loading tuned model from %s", self.tuned_model_path)

        self.pipeline = get_model(self.model_name)
        ait.load(self.pipeline, self.tuned_model_path)

        logger.info("Backend initialized with model: %s, max_batch_size: %d", self.model_name, self.max_batch_size)

    async def handle_batch(
        self, requests: AudioTranscriptionRequest | list[AudioTranscriptionRequest]
    ) -> list[AudioTranscriptionResponse]:
        """Process a batch of audio transcription requests using batching."""
        if not requests:
            return []

        if isinstance(requests, AudioTranscriptionRequest):
            logger.info("Single request received, converting to list")
            requests = [requests]

        logger.info("Processing batch of %d requests", len(requests))
        start_time = time.monotonic_ns()

        try:
            # Extract batch parameters
            audio_paths = [req.audio_path for req in requests]
            logger.info("Audio paths: %s", audio_paths)

            audio_sizes = [Path(path).stat().st_size for path in audio_paths]
            logger.info("Audio sizes: %s", audio_sizes)

            # Generate transcriptions for all audio files in batch
            def generate_batch_transcriptions():
                if self.pipeline is None:
                    raise ValueError("Pipeline is not initialized")
                with torch.no_grad():
                    return self.pipeline.transcribe(
                        audio_paths,
                        override_config=TranscribeConfig(
                            batch_size=min(len(audio_paths), self.max_batch_size),
                            verbose=False,
                            _internal=InternalTranscribeConfig(device=torch.device("cuda")),
                        ),
                    )

            loop = get_or_create_event_loop()
            results = await loop.run_in_executor(None, generate_batch_transcriptions)

            transcription_time = (time.monotonic_ns() - start_time) / 1e9
            logger.info("Batch transcription completed in %.2fs", transcription_time)

            responses = []
            for i, result in enumerate(results):
                request_id = requests[i].request_id
                logger.info("Transcription for request %s: %s", request_id, result.text)

                try:
                    response = AudioTranscriptionResponse(
                        request_id=request_id,
                        transcription=result.text,
                        transcription_time=transcription_time,
                    )

                    responses.append(response)
                    logger.info("Processed request %s", request_id)

                except Exception as e:
                    logger.error("Error processing request %s: %s", request_id, e)
                    # Create error response
                    error_response = AudioTranscriptionResponse(
                        request_id=request_id,
                        transcription="",  # Empty for error
                        transcription_time=transcription_time,
                        error=str(e),
                    )
                    responses.append(error_response)

            return responses

        except Exception as e:
            logger.error("Error processing batch: %s", e)
            # Return error responses for all requests
            error_responses = []
            for req in requests:
                error_response = AudioTranscriptionResponse(
                    request_id=req.request_id or "",
                    transcription="",  # Empty for error
                    transcription_time=(time.monotonic_ns() - start_time) / 1e9,
                    error=str(e),
                )
                error_responses.append(error_response)
            return error_responses

    @dynamo_endpoint(AudioTranscriptionRequest, AudioTranscriptionResponse)
    async def transcribe_audio(self, request: AudioTranscriptionRequest):
        """Transcribe audio with batching."""
        logger.info("Received transcription request: %s...", request.request_id)

        # Process through batch handler - will automatically batch this
        try:
            response = await self.handle_batch(request)
            if response:
                logger.info("Request completed successfully")
                yield response.model_dump()
            else:
                logger.error("Request failed: no response")
                yield AudioTranscriptionResponse.make_error_response(
                    request.request_id, "No response generated"
                ).model_dump()
        except Exception as e:
            logger.error("Request failed: %s", e)
            yield AudioTranscriptionResponse.make_error_response(request.request_id, str(e)).model_dump()


@dynamo_worker()
async def backend_worker(runtime: DistributedRuntime):
    namespace_name = "parakeet_rnnt"
    component_name = "backend"
    endpoint_name = "transcribe_audio"

    component = runtime.namespace(namespace_name).component(component_name)
    await component.create_service()

    logger.info("Created service %s/%s", namespace_name, component_name)

    endpoint = component.endpoint(endpoint_name)
    lease_id = endpoint.lease_id()
    logger.info("Serving endpoint %s on lease %s", endpoint_name, lease_id)

    backend = ParakeetRNNTBatchedBackend(_get_config())
    await backend.initialize_model()
    await endpoint.serve_endpoint(backend.transcribe_audio)


def _get_config() -> dict:
    with Path(os.environ.get("AITUNE_EXAMPLE_CONFIG_PATH", "config.yaml")).open() as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    uvloop.install()

    asyncio.run(backend_worker())
