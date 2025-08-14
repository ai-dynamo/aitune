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
"""AI Dynamo service with ParakeetRNNT model and batching.

This service implements batched audio transcription using batch decorator
to improve throughput by processing multiple audio files together.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Annotated

import aiofiles
import torch
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk import DYNAMO_IMAGE, api, async_on_start, depends, endpoint, service
from dynamo.sdk.lib.config import ServiceConfig
from fastapi import Form, UploadFile
from fastapi.responses import StreamingResponse
from nemo.collections.asr.parts.mixins.transcription import InternalTranscribeConfig, TranscribeConfig
from pydantic import BaseModel, Field

import aitune.torch as ait
from aitune.torch.config import aitune_cache_dir

from ..tune import get_model, tune_model
from .batching import batch, get_or_create_event_loop

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


@service(
    dynamo={"namespace": "inference"},
    resource={"cpu": 4, "memory": "8Gi", "gpu": 1},
    workers=1,
    image=DYNAMO_IMAGE,
)
class ParakeetRNNTBatchedBackend:
    """Backend service for ParakeetRNNT model inference with batching."""

    def __init__(self) -> None:
        """Initialize the ParakeetRNNT backend with batching."""
        logger.info("Starting ParakeetRNNT backend with batching")
        config = ServiceConfig.get_instance()

        self.pipeline = None
        self.model_name = config.get("Backend", {}).get("model_name", "nvidia/parakeet-rnnt-1.1b")
        self.max_batch_size = config.get("Backend", {}).get("max_batch_size", 8)
        self.batch_timeout = config.get("Backend", {}).get("batch_timeout", 0.5)  # seconds

        self.tunning_audio_path = config.get("Backend", {}).get("tunning_audio_path", "./2086-149220-0033.wav")
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

    @async_on_start
    async def on_start(self):
        """Tune the model on start."""
        logger.info("Tuning model on start")

        # Load model
        if not self.tuned_model_path.exists() or self.force_tune:
            # Tune model
            tune_model(
                self.model_name,
                self.tunning_audio_path,
                self.tuned_model_path,
                strategy=None,
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
                            _internal=InternalTranscribeConfig(device="cuda"),
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

    @endpoint()
    async def transcribe_audio(self, request: AudioTranscriptionRequest):
        """Transcribe audio with batching."""
        logger.info("Received transcription request: %s...", request.request_id)

        # Process through batch handler - will automatically batch this
        try:
            response = await self.handle_batch(request)  # type: ignore
            if response:
                logger.info("Request completed successfully")
                yield response.model_dump_json()  # type: ignore
            else:
                logger.error("Request failed: no response")
                error_response = {"error": "No response generated"}
                yield str(error_response)
        except Exception as e:
            logger.error("Request failed: %s", e)
            error_response = {"error": str(e)}
            yield str(error_response)


@service(
    dynamo={"namespace": "inference"},
    image=DYNAMO_IMAGE,
)
class ParakeetRNNTBatchedFrontend:
    """Frontend HTTP API for ParakeetRNNT inference service with batching."""

    backend: ParakeetRNNTBatchedBackend = depends(ParakeetRNNTBatchedBackend)

    def __init__(self) -> None:
        """Initialize the ParakeetRNNT frontend."""
        configure_dynamo_logging(service_name="ParakeetRNNTBatchedFrontend")
        logger.info("Starting ParakeetRNNT frontend with batching")

        config = ServiceConfig.get_instance()

        self.port = config.get("Frontend", {}).get("port", 8000)
        self.audio_storage_path = Path(config.get("Backend", {}).get("audio_storage_path", "/tmp/audio"))
        logger.info("Frontend config port: %d", self.port)

    @api()
    async def transcribe_audio(self, audio_file: UploadFile, request_id: Annotated[str, Form()]):
        """HTTP endpoint for audio transcription with batching.

        Args:
            audio_file: Audio file to transcribe, expects wav format
            request_id: User provided request ID

        Returns:
            StreamingResponse: Streaming response with transcription
        """
        logger.info("Frontend received transcription request: %s...", request_id)

        internal_request_id = "" if request_id is None else request_id
        internal_request_id += "__" + str(uuid.uuid4())

        # store audio file in temp file, expecting wav format for now
        audio_file_path = self.audio_storage_path / f"{internal_request_id}.wav"
        async with aiofiles.open(audio_file_path, "wb") as f:
            await f.write(await audio_file.read())

        request = AudioTranscriptionRequest(
            request_id=request_id,
            audio_path=str(audio_file_path),
        )

        async def content_generator():
            async for response in self.backend.transcribe_audio(request.model_dump_json()):
                logger.info("Frontend received response")
                yield response

            logger.info("Frontend done")

        # audio_file_path.unlink(missing_ok=True)

        return StreamingResponse(content_generator())

    @api()
    async def health(self):
        """Health check endpoint."""
        return {"status": "healthy", "service": "ParakeetRNNT Audio Transcription with Batching"}
