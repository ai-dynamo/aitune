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
"""AI Dynamo service with Stable Diffusion model and batching.

This service implements batched image generation using batch decorator
to improve throughput by processing multiple prompts together.
"""

import base64
import io
import logging
import time
import uuid
from pathlib import Path

import torch
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk import DYNAMO_IMAGE, api, async_on_start, depends, endpoint, service
from dynamo.sdk.lib.config import ServiceConfig
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import aitune.torch as ait
from aitune.torch.config import aitune_cache_dir

from ..model import get_pipeline
from ..tune import tune_model
from .batching import batch, get_or_create_event_loop

logger = logging.getLogger(__name__)


class ImageGenerationRequest(BaseModel):
    """Request model for image generation."""

    request_id: str | None = Field(default=None, description="Request ID")
    prompt: str = Field(description="Text prompt for image generation")
    height: int = Field(default=1024, description="Image height")
    width: int = Field(default=1024, description="Image width")
    num_inference_steps: int = Field(default=20, description="Number of inference steps")
    guidance_scale: float = Field(default=7.5, description="Guidance scale")
    max_sequence_length: int = Field(default=77, description="Maximum sequence length")


class ImageGenerationResponse(BaseModel):
    """Response model for image generation results."""

    request_id: str
    image_data: str  # Base64 encoded image
    prompt: str
    generation_time: float
    error: str | None = Field(default=None, description="Error message")


@service(
    dynamo={"namespace": "inference"},
    resource={"cpu": 4, "memory": "8Gi", "gpu": 1},
    workers=1,
    image=DYNAMO_IMAGE,
)
class StableDiffusionBatchedBackend:
    """Backend service for Stable Diffusion model inference with batching."""

    def __init__(self) -> None:
        """Initialize the Stable Diffusion backend with batching."""
        logger.info("Starting Stable Diffusion backend with batching")
        config = ServiceConfig.get_instance()

        self.pipeline = None
        self.model_name = config.get("Backend", {}).get("model_name", "stabilityai/stable-diffusion-2-1")
        self.max_batch_size = config.get("Backend", {}).get("max_batch_size", 4)
        self.batch_timeout = config.get("Backend", {}).get("batch_timeout", 2.0)  # seconds
        self.force_tune = config.get("Backend", {}).get("force_tune", False)

        self.prompt = config.get("Backend", {}).get("prompt", "A beautiful sunset over a calm ocean")
        self.sizes = config.get("Backend", {}).get("sizes", [(1024, 1024), (512, 512)])
        self.steps = config.get("Backend", {}).get("steps", 20)
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
            # Generate batch sizes, taking just powers of 2 up to max_batch_size
            batch_sizes, bs = [], 1
            while bs <= self.max_batch_size:
                batch_sizes.append(bs)
                bs *= 2

            # Tune model
            tune_model(
                self.model_name,
                self.prompt,
                self.sizes,
                self.steps,
                self.tuned_model_path,
                batch_sizes=batch_sizes,
            )

        logger.info("Loading tuned model from %s", self.tuned_model_path)
        self.pipeline = get_pipeline(self.model_name)
        ait.load(self.pipeline, self.tuned_model_path)  # type: ignore

        logger.info("Backend initialized with model: %s, max_batch_size: %d", self.model_name, self.max_batch_size)

    async def handle_batch(
        self, requests: ImageGenerationRequest | list[ImageGenerationRequest]
    ) -> list[ImageGenerationResponse]:
        """Process a batch of image generation requests using batching."""
        if not requests:
            return []

        if isinstance(requests, ImageGenerationRequest):
            logger.info("Single request received, converting to list")
            requests = [requests]

        logger.info("Processing batch of %d requests", len(requests))
        start_time = time.monotonic_ns()

        try:
            # Extract batch parameters
            prompts = [req.prompt for req in requests]

            # Taking the first request's parameters for the batch (simplified approach)
            height = requests[0].height
            width = requests[0].width
            num_steps = requests[0].num_inference_steps

            # Generate images for all prompts in batch
            def generate_batch_images():
                if self.pipeline is None:
                    raise ValueError("Pipeline is not initialized")
                with torch.no_grad():
                    return self.pipeline(
                        prompt=prompts,
                        height=height,
                        width=width,
                        num_inference_steps=num_steps,
                        num_images_per_prompt=1,
                        return_dict=True,
                    )

            loop = get_or_create_event_loop()
            result = await loop.run_in_executor(None, generate_batch_images)
            images = result.images

            generation_time = (time.monotonic_ns() - start_time) / 1e9
            logger.info("Batch generation completed in %.2fs", generation_time)

            # Convert images to base64 and create responses
            responses = []
            for i, req in enumerate(requests):
                try:
                    # Convert PIL image to base64
                    img_buffer = io.BytesIO()
                    images[i].save(img_buffer, format="JPEG")
                    img_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

                    response = ImageGenerationResponse(
                        request_id=req.request_id or "",
                        image_data=img_data,
                        prompt=req.prompt,
                        generation_time=generation_time,
                    )

                    responses.append(response)
                    logger.info("Processed request %d", i + 1)

                except Exception as e:
                    logger.error("Error processing request %d: %s", i + 1, e, exc_info=True)
                    # Create error response
                    error_response = ImageGenerationResponse(
                        request_id=req.request_id or "",
                        image_data="",  # Empty for error
                        prompt=req.prompt,
                        generation_time=generation_time,
                        error=str(e),
                    )
                    responses.append(error_response)

            return responses

        except Exception as e:
            logger.error("Error processing batch: %s", e)
            # Return error responses for all requests
            error_responses = []
            for req in requests:
                error_response = ImageGenerationResponse(
                    request_id=req.request_id or "",
                    image_data="",  # Empty for error
                    prompt=req.prompt,
                    generation_time=(time.monotonic_ns() - start_time) / 1e9,
                    error=str(e),
                )
                error_responses.append(error_response)
            return error_responses

    @endpoint()
    async def generate_image(self, req: ImageGenerationRequest):
        """Generate image with batching."""
        logger.info("Received generation request: %s...", req.prompt[:50])

        # set request_id to a new uuid, respecting the existing request_id if it exists
        req.request_id = "" if req.request_id is None else req.request_id
        req.request_id += "__" + str(uuid.uuid4())

        # Process through batch handler - will automatically batch this
        try:
            response = await self.handle_batch(req)  # type: ignore
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
class StableDiffusionBatchedFrontend:
    """Frontend HTTP API for Stable Diffusion inference service with batching."""

    backend: StableDiffusionBatchedBackend = depends(StableDiffusionBatchedBackend)

    def __init__(self) -> None:
        """Initialize the Stable Diffusion frontend."""
        configure_dynamo_logging(service_name="StableDiffusionBatchedFrontend")
        logger.info("Starting Stable Diffusion frontend with batching")

        config = ServiceConfig.get_instance()
        self.port = config.get("Frontend", {}).get("port", 8000)
        logger.info("Frontend config port: %d", self.port)

    @api()
    async def generate_image(self, request: ImageGenerationRequest):
        """HTTP endpoint for image generation with batching."""
        logger.info("Frontend received generation request: %s...", request.prompt[:50])

        async def content_generator():
            async for response in self.backend.generate_image(request.model_dump_json()):
                logger.info("Frontend received response")
                yield response

            logger.info("Frontend done")

        return StreamingResponse(content_generator())

    @api()
    async def health(self):
        """Health check endpoint."""
        return {"status": "healthy", "service": "Stable Diffusion Image Generation with Batching"}
