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

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

import torch
import uvloop
import yaml
from aitune_examples_common.batching import batch, get_or_create_event_loop
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from pydantic import BaseModel, Field

import aitune.torch as ait
from aitune.torch.backend import TensorRTBackend, TorchEagerBackend, TorchInductorBackend
from aitune.torch.config import aitune_cache_dir

from ..model import get_pipeline
from ..tune import tune_model

logger = logging.getLogger(__name__)


class ImageGenerationRequest(BaseModel):
    """Request model for image generation."""

    request_id: str | None = Field(default=None, description="Request ID")
    internal_request_id: str | None = Field(default=None, description="Internal request ID")
    prompt: str = Field(description="Text prompt for image generation")
    height: int = Field(default=1024, description="Image height")
    width: int = Field(default=1024, description="Image width")
    num_inference_steps: int = Field(default=20, description="Number of inference steps")
    guidance_scale: float = Field(default=7.5, description="Guidance scale")
    max_sequence_length: int = Field(default=77, description="Maximum sequence length")


class ImageGenerationResponse(BaseModel):
    """Response model for image generation results."""

    request_id: str
    internal_request_id: str
    image_path: str
    prompt: str
    generation_time: float
    error: str | None = Field(default=None, description="Error message")

    @staticmethod
    def make_error_response(request_id: str, internal_request_id: str, error: str) -> "ImageGenerationResponse":
        """Make an error response."""
        return ImageGenerationResponse(
            request_id=request_id,
            internal_request_id=internal_request_id,
            image_path="",
            prompt="",
            generation_time=0,
            error=error,
        )


class StableDiffusionBatchedBackend:
    """Backend service for Stable Diffusion model inference with batching."""

    def __init__(self, config: dict) -> None:
        """Initialize the Stable Diffusion backend with batching."""
        logger.info("Starting Stable Diffusion backend with batching")

        self.pipeline = None
        self.model_name = config.get("Backend", {}).get("model_name", "stabilityai/stable-diffusion-2-1")
        self.max_batch_size = config.get("Backend", {}).get("max_batch_size", 4)
        self.batch_timeout = config.get("Backend", {}).get("batch_timeout", 2.0)  # seconds
        self.force_tune = config.get("Backend", {}).get("force_tune", False)
        self.image_storage_path = Path(config.get("Backend", {}).get("image_storage_path", "/images"))
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

    async def initialize_model(self):
        """Tune the model on start."""
        logger.info("Tuning model on start")

        strategy = ait.FirstWinsStrategy(
            backends=[
                TensorRTBackend(),
                TorchInductorBackend(),
                TorchEagerBackend(),
            ]
        ).enable_find_max_batch_size(enable=False)

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
                strategy=strategy,
            )

        logger.info("Loading tuned model from %s", self.tuned_model_path)
        self.pipeline = get_pipeline(self.model_name)
        ait.load(self.pipeline, self.tuned_model_path)

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
                    # Save image to storage
                    image_path = self.image_storage_path / f"{req.internal_request_id}.jpg"
                    with image_path.open("wb") as f:
                        images[i].save(f, format="JPEG")

                    response = ImageGenerationResponse(
                        request_id=req.request_id or "",
                        internal_request_id=req.internal_request_id,
                        image_path=str(image_path),
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
                        internal_request_id=req.internal_request_id,
                        image_path="",  # Empty for error
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
                    internal_request_id=req.internal_request_id,
                    image_path="",
                    prompt=req.prompt,
                    generation_time=(time.monotonic_ns() - start_time) / 1e9,
                    error=str(e),
                )
                error_responses.append(error_response)
            return error_responses

    @dynamo_endpoint(ImageGenerationRequest, ImageGenerationResponse)
    async def generate_image(self, request: ImageGenerationRequest):
        """Generate image with batching."""
        logger.info("Received generation request")

        # set request_id to a new uuid, respecting the existing request_id if it exists
        request.internal_request_id = "" if request.request_id is None else request.request_id
        request.internal_request_id += "__" + str(uuid.uuid4())

        # Process through batch handler - will automatically batch this
        try:
            response = await self.handle_batch(request)
            if response:
                logger.info("Request completed successfully")
                yield response.model_dump()
            else:
                logger.error("Request failed: no response")
                yield ImageGenerationResponse.make_error_response(
                    request.request_id, request.internal_request_id, "No response generated"
                ).model_dump()
        except Exception as e:
            logger.error("Request failed: %s", e)
            yield ImageGenerationResponse.make_error_response(
                request.request_id, request.internal_request_id, str(e)
            ).model_dump()


@dynamo_worker()
async def backend_worker(runtime: DistributedRuntime):
    namespace_name = "stable_diffusion"
    component_name = "backend"
    endpoint_name = "generate_image"
    lease_id = runtime.etcd_client().primary_lease_id()

    component = runtime.namespace(namespace_name).component(component_name)
    await component.create_service()

    logger.info("Created service %s/%s", namespace_name, component_name)

    endpoint = component.endpoint(endpoint_name)

    logger.info("Serving endpoint %s on lease %s", endpoint_name, lease_id)

    backend = StableDiffusionBatchedBackend(_get_config())
    await backend.initialize_model()
    await endpoint.serve_endpoint(backend.generate_image)


def _get_config() -> dict:
    with Path(os.environ.get("AITUNE_EXAMPLE_CONFIG_PATH", "config.yaml")).open() as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    uvloop.install()

    asyncio.run(backend_worker())
