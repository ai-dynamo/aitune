# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""AI Dynamo service with ResNet model and batching.

This service implements batched image classification using batch decorator
to improve throughput by processing multiple images together.
"""

import asyncio
import logging
import os
import sys
import time
from asyncio import subprocess
from pathlib import Path
from typing import Any

import torch
import uvloop
import yaml
from aitune_examples_common.batching import batch, get_or_create_event_loop
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_tensor import Tensor

import aitune.torch as ait
from aitune.torch.config import aitune_cache_dir

from ..model import get_model, get_transform

logger = logging.getLogger(__name__)


class ImageClassificationRequest(BaseModel):
    """Request model for image classification."""

    request_id: str | None = Field(default=None, description="Request ID")
    internal_request_id: str | None = Field(default=None, description="Internal request ID")
    image_path: str = Field(description="Image path from storage")
    tensor_image: Tensor[torch.Tensor, Any, Any] | None = Field(default=None, description="Tensor image")


class ImageClassificationResponse(BaseModel):
    """Response model for image classification results."""

    request_id: str | None = Field(default=None, description="Request ID")
    prediction: str
    confidence: float
    class_id: int
    inference_time: float
    error: str | None = Field(default=None, description="Error message")

    @staticmethod
    def make_error_response(request_id: str, error: str) -> "ImageClassificationResponse":
        """Create an error response."""
        return ImageClassificationResponse(
            request_id=request_id,
            prediction="",
            confidence=0.0,
            class_id=-1,
            inference_time=0,
            error=error,
        )


class ResNetBatchedBackend:
    """Backend service for ResNet model inference with batching."""

    def __init__(self, config: dict) -> None:
        """Initialize the ResNet backend with batching."""
        logger.info("Starting ResNet backend with batching")

        self.model = None
        self.transform = None
        self.class_names = []

        self.model_name = config.get("Backend", {}).get("model_name", "resnet50")
        self.max_batch_size = config.get("Backend", {}).get("max_batch_size", 4)
        self.batch_timeout = config.get("Backend", {}).get("batch_timeout", 0.5)  # seconds
        self.image_path = config.get("Backend", {}).get("image_path", "/opt/aitune/dynamo/app/dog.webp")
        self.classes_file = config.get("Backend", {}).get("classes_file", "/opt/aitune/dynamo/app/imagenet_classes.txt")
        self.pretrained = config.get("Backend", {}).get("pretrained", True)
        self.force_tune = config.get("Backend", {}).get("force_tune", False)
        self.image_storage_path = config.get("Backend", {}).get("image_storage_path", "/tmp/images")

        self.tuned_model_path = config.get("Backend", {}).get("tuned_model_path")
        if self.tuned_model_path is None:
            self.tuned_model_path = aitune_cache_dir() / f"{self.model_name}.pt"
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
        """Initialize the model on start."""
        logger.info("Initializing ResNet model: %s", self.model_name)

        # Load model and transform
        logger.info("Loading tuned model from %s", self.tuned_model_path)
        self.model = get_model(self.model_name, self.pretrained)
        self.transform = get_transform(self.model)
        ait.load(self.model, self.tuned_model_path)

        # Load class names
        classes_file = Path(self.classes_file)
        if classes_file.exists():
            with classes_file.open("r") as f:
                self.class_names = [line.strip() for line in f.readlines()]

        logger.info("Backend initialized with model: %s, max_batch_size: %d", self.model_name, self.max_batch_size)

    async def handle_batch(
        self, requests: ImageClassificationRequest | list[ImageClassificationRequest]
    ) -> ImageClassificationResponse | list[ImageClassificationResponse]:
        """Process a batch of image classification requests using batching."""
        if not requests:
            return []

        if isinstance(requests, ImageClassificationRequest):
            logger.info("Single request received, converting to list")
            requests = [requests]

        logger.info("Processing batch of %d requests", len(requests))
        start_time = time.monotonic_ns()

        try:
            batch_tensor = [r.tensor_image for r in requests]
            batch_tensor = torch.cat(batch_tensor, dim=0).cuda()
            logger.info("Batch tensor shape: %s", tuple(batch_tensor.shape))

            # Run inference on the batch
            def generate_batch_predictions():
                with torch.no_grad():
                    output = self.model(batch_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidences, predicted = torch.max(probabilities, 1)
                    return confidences, predicted

            loop = get_or_create_event_loop()
            confidences, predicted = await loop.run_in_executor(None, generate_batch_predictions)

            inference_time = (time.monotonic_ns() - start_time) / 1e9
            logger.info("Batch inference completed in %.2fs", inference_time)

            # Create responses for each request, they have to be in correct order
            responses = []
            for idx, req in enumerate(requests):
                confidence = confidences[idx].item()
                class_id = predicted[idx].item()

                # Get class name
                if self.class_names and class_id < len(self.class_names):
                    prediction = self.class_names[class_id]
                else:
                    prediction = f"class_{class_id}"

                responses.append(
                    ImageClassificationResponse(
                        request_id=req.request_id,
                        prediction=prediction,
                        confidence=confidence,
                        class_id=class_id,
                        inference_time=inference_time,
                    )
                )

            return responses

        except Exception as e:
            logger.exception("Batch processing failed")
            error_responses = []
            for req in requests:
                error_responses.append(ImageClassificationResponse.make_error_response(req.request_id, str(e)))
            return error_responses

    def _decode_image(self, requests: ImageClassificationRequest) -> torch.Tensor:
        """Decode images from request paths."""
        image = Image.open(requests.image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).cuda()

    @dynamo_endpoint(ImageClassificationRequest, ImageClassificationResponse)
    async def classify_image(self, request: ImageClassificationRequest):
        """Classify image with batching."""
        logger.info("Received classification request: %s...", request.request_id)

        # Process through batch handler - will automatically batch requests
        try:
            # decode image should be done before batching, to avoid matching requests with predictions in case of errors
            request.tensor_image = self._decode_image(request)

            response = await self.handle_batch(request)
            if response and response.error is None:
                logger.info("Request completed successfully")
                # Return the first response since we're processing a single request
                yield response.model_dump()
            else:
                logger.error("Request failed: no response")
                yield ImageClassificationResponse.make_error_response(
                    request.request_id, "No response generated"
                ).model_dump()
        except Exception as e:
            logger.error("Request failed: %s", e)
            yield ImageClassificationResponse.make_error_response(request.request_id, str(e)).model_dump()

    async def tune_model(self):
        """Tune the model."""
        if self.tuned_model_path.exists() and not self.force_tune:
            logger.info("Model is already tuned, skipping tuning")
            return

        logger.info("Tuning model...")
        logger.info("  Model name: %s", self.model_name)
        logger.info("  Tuned model path: %s", self.tuned_model_path)
        logger.info("  Image path: %s", self.image_path)
        logger.info("  Max batch size: %s", self.max_batch_size)

        process = await subprocess.create_subprocess_exec(
            sys.executable,
            "-m",
            "resnet.tune",
            "--model-name",
            self.model_name,
            "--tuned-model-path",
            str(self.tuned_model_path),
            "--image-path",
            self.image_path,
            "--max-batch-size",
            str(self.max_batch_size),
        )

        await process.wait()
        if process.returncode != 0:
            logger.error("Tuning model failed with return code %s", process.returncode)
            raise RuntimeError("Tuning model failed")

        logger.info("Tuning model completed")


@dynamo_worker()
async def backend_worker(runtime: DistributedRuntime):
    namespace_name = "resnet"
    component_name = "backend"
    endpoint_name = "classify_image"

    component = runtime.namespace(namespace_name).component(component_name)
    await component.create_service()

    logger.info("Created service %s/%s", namespace_name, component_name)

    endpoint = component.endpoint(endpoint_name)
    lease_id = endpoint.lease_id()
    logger.info("Serving endpoint %s on lease %s", endpoint_name, lease_id)

    backend = ResNetBatchedBackend(_get_config())
    await backend.tune_model()
    await backend.initialize_model()
    await endpoint.serve_endpoint(backend.classify_image)


def _get_config() -> dict:
    with Path(os.environ.get("AITUNE_EXAMPLE_CONFIG_PATH", "config.yaml")).open("r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    uvloop.install()
    asyncio.run(backend_worker())
