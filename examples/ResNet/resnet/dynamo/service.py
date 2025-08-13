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
"""AI Dynamo service with ResNet model.

This service is run using docker compose, please refer to the README.md file for more details.
"""

import base64
import io
import logging
from pathlib import Path

import torch
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk import DYNAMO_IMAGE, api, depends, endpoint, service
from dynamo.sdk.lib.config import ServiceConfig
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

import aitune.torch as ait
from aitune.torch.config import aitune_cache_dir

from ..model import get_model, get_transform
from ..tune import tune_model

logger = logging.getLogger(__name__)


class ImageRequest(BaseModel):
    """Request model for image inference."""

    image_data: str = Field(description="Base64 encoded image data")


class InferenceResponse(BaseModel):
    """Response model for inference results."""

    prediction: str
    confidence: float
    class_id: int


@service(
    dynamo={"namespace": "inference"},
    resource={"cpu": 2, "memory": "2Gi", "gpu": 1},
    workers=1,
    image=DYNAMO_IMAGE,
)
class ResNetBackend:
    """Backend service for ResNet model inference."""

    def __init__(self) -> None:
        """Initialize the ResNet backend."""
        logger.info("Starting ResNet backend")
        config = ServiceConfig.get_instance()
        self.model_name = config.get("Backend", {}).get("model_name", "resnet50")
        self.image_path = config.get("Backend", {}).get("image_path", "/opt/dynamo/app/dog.webp")
        self.classes_file = config.get("Backend", {}).get("classes_file", "/opt/dynamo/app/imagenet_classes.txt")
        self.pretrained = config.get("Backend", {}).get("pretrained", True)

        tunned_model_path = aitune_cache_dir() / f"{self.model_name}.pt"
        if not tunned_model_path.exists():
            tune_model(self.model_name, self.image_path, tunned_model_path)

        logger.info("Loading tuned model from %s", tunned_model_path)
        self.model = get_model(self.model_name, self.pretrained)
        self.transform = get_transform(self.model)
        ait.load(self.model, tunned_model_path)

        self.class_names = []
        classes_file = Path(self.classes_file)
        if classes_file.exists():
            with classes_file.open("r") as f:
                self.class_names = [line.strip() for line in f.readlines()]

        logger.info("Backend initialized with model: %s", self.model_name)

    @endpoint()
    async def predict(self, req: ImageRequest):
        """Perform image classification inference."""
        logger.info("Backend processing image of size: %s", len(req.image_data))

        # Load and preprocess image
        import torch.nn.functional as F  # noqa: N812

        image_data = base64.b64decode(req.image_data)
        data = io.BytesIO(image_data)

        image = Image.open(data).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).cuda()

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Get class name (assuming ImageNet classes)
        if self.class_names:
            prediction = self.class_names[predicted.item()]
        else:
            prediction = f"class_{predicted.item()}"

        response = InferenceResponse(prediction=prediction, confidence=confidence.item(), class_id=predicted.item())

        logger.info("Backend prediction: %s (confidence: %.3f)", prediction, response.confidence)
        yield response.model_dump_json()


@service(
    dynamo={"namespace": "inference"},
    image=DYNAMO_IMAGE,
)
class ResNetFrontend:
    """Frontend HTTP API for ResNet inference service."""

    backend = depends(ResNetBackend)

    def __init__(self) -> None:
        """Initialize the ResNet frontend."""
        configure_dynamo_logging(service_name="ResNetFrontend")
        logger.info("Starting ResNet frontend")

        config = ServiceConfig.get_instance()
        self.port = config.get("Frontend", {}).get("port", 8000)
        logger.info("Frontend config port: %d", self.port)

    @api()
    async def predict(self, request: ImageRequest):
        """HTTP endpoint for image classification."""
        logger.info("Frontend received prediction request for image of size: %s", len(request.image_data))

        async def content_generator():
            async for response in self.backend.predict(request.model_dump_json()):
                logger.info("Frontend received response: %s", response)
                yield response

            logger.info("Frontend done")

        return StreamingResponse(content_generator())

    @api()
    async def health(self):
        """Health check endpoint."""
        return {"status": "healthy", "service": "ResNet Inference"}
