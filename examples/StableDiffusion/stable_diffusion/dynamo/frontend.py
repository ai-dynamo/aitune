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
"""AI Dynamo Stable Diffusion frontend."""

import asyncio
import logging
import uuid

import uvloop
from dynamo.runtime import DistributedRuntime, dynamo_worker
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from uvicorn import Config, Server

logger = logging.getLogger(__name__)

app = FastAPI()


class ImageGenerationRequest(BaseModel):
    """Request model for image generation."""

    request_id: str | None = None
    internal_request_id: str | None = None
    prompt: str
    height: int = 512
    width: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    max_sequence_length: int = 77


@dynamo_worker()
async def frontend_worker(runtime: DistributedRuntime):
    namespace_name = "stable_diffusion"
    component_name = "backend"
    endpoint_name = "generate_image"

    endpoint = runtime.namespace(namespace_name).component(component_name).endpoint(endpoint_name)

    client = await endpoint.client()
    await client.wait_for_instances()

    logger.info("Client initialized")

    @app.post("/generate_image")
    async def generate_image(request: ImageGenerationRequest):
        """HTTP endpoint for image generation with batching.

        Args:
            request: Image generation request

        Returns:
            StreamingResponse: Streaming response with generated image
        """
        logger.info("Frontend received generation request: %s...", request.prompt[:50])

        request.internal_request_id = "" if request.request_id is None else request.request_id
        request.internal_request_id += "__" + str(uuid.uuid4())

        stream = await client.generate(request.model_dump())

        response_data = {}
        async for response in stream:
            logger.info("Response received")
            response_data = response.data()

        if "image_path" in response_data and response_data["image_path"]:
            image_path = response_data["image_path"]
            return StreamingResponse(
                _get_image_handler(image_path),
                headers={
                    "X-Prompt": response_data["prompt"],
                    "X-Request-ID": response_data["request_id"],
                },
                media_type="image/jpeg",
            )

        if "error" in response_data and response_data["error"]:
            return JSONResponse({"error": response_data["error"]}, status_code=500)

        logger.info("No response received")
        return JSONResponse({"error": "No response received"}, status_code=500)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    server = Server(Config(app=app, host="0.0.0.0", port=8000))
    frontend_future = server.serve()

    logger.info("Endpoint future completed")
    await frontend_future


def _get_image_handler(image_path: str):
    with open(image_path, "rb") as f:
        yield from f


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    uvloop.install()

    asyncio.run(frontend_worker())
