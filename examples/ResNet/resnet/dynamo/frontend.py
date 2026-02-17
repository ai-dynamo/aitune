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
"""AI Dynamo ResNet frontend."""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Annotated

import aiofiles
import uvloop
from dynamo.runtime import DistributedRuntime, dynamo_worker
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import JSONResponse
from uvicorn import Config, Server

logger = logging.getLogger(__name__)

app = FastAPI()


@dynamo_worker()
async def frontend_worker(runtime: DistributedRuntime):
    logging.basicConfig(level=logging.INFO, force=True)

    namespace_name = "resnet"
    component_name = "backend"
    endpoint_name = "classify_image"

    endpoint = runtime.namespace(namespace_name).component(component_name).endpoint(endpoint_name)

    client = await endpoint.client()
    await client.wait_for_instances()

    logger.info("Client initialized")

    image_storage_path = Path("/tmp/images")
    image_storage_path.mkdir(parents=True, exist_ok=True)

    @app.post("/classify_image")
    async def classify_image(request_id: Annotated[str, Form()], image_file: UploadFile):
        """HTTP endpoint for image classification with batching.

        Args:
            request_id: User provided request ID
            image_file: Image file to classify

        Returns:
            JSON response with classification results
        """
        # Read image file into storage
        logger.info("Frontend received classification request")

        internal_request_id = "" if request_id is None else request_id
        internal_request_id += "__" + str(uuid.uuid4())

        # Determine file extension from content type or filename
        content_type = image_file.content_type or ""
        if "jpeg" in content_type or "jpg" in content_type:
            extension = ".jpg"
        elif "png" in content_type:
            extension = ".png"
        elif "webp" in content_type:
            extension = ".webp"
        else:
            # Try to get extension from filename
            filename = image_file.filename or ""
            if "." in filename:
                extension = "." + filename.split(".")[-1]
            else:
                extension = ".jpg"  # default

        # Store image file in temp file
        image_file_path = image_storage_path / f"{internal_request_id}{extension}"
        async with aiofiles.open(image_file_path, "wb") as f:
            while chunk := await image_file.read(1024 * 1024):
                await f.write(chunk)

        # Prepare request for backend
        request = {
            "request_id": request_id,
            "internal_request_id": internal_request_id,
            "image_path": str(image_file_path),
        }

        stream = await client.generate(json.dumps(request))
        async for response in stream:
            logger.info(" .... Response: %s", response.data())
            return response.data()

        logger.info("No response received")
        return JSONResponse({"error": "No response received"}, status_code=500)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    server = Server(Config(app=app, host="0.0.0.0", port=8000))
    await server.serve()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(frontend_worker())
