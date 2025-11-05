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
"""AI Dynamo E5Large frontend."""

import asyncio
import json
import logging

import uvloop
from dynamo.runtime import DistributedRuntime, dynamo_worker
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uvicorn import Config, Server

logger = logging.getLogger(__name__)

app = FastAPI()


class EmbeddingRequestModel(BaseModel):
    """HTTP request model for embedding."""

    request_id: str | None = None
    sentence: str


@dynamo_worker()
async def frontend_worker(runtime: DistributedRuntime):
    """Dynamo worker for E5Large frontend service."""
    logging.basicConfig(level=logging.INFO, force=True)

    namespace_name = "e5large"
    component_name = "backend"
    endpoint_name = "embed_sentences"

    endpoint = runtime.namespace(namespace_name).component(component_name).endpoint(endpoint_name)

    client = await endpoint.client()
    await client.wait_for_instances()

    logger.info("Client initialized")

    @app.post("/embed_sentences")
    async def embed_sentences(request: EmbeddingRequestModel):
        """HTTP endpoint for text embedding with batching.

        Args:
            request: Embedding request with sentences

        Returns:
            JSON response with embeddings
        """
        logger.info("Frontend received embedding request")

        # Prepare request for backend
        backend_request = {
            "request_id": request.request_id,
            "sentence": request.sentence,
        }

        stream = await client.generate(json.dumps(backend_request))
        async for response in stream:
            data = response.data()
            logger.info(" .... Response: %s, %s, %s", data["request_id"], data["error"], data["inference_time"])
            return data

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
