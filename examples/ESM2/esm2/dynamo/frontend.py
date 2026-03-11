# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AI Dynamo FLUX frontend."""

import asyncio
import logging
import uuid

import uvloop
from dynamo.runtime import DistributedRuntime, dynamo_worker
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uvicorn import Config, Server

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG, force=True)

app = FastAPI()


class SequenceGenerationRequest(BaseModel):
    """Request model for sequence generation."""

    request_id: str | None = None
    internal_request_id: str | None = None
    sequence: str


@dynamo_worker()
async def frontend_worker(runtime: DistributedRuntime):
    namespace_name = "esm2"
    component_name = "backend"
    endpoint_name = "generate_sequence"

    endpoint = runtime.namespace(namespace_name).component(component_name).endpoint(endpoint_name)

    client = await endpoint.client()
    await client.wait_for_instances()

    logger.info("Client initialized")

    @app.post("/generate_sequence")
    async def generate_sequence(request: SequenceGenerationRequest):
        """HTTP endpoint for sequence generation with batching.

        Args:
            request: Masked sequence generation request

        Returns:
            StreamingResponse: Streaming response with generated missing protein sequence
        """
        logger.info("Frontend received generation request")

        request.internal_request_id = "" if request.request_id is None else request.request_id
        request.internal_request_id += "__" + str(uuid.uuid4())

        stream = await client.generate(request.model_dump())

        response_data = {}
        async for response in stream:
            logger.info("Response received")
            response_data = response.data()

            return JSONResponse(response_data)

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


if __name__ == "__main__":
    uvloop.install()

    asyncio.run(frontend_worker())
