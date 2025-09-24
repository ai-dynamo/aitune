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
"""AI Dynamo ParakeetRNNT frontend."""

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
    namespace_name = "parakeet_rnnt"
    component_name = "backend"
    endpoint_name = "transcribe_audio"

    endpoint = runtime.namespace(namespace_name).component(component_name).endpoint(endpoint_name)

    client = await endpoint.client()
    await client.wait_for_instances()

    logger.info("Client initialized")

    audio_storage_path = Path("/tmp/audio")
    audio_storage_path.mkdir(parents=True, exist_ok=True)

    @app.post("/transcribe_audio")
    async def transcribe_audio(request_id: Annotated[str, Form()], audio_file: UploadFile):
        """HTTP endpoint for audio transcription with batching.

        Args:
            request_id: User provided request ID
            audio_file: Audio file to transcribe, expects wav format

        Returns:
            StreamingResponse: Streaming response with transcription
        """
        # Read audio file into storage
        logger.info("Frontend received transcription request")

        internal_request_id = "" if request_id is None else request_id
        internal_request_id += "__" + str(uuid.uuid4())

        # store audio file in temp file, expecting wav format for now
        audio_file_path = audio_storage_path / f"{internal_request_id}.wav"
        async with aiofiles.open(audio_file_path, "wb") as f:
            while chunk := await audio_file.read(1024 * 1024):
                await f.write(chunk)

        # Prepare request for backend
        request = {
            "request_id": request_id,
            "audio_path": str(audio_file_path),
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
    frontend_future = server.serve()

    logger.info("Endpoint future completed")
    await frontend_future


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    uvloop.install()

    asyncio.run(frontend_worker())
