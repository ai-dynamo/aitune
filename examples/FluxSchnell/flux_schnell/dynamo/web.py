# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt UI and same-origin proxy for this example's Dynamo deployment."""

from __future__ import annotations

import base64
import io
import json
import logging
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel
from typing_extensions import Buffer

from ..config import Recipe

logger = logging.getLogger(__name__)
MAX_BODY = 64 * 1024
READ_TIMEOUT_S = 10.0
MAX_CONNECTIONS = 32


class _DeadlineReader(io.RawIOBase):
    """Bound a whole read phase, including clients that keep sending small chunks."""

    def __init__(self, connection: socket.socket) -> None:
        super().__init__()
        self.connection = connection
        self.deadline = 0.0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Request read deadline exceeded")
        self.connection.settimeout(remaining)
        return self.connection.recv_into(buffer)


class WebConfig(BaseModel):
    model_name: str
    width: int
    height: int
    steps: int


def generate_image(endpoint: str, cfg: WebConfig, prompt: str) -> bytes:
    with urlopen(endpoint + "/v1/models", timeout=10) as response:
        models = json.load(response)["data"]
    names = [model["id"] for model in models if model["id"] == cfg.model_name]
    if len(names) != 1:
        raise ValueError(f"Expected model {cfg.model_name} in Dynamo; found {len(names)}. Check flux_schnell serve.")
    payload = {
        "model": names[0],
        "prompt": prompt,
        "n": 1,
        "size": f"{cfg.width}x{cfg.height}",
        "response_format": "b64_json",
        "nvext": {"seed": secrets.randbits(32)},
    }
    request = Request(
        endpoint + "/v1/images/generations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=600) as response:
        data = json.load(response)["data"]
    if len(data) != 1:
        raise ValueError("Service returned an unexpected number of images")
    image = base64.b64decode(data[0]["b64_json"], validate=True)
    if not image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Service did not return a PNG image")
    return image


class WebServer(ThreadingHTTPServer):
    cfg: WebConfig
    endpoint: str
    page: bytes
    generation_lock: threading.Lock

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler]) -> None:
        self._connection_slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(server_address, handler_class)

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        if not self._connection_slots.acquire(blocking=False):
            logger.warning("UI connection limit reached; closing connection from %s", client_address)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._connection_slots.release()
            logger.exception("Failed to start UI request handler for %s", client_address)
            raise

    def process_request_thread(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


class Handler(BaseHTTPRequestHandler):
    server: WebServer
    _reader: _DeadlineReader

    def setup(self) -> None:
        super().setup()
        self.rfile.close()
        self._reader = _DeadlineReader(self.connection)
        self.rfile = io.BufferedReader(self._reader)

    def handle_one_request(self) -> None:
        self._reader.deadline = time.monotonic() + READ_TIMEOUT_S
        super().handle_one_request()

    def parse_request(self) -> bool:
        try:
            return super().parse_request()
        finally:
            self.connection.settimeout(READ_TIMEOUT_S)

    def respond(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def error(self, status: int, message: str) -> None:
        self.respond(status, json.dumps({"error": message}).encode())

    def do_GET(self) -> None:
        if self.path == "/":
            self.respond(200, self.server.page, "text/html; charset=utf-8")
        elif self.path == "/api/config":
            self.respond(
                200,
                json.dumps({
                    "width": self.server.cfg.width,
                    "height": self.server.cfg.height,
                    "steps": self.server.cfg.steps,
                    "seed": "random",
                    "variant": "aitune",
                }).encode(),
            )
        else:
            self.error(404, "Not found")

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self.error(404, "Not found")
            return
        if self.headers.get_content_type() != "application/json":
            self.error(415, "Expected application/json")
            return
        try:
            prompt = self.read_prompt()
        except TimeoutError as exc:
            logger.warning("Prompt body read timed out for %s: %s", self.client_address, exc)
            self.close_connection = True
            self.error(408, "Timed out reading prompt request")
            return
        except (ValueError, UnicodeError) as exc:
            self.error(400, str(exc))
            return
        if not self.server.generation_lock.acquire(blocking=False):
            self.error(409, "An image is already being generated. Try again when it finishes.")
            return
        try:
            image = generate_image(self.server.endpoint, self.server.cfg, prompt)
        except HTTPError as exc:
            logger.warning("Dynamo request failed: %s", exc)
            self.error(502, f"Image service returned HTTP {exc.code}. Check its backend output for details.")
        except (URLError, OSError) as exc:
            logger.warning("Dynamo connection failed: %s", exc)
            self.error(
                502, "Cannot reach the image service or the request timed out. Check flux_schnell serve is ready."
            )
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Invalid Dynamo response: %s", exc)
            self.error(502, str(exc))
        else:
            self.respond(200, image, "image/png")
        finally:
            self.server.generation_lock.release()

    def read_prompt(self) -> str:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Invalid Content-Length header") from None
        if not 0 < length <= MAX_BODY:
            raise ValueError("Prompt request must be between 1 byte and 64 KiB")
        self._reader.deadline = time.monotonic() + READ_TIMEOUT_S
        try:
            body = self.rfile.read(length)
        finally:
            self.connection.settimeout(READ_TIMEOUT_S)
        if len(body) != length:
            raise ValueError("Incomplete request body")
        payload = json.loads(body)
        prompt = payload.get("prompt") if isinstance(payload, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a prompt first")
        return prompt.strip()


def run(recipe: Recipe) -> None:
    deployment, workload = recipe.deployment, recipe.workload
    with WebServer((deployment.web_host, deployment.web_port), Handler) as server:
        server.cfg = WebConfig(
            model_name=recipe.model.base_model,
            width=workload.width,
            height=workload.height,
            steps=workload.num_inference_steps,
        )
        server.endpoint = recipe.endpoint
        server.page = Path(__file__).with_name("web.html").read_bytes()
        server.generation_lock = threading.Lock()
        print(
            f"Web app: http://{deployment.web_host}:{server.server_port}; image service: {recipe.endpoint}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutdown requested; stopping the web app")
