# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local prompt UI and same-origin proxy for an already running Dynamo service."""

from __future__ import annotations

import base64
import json
import logging
import secrets
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Config, argument_parser, configure

logger = logging.getLogger(__name__)
MAX_BODY = 64 * 1024


def generate_image(endpoint: str, variant: str, cfg: Config, prompt: str) -> bytes:
    with urlopen(endpoint + "/v1/models", timeout=10) as response:
        models = json.load(response)["data"]
    names = [model["id"] for model in models if model["id"].startswith(f"flux-schnell-{variant}-")]
    if len(names) != 1:
        raise ValueError(f"Expected one running FLUX {variant} service; found {len(names)}. Check run_dynamo.sh.")
    payload = {
        "model": names[0],
        "prompt": prompt,
        "n": 1,
        "size": f"{cfg['workload']['width']}x{cfg['workload']['height']}",
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
    cfg: Config
    endpoint: str
    variant: str
    page: bytes
    generation_lock: threading.Lock


class Handler(BaseHTTPRequestHandler):
    server: WebServer

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
                    "width": self.server.cfg["workload"]["width"],
                    "height": self.server.cfg["workload"]["height"],
                    "steps": self.server.cfg["workload"]["num_inference_steps"],
                    "seed": "random",
                    "variant": self.server.variant,
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
        except (ValueError, UnicodeError) as exc:
            self.error(400, str(exc))
            return
        if not self.server.generation_lock.acquire(blocking=False):
            self.error(409, "An image is already being generated. Try again when it finishes.")
            return
        try:
            image = generate_image(self.server.endpoint, self.server.variant, self.server.cfg, prompt)
        except HTTPError as exc:
            logger.warning("Dynamo request failed: %s", exc)
            self.error(502, f"Image service returned HTTP {exc.code}. Check its backend output for details.")
        except (URLError, OSError) as exc:
            logger.warning("Dynamo connection failed: %s", exc)
            self.error(502, "Cannot reach the image service or the request timed out. Check run_dynamo.sh is ready.")
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Invalid Dynamo response: %s", exc)
            self.error(502, str(exc))
        else:
            self.respond(200, image, "image/png")
        finally:
            self.server.generation_lock.release()

    def read_prompt(self) -> str:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_BODY:
            raise ValueError("Prompt request must be between 1 byte and 64 KiB")
        payload = json.loads(self.rfile.read(length))
        prompt = payload.get("prompt") if isinstance(payload, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a prompt first")
        return prompt.strip()


def create_server(cfg: Config, endpoint: str, variant: str = "aitune", port: int = 8080) -> WebServer:
    server = WebServer(("127.0.0.1", port), Handler)
    server.cfg, server.endpoint, server.variant = cfg, endpoint.rstrip("/"), variant
    server.page = files(__package__).joinpath("web.html").read_bytes()
    server.generation_lock = threading.Lock()
    return server


def run(args: Namespace, cfg: Config) -> None:
    endpoint = args.endpoint or f"http://127.0.0.1:{cfg['deployment']['port']}"
    with create_server(cfg, endpoint, args.variant, args.port) as server:
        print(
            f"Web app: http://127.0.0.1:{server.server_port}; image service: {endpoint}; stop with Ctrl-C", flush=True
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutdown requested; stopping the web app")


def main() -> None:
    parser = argument_parser(__doc__)
    parser.add_argument("--endpoint", help="Dynamo base URL; defaults to the recipe port")
    parser.add_argument("--variant", choices=("original", "aitune"), default="aitune")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    run(args, configure(args))


if __name__ == "__main__":
    main()
