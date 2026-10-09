# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local prompt UI and same-origin proxy for an already running Dynamo service."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import logging
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)
MAX_BODY = 64 * 1024


def generate_image(endpoint, variant, cfg, prompt):
    with urlopen(endpoint + "/v1/models", timeout=10) as response:
        models = json.load(response)["data"]
    names = [model["id"] for model in models if model["id"].startswith(f"flux-schnell-{variant}-")]
    if len(names) != 1:
        raise ValueError(f"Expected one running FLUX {variant} service; found {len(names)}. Check recipe serve.")
    payload = {"model": names[0], "prompt": prompt, "n": 1,
               "size": f"{cfg['workload']['width']}x{cfg['workload']['height']}", "response_format": "b64_json"}
    request = Request(endpoint + "/v1/images/generations", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=600) as response:
        data = json.load(response)["data"]
    if len(data) != 1:
        raise ValueError("Service returned an unexpected number of images")
    image = base64.b64decode(data[0]["b64_json"], validate=True)
    if not image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Service did not return a PNG image")
    return image


def create_server(cfg, endpoint, variant="aitune", host="127.0.0.1", port=8080):
    page = files(__package__).joinpath("web.html").read_bytes()
    generation_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def error(self, status, message):
            self.respond(status, json.dumps({"error": message}).encode())

        def do_GET(self):
            if self.path == "/":
                self.respond(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/config":
                self.respond(200, json.dumps({"width": cfg["workload"]["width"],
                    "height": cfg["workload"]["height"], "steps": cfg["workload"]["num_inference_steps"],
                    "seed": cfg["deployment"]["seed"], "variant": variant}).encode())
            else:
                self.error(404, "Not found")

        def do_POST(self):
            if self.path != "/api/generate":
                self.error(404, "Not found")
                return
            if self.headers.get_content_type() != "application/json":
                self.error(415, "Expected application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Prompt request must be between 1 byte and 64 KiB")
                payload = json.loads(self.rfile.read(length))
                prompt = payload.get("prompt") if isinstance(payload, dict) else None
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Enter a prompt first")
            except (ValueError, UnicodeError) as exc:
                self.error(400, str(exc))
                return
            if not generation_lock.acquire(blocking=False):
                self.error(409, "An image is already being generated. Try again when it finishes.")
                return
            try:
                image = generate_image(endpoint.rstrip("/"), variant, cfg, prompt.strip())
            except HTTPError as exc:
                logger.warning("Dynamo request failed: %s", exc)
                self.error(502, f"Image service returned HTTP {exc.code}. Check its worker.log for details.")
            except (URLError, OSError) as exc:
                logger.warning("Dynamo connection failed: %s", exc)
                self.error(502, "Cannot reach the image service or the request timed out. Check recipe serve is ready.")
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Invalid Dynamo response: %s", exc)
                self.error(502, str(exc))
            else:
                self.respond(200, image, "image/png")
            finally:
                generation_lock.release()

    return ThreadingHTTPServer((host, port), Handler)


def run(args, cfg):
    endpoint = args.endpoint or f"http://127.0.0.1:{cfg['deployment']['port']}"
    with create_server(cfg, endpoint, args.variant, args.host, args.port) as server:
        print(f"Web app: http://{args.host}:{server.server_port}; image service: {endpoint}; stop with Ctrl-C", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
