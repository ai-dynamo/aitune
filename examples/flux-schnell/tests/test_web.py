# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise browser-facing HTTP requests against a small fake Dynamo service."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from flux_schnell_recipe.deployment.dynamo.web import create_server

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6S8AAAAASUVORK5CYII=")
CFG = {"workload": {"width": 1024, "height": 1024, "num_inference_steps": 4}, "deployment": {"seed": 42}}


class WebTests(unittest.TestCase):
    def start_server(self, server):
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()

        def stop():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.addCleanup(stop)
        return f"http://127.0.0.1:{server.server_port}"

    def setUp(self):
        self.payloads = []
        self.models = [{"id": "flux-schnell-aitune-test"}]
        self.upstream_status = 200
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        test = self

        class Dynamo(BaseHTTPRequestHandler):
            def do_GET(self):
                test.assertEqual(self.path, "/v1/models")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"data": test.models}).encode())

            def do_POST(self):
                test.assertEqual(self.path, "/v1/images/generations")
                test.payloads.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                test.entered.set()
                test.release.wait(timeout=5)
                self.send_response(test.upstream_status)
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"b64_json": base64.b64encode(PNG).decode()}]}).encode())

            def log_message(self, *_):
                pass

        endpoint = self.start_server(ThreadingHTTPServer(("127.0.0.1", 0), Dynamo))
        self.url = self.start_server(create_server(CFG, endpoint, port=0))

    def generate(self, prompt="A lake at sunrise"):
        return urlopen(Request(self.url + "/api/generate", data=json.dumps({"prompt": prompt}).encode(),
                               headers={"Content-Type": "application/json"}), timeout=10)

    def test_page_settings_and_prompt_to_png(self):
        with urlopen(self.url) as response:
            self.assertIn(b'Your prompt', response.read())
        with urlopen(self.url + "/api/config") as response:
            settings = json.load(response)
            self.assertEqual(settings["seed"], 42)
            self.assertEqual(settings["variant"], "aitune")
        with self.generate("  Custom prompt: café by the sea  ") as response:
            self.assertEqual(response.headers["Content-Type"], "image/png")
            self.assertEqual(response.read(), PNG)
        self.assertEqual(self.payloads, [{"model": "flux-schnell-aitune-test", "prompt": "Custom prompt: café by the sea",
                                         "n": 1, "size": "1024x1024", "response_format": "b64_json"}])

    def test_empty_prompt_does_not_reach_service(self):
        with self.assertRaises(HTTPError) as caught:
            self.generate("   ")
        with caught.exception as response:
            self.assertEqual(response.code, 400)
            self.assertIn("Enter a prompt", json.load(response)["error"])
        self.assertEqual(self.payloads, [])

    def test_service_failure_allows_retry(self):
        self.upstream_status = 500
        with self.assertRaises(HTTPError) as caught:
            self.generate()
        with caught.exception as response:
            self.assertEqual(response.code, 502)
            self.assertIn("HTTP 500", json.load(response)["error"])
        self.upstream_status = 200
        with self.generate() as response:
            self.assertEqual(response.read(), PNG)

    def test_wrong_variant_is_not_selected(self):
        self.models = [{"id": "flux-schnell-original-test"}]
        with self.assertRaises(HTTPError) as caught:
            self.generate()
        with caught.exception as response:
            self.assertEqual(response.code, 502)
            self.assertIn("found 0", json.load(response)["error"])
        self.assertEqual(self.payloads, [])

    def test_second_request_is_rejected_while_busy(self):
        self.release.clear()
        responses = []

        def first_request():
            with self.generate() as response:
                responses.append(response.read())

        thread = threading.Thread(target=first_request)
        thread.start()
        try:
            self.assertTrue(self.entered.wait(timeout=5))
            with self.assertRaises(HTTPError) as caught:
                self.generate()
            with caught.exception as response:
                self.assertEqual(response.code, 409)
        finally:
            self.release.set()
            thread.join(timeout=5)
        self.assertEqual(responses, [PNG])


if __name__ == "__main__":
    unittest.main()
