# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["ai-dynamo<2.0.0", "openai", "aitune", "sentence-transformers"]
# use_gated_hf_token = true
# [environment]
# DYN_DISCOVERY_BACKEND = "file"
# DYN_EVENT_PLANE = "zmq"
# DYN_REQUEST_PLANE = "tcp"
# DYN_ROUTER_USE_KV_EVENTS = "false"
# AUTOWRAPT_BOOTSTRAP = "aitune_enable_jit_tuning"
# AITUNE_CONSOLE_OUTPUT=1
# ///
"""JIT-tuned Dynamo worker end-to-end integration test.

Starts a real Dynamo HTTP frontend and a JIT-tuned E5Large embedding backend,
sends one OpenAI-compatible embedding request, asserts the response shape,
then kills all subprocesses.

Run with:
    uv run tests/functional/dynamo/002_jit_worker.py
"""

import multiprocessing
import subprocess
import sys
import time
import urllib.request

import aitune.dynamo as dyn

_MODEL = "intfloat/e5-large-v2"
_PORT = 8000
_BASE_URL = f"http://localhost:{_PORT}/v1"


def run_backend() -> None:
    """Entry point for the JIT-tuned backend multiprocessing.Process."""
    import logging

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    import aitune.torch as ait
    import aitune.torch.jit.enable  # noqa: F401

    logging.basicConfig(level=logging.DEBUG, force=True)

    # Keeps JIT-compiled sub-modules on CPU between requests.
    # Required because SentenceTransformer overrides .device and calls
    # .to(self.device) in encode(), which breaks if sub-modules are on meta.
    ait.config.device_after_tuning = "cpu"

    model = SentenceTransformer(_MODEL, device="cuda")
    model.eval()

    with torch.no_grad():
        model.encode(["hello world"])  # warmup 1
        model.encode(["hello world"])  # warmup 2 — JIT tuning fires here

    def embed(request) -> np.ndarray:
        sentences = request.input if isinstance(request.input, list) else [request.input]
        return model.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
            device="cuda",
        )

    config = dyn.DynamoWorkerConfig(type="embedding", model_path=_MODEL)
    dyn.dynamo_worker(embed, config)


def _poll(url: str, match: str, retries: int = 40, delay: float = 10) -> None:
    """Poll *url* until *match* appears in the response body.

    Args:
        url: HTTP URL to GET.
        match: Substring to look for in the response body.
        retries: Maximum number of attempts.
        delay: Seconds between attempts.

    Raises:
        RuntimeError: If *match* is not found within *retries* attempts.
    """
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if match in resp.read().decode():
                    return
        except OSError:
            pass
        if attempt < retries:
            print(f"  waiting for {match!r} in {url} (attempt {attempt}/{retries})...")
            time.sleep(delay)
    raise RuntimeError(f"Timed out waiting for {match!r} to appear in {url}")


def main() -> None:
    """Run the full e2e flow: frontend + JIT-tuned backend + one embedding request."""
    frontend = subprocess.Popen(
        [sys.executable, "-m", "dynamo.frontend", "--http-port", str(_PORT)],
    )
    ctx = multiprocessing.get_context("spawn")
    backend = ctx.Process(target=run_backend, daemon=True)
    backend.start()

    try:
        print("Waiting for Dynamo endpoint to register...")
        _poll(f"http://localhost:{_PORT}/health", "dyn://aitune.backend.generate")

        print("Waiting for model to appear in /v1/models...")
        _poll(f"http://localhost:{_PORT}/v1/models", _MODEL)

        from openai import OpenAI

        client = OpenAI(base_url=_BASE_URL, api_key="unused")
        response = client.embeddings.create(model=_MODEL, input="hello world")

        assert len(response.data) == 1, f"Expected 1 embedding, got {len(response.data)}"
        assert response.data[0].object == "embedding", f"Unexpected object type: {response.data[0].object}"
        assert len(response.data[0].embedding) == 1024, f"Expected 1024 dims, got {len(response.data[0].embedding)}"

        print("OK — JIT-tuned embedding shape (1, 1024)")
    finally:
        backend.kill()
        backend.join(timeout=5)
        frontend.kill()
        frontend.wait()


if __name__ == "__main__":
    main()
