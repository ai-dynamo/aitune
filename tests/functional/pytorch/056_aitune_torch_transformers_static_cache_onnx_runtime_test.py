# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["transformers>=5,<6"]
# docker_image = "nvcr.io/nvidia/pytorch:26.06-py3"
# scope = "always"
# allow_failure = false
#
# [[pip_install]]
# packages = ["onnxruntime-gpu"]
# flags = ["--upgrade", "--pre", "--index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"]
# ///

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aitune.torch.backend import ONNXRuntimeBackend
from tests.utilities.transformers_static_cache_aot import run_static_cache_aot_test


def test_tune_transformers_static_cache_with_onnx_runtime() -> None:
    """Validate the StaticCache lifecycle with ONNX Runtime."""
    run_static_cache_aot_test(ONNXRuntimeBackend())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, force=True)
    test_tune_transformers_static_cache_with_onnx_runtime()
