# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test distributed AOT tuning of Transformers native TP with TorchEager.

The pytest parent relaunches this file with torchrun. Each worker re-enters the test through the shared driver and runs
it with rank-local distributed state.
"""

# /// script
# dependencies = ["accelerate", "huggingface-hub", "transformers>=5.1,<6"]
# scope = "always"
# allow_failure = false
# use_gated_hf_token = true
# additional_tags = ["gpu/4"]
# [environment]
# TQDM_DISABLE=1
# ///
import logging
import sys
from pathlib import Path

from aitune.torch.backend import TorchEagerBackend

# Keep the shared driver importable when this file runs as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.utilities.transformers_native_tp import run_aot_native_tp_test


def test_tune_transformers_native_tp_eager_backend() -> None:
    """Tune and run a native-TP model with the eager backend through the explicit tuning flow."""
    run_aot_native_tp_test(
        TorchEagerBackend,
        "transformers-native-tp-eager",
        __file__,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    test_tune_transformers_native_tp_eager_backend()
