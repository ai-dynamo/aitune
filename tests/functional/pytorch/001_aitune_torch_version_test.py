# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Metadata of the script.
# For more info https://peps.python.org/pep-0723/
#
# Every key is optional and the whole metadata is optional. Only first metadata entry is used.
# /// script
# # Optional, docker image for the job (double commented to disable for this simple test)
# # docker_image = "nvcr.io/nvidia/pytorch:24.12-py3"
#
# # Optional, dependencies installed before the job
# dependencies = ["fire"]
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "always"
#
# # Optional tags which can request particular CI runner with given tags (double commented to disable for this simple test)
# # additional_tags = ["gpu/rtx-a6000"]
#
# # Optional, extra environment variables dict for the job
# [environment]
# EXTRA_ENV = "tests/functional/pytorch"
#
# # Optional, arguments passed to the jobs, you can create any number of argument dicts, each dict will be passed to a separate job
# [[arguments]] # job no. 1
# name = "test1"
#
# [[arguments]] # job no. 2
# name = "test2"
# ///

# Example 2:
# /// script
# scope = "always"
# ///

# Example 3:
# /// script
# dependencies = ["torch"]
# ///

import os

import fire
import torch

import aitune


def main(name: str = ""):
    assert os.environ.get("EXTRA_ENV") == "tests/functional/pytorch"
    assert name in ["test1", "test2"]

    print("Test:", name)  # noqa: T201
    print("Torch: ", torch.__version__)  # noqa: T201
    print("AITune: ", aitune.__version__)  # noqa: T201


if __name__ == "__main__":
    fire.Fire(main)
