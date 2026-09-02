# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate licenses for packages."""

# /// script
# dependencies = ["toml", "pip-licenses"]
# ///

import argparse
import logging
import pathlib
import subprocess
import sys

import toml

LOGGER = logging.getLogger(__name__)

VALID_LICENSES = (
    "MIT",
    "MIT License",
    "BSD",
    "BSD License",
    "BSD-3-Clause",  # Used by protobuf:3.20.0
    "3-Clause BSD License",
    "Apache Software License",
    "Apache License v2.0",
    "Apache 2.0",
    "Apache-2.0",
    "Apache",
    "Apache-2.0 OR BSD-2-Clause",
    "ISC License (ISCL)",
    "ISC",
    "CMU License (MIT-CMU)",
    "MIT-CMU",
    "Python Software Foundation License",
    "Apache 2.0 License",
    "MIT OR Apache-2.0",
    "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",  # Used by numpy:2.2.x
    "Apache-2.0 WITH LLVM-exception",  # Used by nvtx:0.2.15
)

KNOWN_LICENSES = {
    "dataclasses": "Apache Software License",
    "tensorrt": "NVIDIA Proprietary",
    "tensorrt-cu12": "NVIDIA Proprietary",
    "tensorrt-cu13": "NVIDIA Proprietary",
    "nvidia-cudnn-cu12": "NVIDIA Proprietary",
    "cuda-bindings": "Apache-2.0 license",  # https://github.com/NVIDIA/cuda-python/blob/main/LICENSE
    "torchao": "BSD-3-Clause license",  # https://github.com/pytorch/ao/blob/main/LICENSE
    "pillow": "MIT-CMU",
    "tox-uv": "MIT",
    "pdbpp": "BSD-3-Clause license",
    "ipykernel": "BSD-3-Clause license",
    "build": "MIT License",
    "sentencepiece": "Apache-2.0 license",
    "twine": "Apache-2.0 license",
    "wrapt": "BSD-2-Clause license",  # https://github.com/GrahamDumpleton/wrapt/blob/develop/LICENSE
    "torch": "Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT",
}

PROJECT_NAME = "aitune"


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pyproject-file",
        help="Path from which files should be checked",
        default="pyproject.toml",
    )
    parser.add_argument("--extras", required=False, default="")
    args = parser.parse_args(argv)

    is_valid = _validate_licenses(args)

    if not is_valid:
        LOGGER.error("==============================")
        LOGGER.error("Licenses for packages failed. Please check the output above.")
        LOGGER.error("==============================")
        return 1
    return 0


def _validate_licenses(args):
    if not pathlib.Path(args.pyproject_file).is_file():
        raise ValueError(f"Unable to read {args.pyproject_file}")

    pyproject_data = toml.load(args.pyproject_file)

    packages = pyproject_data["project"]["dependencies"]
    for dependencies in pyproject_data["project"].get("optional-dependencies", {}).values():
        packages += [dependency for dependency in dependencies if not dependency.startswith(f"{PROJECT_NAME}[")]

    package_names = [_get_package_name(package) for package in packages]
    package_names = [package for package in package_names if package not in KNOWN_LICENSES]

    valid_licenses = ";".join(VALID_LICENSES)

    LOGGER.warning("Known licenses:")
    for package, known_license in KNOWN_LICENSES.items():
        LOGGER.warning("  %s: %s", package, known_license)

    LOGGER.warning("Checking licenses for %d packages", len(package_names))

    command = [
        "pip-licenses",
        "--allow-only",
        valid_licenses,
        "--packages",
        *package_names,
    ]
    print(" ".join(command))  # noqa: T201
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    )
    process_output = ""
    while True:
        output_chunk = process.stdout.readline()
        if output_chunk == "" and process.poll() is not None:
            break
        if output_chunk:
            print(output_chunk, end="")  # noqa: T201
            process_output += output_chunk

    result = process.poll()

    return result == 0


def _get_package_name(package):
    if ";" in package:
        package = package.split(";")[0]

    if "==" in package:
        package_name = package.split("==")[0]
    elif ">=" in package:
        package_name = package.split(">=")[0]
    elif ">" in package:
        package_name = package.split(">")[0]
    elif "<" in package:
        package_name = package.split("<")[0]
    elif "<=" in package:
        package_name = package.split("<=")[0]
    elif "@" in package:
        package_name = package.split("@")[0]
    elif "~=" in package:
        package_name = package.split("~=")[0]
    else:
        package_name = package

    # if extras defined
    if "[" in package_name:
        package_name = package_name.split("[")[0]

    package_name = package_name.strip()

    return package_name


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
