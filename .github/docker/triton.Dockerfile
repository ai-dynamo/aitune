#syntax=docker/dockerfile:1.3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG FROM_IMAGE=nvcr.io/nvidia/tritonserver:26.05-py3
FROM ${FROM_IMAGE}

ARG USER_ID=1001
ARG GROUP_ID=1001
ARG AITUNE_EXTRAS

USER root

RUN python3 -m pip install --extra-index-url https://pypi.nvidia.com "tensorrt==$TRT_VERSION"

RUN --mount=type=bind,source=dist,target=/aitune_dist \
    if [ -z "$AITUNE_EXTRAS" ]; then \
        echo "AITUNE_EXTRAS must match the Triton image's PyTorch backend" >&2; \
        exit 1; \
    fi && \
    wheel="$(realpath /aitune_dist/*.whl)" && \
    python3 -m pip install --extra-index-url https://pypi.nvidia.com \
        "${wheel}[${AITUNE_EXTRAS}]" && \
    onnxruntime_wheel_dir="$(mktemp -d)" && \
    python3 -m pip download \
        --no-deps \
        --dest "$onnxruntime_wheel_dir" \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \
        onnxruntime-gpu && \
    python3 -m pip install --force-reinstall "$onnxruntime_wheel_dir"/onnxruntime_gpu-*.whl && \
    rm -r -- "$onnxruntime_wheel_dir"

RUN groupadd --gid ${GROUP_ID} runner && \
    useradd --create-home --uid ${USER_ID} --gid ${GROUP_ID} runner

ENV HOME=/home/runner

USER runner
