#syntax=docker/dockerfile:1.3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG FROM_IMAGE=nvcr.io/nvidia/tritonserver:26.05-py3
FROM ${FROM_IMAGE}

ARG USER_ID=1001
ARG GROUP_ID=1001

USER root

RUN groupadd --gid ${GROUP_ID} runner && \
    useradd --create-home --uid ${USER_ID} --gid ${GROUP_ID} runner

ENV HOME=/home/runner

RUN python3 -m pip install --extra-index-url https://pypi.nvidia.com \
        "tensorrt==$TRT_VERSION" && \
    python3 -m pip install \
        onnxruntime-gpu \
        --force-reinstall \
        --upgrade \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/

USER runner
