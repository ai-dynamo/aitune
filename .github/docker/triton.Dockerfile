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

USER runner
