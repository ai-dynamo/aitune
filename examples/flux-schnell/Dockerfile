# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Version-pinned candidate environment; record the built image digest during GPU evaluation.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3
FROM ${BASE_IMAGE}
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/
ARG EXAMPLE_DIR=examples/flux-schnell
COPY ${EXAMPLE_DIR} /workspace/flux-schnell
WORKDIR /workspace/flux-schnell
# Isolate the locked stack from packages already installed in the NGC image.
ENV UV_LINK_MODE=copy UV_PYTHON=3.12 HF_HOME=/cache/huggingface TZ=UTC
RUN uv sync --frozen --no-dev
CMD ["bash"]
