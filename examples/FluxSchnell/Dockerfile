# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3
FROM ${BASE_IMAGE}

ENV VIRTUAL_ENV=/workspace/aitune/examples/FluxSchnell/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AITUNE=1.0.0
ENV PYTHONUNBUFFERED=1

# Avoid mixing upgraded PyTorch packages with binary extensions from the base image.
RUN command -v uv || python -m pip install uv
COPY pyproject.toml LICENSE /workspace/aitune/
COPY docs/pypi.rst /workspace/aitune/docs/pypi.rst
COPY aitune /workspace/aitune/aitune
RUN rm -rf /workspace/aitune/examples/FluxSchnell
COPY examples/FluxSchnell /workspace/aitune/examples/FluxSchnell
WORKDIR /workspace/aitune/examples/FluxSchnell
RUN uv sync --locked --extra dynamo
RUN uv pip check --python "${VIRTUAL_ENV}/bin/python"

EXPOSE 8080
CMD ["flux_schnell", "serve"]
