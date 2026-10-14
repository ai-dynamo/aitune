# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3
FROM ${BASE_IMAGE}

ENV VIRTUAL_ENV=/workspace/flux-schnell/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0
ENV WEB_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

RUN test -x "${VIRTUAL_ENV}/bin/python" || python -m venv --system-site-packages "${VIRTUAL_ENV}"
RUN command -v uv || python -m pip install uv
COPY pyproject.toml LICENSE /workspace/aitune/
COPY docs/pypi.rst /workspace/aitune/docs/pypi.rst
COPY aitune /workspace/aitune/aitune
COPY examples/common/pyproject.toml /workspace/aitune/examples/common/pyproject.toml
COPY examples/common/aitune_examples_common /workspace/aitune/examples/common/aitune_examples_common
COPY examples/FluxSchnell /workspace/aitune/examples/FluxSchnell
WORKDIR /workspace/aitune/examples/FluxSchnell
RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" '.[dynamo]'

EXPOSE 8080
CMD ["bash", "run_dynamo.sh", "--serve"]
