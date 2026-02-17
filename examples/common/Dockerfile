# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04

ARG EXAMPLE_NAME
ENV EXAMPLE_NAME=${EXAMPLE_NAME}

ARG AITUNE_VERSION=1.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${AITUNE_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq libucx0 python3-dev python3-pip python3-venv && apt-get clean

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy
RUN uv venv /opt/dynamo/venv --python 3.12

ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# use aitune from local build add to build context "--build-context aitune=../.."
COPY --from=aitune / /opt/aitune

# depth of directory matches repository - to avoid overriding pyproject.toml dependencies paths
COPY . /opt/aitune/dynamo/app

WORKDIR /opt/aitune/dynamo/app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install .[dynamo]

CMD python -m $EXAMPLE_NAME.dynamo.backend
