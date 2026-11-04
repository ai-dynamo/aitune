# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM nvcr.io/nvidia/pytorch:26.06-py3

COPY dist/*.whl /tmp/aitune-dist/
RUN python -m pip install --break-system-packages pydantic /tmp/aitune-dist/*.whl && \
    rm -rf /tmp/aitune-dist && \
    groupadd --force --gid 1001 runner && useradd --create-home --uid 1001 --gid 1001 runner

# Torch-TensorRT resolves the username for the runner's non-root UID.
USER 1001:1001
