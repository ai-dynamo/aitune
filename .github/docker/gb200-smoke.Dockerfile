# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM nvcr.io/nvidia/pytorch:26.01-py3

# Torch-TensorRT resolves the username for the runner's non-root UID.
RUN groupadd --force --gid 1001 runner && useradd --create-home --uid 1001 --gid 1001 runner
USER 1001:1001
