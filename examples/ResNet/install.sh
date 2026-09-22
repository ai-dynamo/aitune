#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

requirements=()
if [[ -n "${TRT_VERSION:-}" ]]; then
  requirements+=("tensorrt==$TRT_VERSION")
fi

if (( ${#requirements[@]} == 0 )); then
  echo "Triton container package versions are unavailable; skipping container-specific installation"
  exit 0
fi

python3 -m pip install --extra-index-url https://pypi.nvidia.com "${requirements[@]}"
