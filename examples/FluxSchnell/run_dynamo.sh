#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage: ./run_dynamo.sh [--recipe recipes/single-gpu.yaml] [--serve]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec python -m flux_schnell.dynamo.serve "$@"
