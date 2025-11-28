# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Configuration for JIT module."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from aitune.torch.backend.backend import Backend
from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend, TensorRTBackendConfig

DEFAULT_JIT_CACHE_DIR = Path.home() / ".cache" / "aitune.jit"


@dataclass
class Config:
    """Configuration for JIT module."""

    dry_run: bool = False  # whether to perform dry-run tuning
    dry_run_failure_probability: float = 0.2  # probability of failure in dry-run mode to imitate tuning failure
    inspect_mode: bool = False  # whether to perform inspect mode

    min_samples: int = 2  # minimum number of samples recorded before tuning
    batch_axis_required: bool = True  # if True, the batch axis must detected in the input data
    max_depth_level: int = 2  # maximum depth of the module hierarchy
    min_parameters: int = 0  # minimum number of parameters to be tuned
    detect_graph_breaks: bool = True  # if True, graph break detection is enabled before tuning
    skip_modules: list[str] = field(default_factory=list)  # list of modules (class names) to skip

    cache_dir: Path = Path(os.environ.get("AITUNE_JIT_CACHE_DIR", DEFAULT_JIT_CACHE_DIR))
    backends: list[Backend] = field(
        default_factory=lambda: [
            TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=True)),
            TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=False)),
        ]
    )  # backends to use for JIT tuning


config = Config()
