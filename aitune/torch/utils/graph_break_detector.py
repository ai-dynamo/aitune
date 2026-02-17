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
"""Torch eager backend which logs graph breaks."""

import logging

import torch

logger = logging.getLogger(__name__)


class GraphBreakDetector:
    """Graph break detector.

    Graph breaks occur when `torch.dynamo` cannot capture a continuous portion of Python code
    into a single, optimized computation graph (an FX graph). This forces execution back to
    eager mode, interrupting optimization and reducing compilation benefits.

    Causes of Graph Breaks:

    * Data-dependent control flow: `if` statements or loops driven by tensor values.
    * Unsupported Python built-ins/C extensions: Opaque functions `torch.dynamo` can't trace.
    * Dynamic output shapes: Operations like `torch.nonzero()` with value-dependent output shapes.
    * Mutating global state or inputs: Code modifying state outside explicit tensor ops.
    * Tensor subclasses and custom objects: User-defined classes not fully transparent to compiler.
    * Backward hooks/PyTorch internals: Specific PyTorch features interfering with tracing.
    * Unrecognized code patterns/internal errors: Less common, but possible.

    Those reasons can prevent creating correct computational graph. This class uses `torch.compile` to detect
    such graph breaks.

    Note: detections can be flaky in debug mode raising false positives signals.

    """

    def __init__(self):
        """Initializes the graph break detector."""
        self._exceptions = []

    def detect(self, module: torch.nn.Module, args, kwargs):
        """Uses torch dynamo to detect the model i.e. graph breaks."""
        try:
            torch.compile(module, fullgraph=True)(*args, **kwargs)
        except Exception as e:
            logger.debug("Graph break detected: %s", e)
            self._exceptions.append(e)

    def has_graph_breaks(self) -> bool:
        """Returns True if graph breaks were detected during compilation."""
        return len(self._exceptions) > 0
