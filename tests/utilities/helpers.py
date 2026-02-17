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
"""Helper functions and classes for testing."""

import pytest

from aitune.torch.utils.cuda import is_available as is_cuda_available

requires_cuda = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available")


class TestSink:
    """Sink for capturing output from PatchedModule.print_hierarchy.

    This class provides a simple interface for capturing text output
    that can be used to test print statements and other text-based
    output in tests.
    """

    def __init__(self):
        """Initialize the TestSink with an empty output list."""
        self.output = []

    def write(self, text):
        """Write text to the sink.

        Args:
            text (str): The text to append to the output list.
        """
        self.output.append(text)
