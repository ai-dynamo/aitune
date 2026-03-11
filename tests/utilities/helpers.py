# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
